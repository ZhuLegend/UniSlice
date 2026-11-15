import socket
import json
import tempfile
import tomllib
import unittest
from unittest.mock import Mock, patch
from pathlib import Path

from run import (allocation_plan_identity, code_identity, ROOT, ensure_ports_free,
                 http_request_timeout_seconds, resolved_allocation, scheduler_config,
                 sender_config, single_phase_workload, validate_allocation_plan,
                 prediction_csv_paths, validate_replay, wait_measurement_barrier)
from monitor import descendants, ProcessMonitor


class SchedulerConfigurationTest(unittest.TestCase):
    def test_prediction_csvs_do_not_change_execution_configuration(self):
        original = json.dumps(self.config, sort_keys=True)
        path = ROOT/'resource/simulation/experiments/e1/configs/common_08.json'
        paths = prediction_csv_paths(path, self.config)
        self.assertEqual(set(paths), {'nvidia_a100', 'nvidia_a100_half'})
        self.assertEqual(paths['nvidia_a100'].name, 'latency-0.8.csv')
        self.assertEqual(json.dumps(self.config, sort_keys=True), original)
        self.assertNotIn('latency_prediction_scales', tomllib.loads(scheduler_config(self.config, 12.5)))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'map.json'
            for bad in [{'missing': 'csv'}, {'nvidia_a100': 0.8}, {'nvidia_a100': 'absent.csv'}]:
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError): prediction_csv_paths(path, self.config)

    def test_sender_accepts_variable_request_batches(self):
        config = {'models': ['alexnet']}
        output = tomllib.loads(sender_config(config, 'http://localhost', Path('/tmp'), 100, 100, (1,32), 60))
        self.assertEqual(output['models'][0]['batch_size_range'], [1,32])
        self.assertEqual(output['models'][0]['rate_range'], [100,100])

    def test_memory_guard_stops_its_processes_and_preserves_evidence(self):
        process = Mock(pid=123)
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            monitor = ProcessMonitor(root/'host.jsonl', [process], max_rss_kib=100)
            monitor.stopped.set()  # One sample, without a background thread.
            with patch('monitor.subprocess.check_output', return_value='123 1 101 0.0\n'), patch('monitor.os.getpid', return_value=999):
                monitor.run()
            self.assertTrue(monitor.limit_exceeded.is_set())
            process.terminate.assert_called_once()
            self.assertEqual(json.loads((root/'memory_guard.json').read_text())['rss_kib'], 101)
    def test_http_deadline_allows_application_timeout_response(self):
        self.assertEqual(http_request_timeout_seconds({}), 310)
        self.assertEqual(http_request_timeout_seconds({'inference_timeout_ms':1001}), 12)
    def test_process_sampler_only_includes_the_simulation_tree(self):
        rows = {1: {'ppid': 0}, 2: {'ppid': 1}, 3: {'ppid': 2}, 4: {'ppid': 1}}
        self.assertEqual(descendants(rows, [2]), {2, 3})

    def setUp(self):
        self.config = json.loads((ROOT/'resource/simulation/paper-a100.json').read_text())

    def test_physical_nodes_share_two_experiment_pools(self):
        generated = tomllib.loads(scheduler_config(self.config, 12.5))
        source = tomllib.loads((ROOT/'scripts/exper_config/4nv+4nv/scheduler.toml').read_text())
        self.assertEqual(generated['batch_extractor_strategy'], source['batch_extractor_strategy'])
        self.assertEqual(len(generated['remote_nodes']), 12)
        self.assertEqual(set(generated['device_combinations']), {'nvidia_a100', 'nvidia_a100_half'})
        for template, count in [('nvidia_a100', 4), ('nvidia_a100_half', 8)]:
            self.assertEqual(sum(n['template_name'] == template for n in generated['remote_nodes'].values()), count)
            self.assertEqual(generated['device_combinations'][template]['config'], source['device_combinations'][template]['config'])

    def test_rejects_different_capacities_in_same_pool(self):
        self.config['nodes'][-1]['template_name'] = 'nvidia_a100'
        with self.assertRaisesRegex(ValueError, 'inconsistent device template'):
            scheduler_config(self.config, 12.5)

    def test_policy_is_read_from_source_including_nested_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'policy.toml'
            source.write_text('batch_extractor_strategy = { type="TimeoutProbabilitySplitAndSatisfy", timeout={secs=1,nanos=42}, inner={threshold=0.25,inner={}} }')
            self.config['batching_config'] = str(source)
            generated = tomllib.loads(scheduler_config(self.config, 12.5))
            self.assertEqual(generated['batch_extractor_strategy'], tomllib.loads(source.read_text())['batch_extractor_strategy'])


class CodeIdentityTest(unittest.TestCase):
    def test_dirty_source_and_rebuilt_binary_cannot_reuse_an_old_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'crates/demo').mkdir(parents=True)
            (root/'target/debug').mkdir(parents=True)
            source = root/'crates/demo/lib.rs'
            source.write_text('first edit')
            for name in ('mock_virtual', 'mock_vgpu', 'scheduler_daemon', 'task_sender'):
                (root/'target/debug'/name).write_bytes(b'first build')
            original = code_identity(root, root/'target/debug')
            source.write_text('second edit')
            edited = code_identity(root, root/'target/debug')
            self.assertNotEqual(original['source_sha256'], edited['source_sha256'])
            self.assertEqual(original['binary_sha256'], edited['binary_sha256'])
            (root/'target/debug/task_sender').write_bytes(b'second build')
            rebuilt = code_identity(root, root/'target/debug')
            self.assertNotEqual(edited['binary_sha256'], rebuilt['binary_sha256'])


class PortCheckTest(unittest.TestCase):
    def test_rejects_active_listener_and_allows_restart_after_connection(self):
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(('127.0.0.1', 0))
            port = server.getsockname()[1]
            server.listen(1)
            with self.assertRaises(OSError):
                ensure_ports_free([port])
            with socket.create_connection(('127.0.0.1', port)) as client:
                peer, _ = server.accept()
                with peer:
                    peer.shutdown(socket.SHUT_WR)
                self.assertEqual(client.recv(1), b'')
        # The old server may still have TIME_WAIT connections, but no listener.
        ensure_ports_free([port])


class ExperimentInputTest(unittest.TestCase):
    def test_skewed_rates_initialize_the_scheduler_and_sender_consistently(self):
        rates = {'resnet50':75,'vgg19':150}
        workload,replay = single_phase_workload(list(rates),rates,30,20260923,(1,32))
        self.assertEqual(workload['initial_rates'],rates)
        self.assertEqual(workload['phases'][0]['rates'],rates)
        config = {'models':list(rates)}
        sender = tomllib.loads(sender_config(config,'http://localhost','unused',rates,rates,(1,32),30))
        self.assertEqual({model['name']:model['rate_range'] for model in sender['models']},
                         {model:[value,value] for model,value in rates.items()})
        samples = {model:sum(batch for _,name,batch in replay if name==model) for model in rates}
        self.assertGreater(samples['vgg19'],samples['resnet50'])
        for bad in [{'resnet50':75},{'resnet50':True,'vgg19':150},{'resnet50':75,'vgg19':0}]:
            with self.assertRaises(ValueError):single_phase_workload(list(rates),bad,30,1,(1,32))

    def test_single_phase_has_reproducible_precise_input_and_samples_rate(self):
        workload, replay = single_phase_workload(['resnet50','vgg19'],100,30,20260923,(1,32))
        self.assertEqual((workload,replay),single_phase_workload(['vgg19','resnet50'],100,30,20260923,(1,32)))
        self.assertEqual(len(workload['phases']),1)
        self.assertEqual(workload['initial_rates'],{'resnet50':100,'vgg19':100})
        self.assertTrue(all(0 <= t < 30 and 1 <= batch <= 32 for t,_,batch in replay))
        self.assertGreater(len({batch for _,_,batch in replay}),1)

    def test_replay_rejects_invalid_models_batches_order_and_window(self):
        for replay in [[],[[0,'unknown',1]],[[0,'resnet50',True]],[[0,'resnet50',33]],
                       [[30,'resnet50',1]],[[float('nan'),'resnet50',1]],
                       [[2,'resnet50',1],[1,'resnet50',1]]]:
            with self.assertRaises(ValueError): validate_replay(replay,['resnet50'],30,(1,32))
        replay=[[0,'resnet50',1],[29.999,'resnet50',32]]
        self.assertIs(validate_replay(replay,['resnet50'],30,(1,32)),replay)

    def test_resolved_plan_must_match_input_but_placement_order_is_not_identity(self):
        config={'models':['resnet50','vgg19'],'nodes':[{'template_name':'parent_full','share_targets':['1g.10gb']}]}
        plan={'schema_version':1,'placements':[
            {'template_name':'parent_full','model':model,'slice':'1g.10gb','count':count}
            for model,count in [('resnet50',3),('vgg19',4)]]}
        validate_allocation_plan(plan,config)
        reversed_plan=dict(plan,placements=list(reversed(plan['placements'])))
        self.assertEqual(allocation_plan_identity(plan),allocation_plan_identity(reversed_plan))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'resolved.json'
            path.write_text(json.dumps({'mode':'fixed','applied':True,'plan':reversed_plan}))
            self.assertEqual(resolved_allocation(path,'fixed',config,plan)['plan'],reversed_plan)
            different=json.loads(json.dumps(plan)); different['placements'][0]['count']=2
            with self.assertRaises(ValueError): resolved_allocation(path,'fixed',config,different)
            with self.assertRaises(ValueError): resolved_allocation(path,'frozen_greedy',config,plan)
        bad=json.loads(json.dumps(plan)); bad['placements'][0]['count']=True
        with self.assertRaises(ValueError): validate_allocation_plan(bad,config)

    def test_barrier_publishes_ready_before_observing_release_and_rejects_stale_file(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory); release=out/'release.json'
            def wait(check, processes, timeout, label):
                self.assertEqual(json.loads((out/'calibration-ready.json').read_text()),{'runner_pid':12})
                self.assertFalse(check())
                release.write_text('{"released_at_ms":123}')
                self.assertTrue(check())
            with patch('run.wait_until',side_effect=wait):
                wait_measurement_barrier(release,out,[],2,{'runner_pid':12})
            self.assertEqual(json.loads((out/'calibration-release.json').read_text())['release'],{'released_at_ms':123})
            with self.assertRaises(ValueError): wait_measurement_barrier(release,out,[],2,{})


if __name__ == '__main__':
    unittest.main()
