import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import uuid

from calibrate_parallel import (OwnedMonitor, batch_distribution, cpu_seconds, fixture,
                                load_collector, paired_drift, slot_configuration, slot_report)
from run import ROOT, dump


class CalibrationTest(unittest.TestCase):
    def test_fixture_retains_profile_and_lifecycle_with_seven_small_replicas(self):
        path=ROOT/'resource/simulation/paper-a100.json'
        original=json.loads(path.read_text())
        config,plan=fixture(path)
        self.assertEqual(config['models'],['resnet50','vgg19'])
        self.assertEqual(len(config['nodes']),1)
        self.assertEqual(config['nodes'][0]['gpus'],1)
        self.assertEqual(sum(row['count'] for row in plan['placements']),7)
        for key in ['create_delay_ms','destroy_delay_ms','model_lifecycle']:
            self.assertEqual(config[key],original[key])
        self.assertEqual(Path(config['latency_csv']),(path.parent/original['latency_csv']).resolve())

    def test_owned_guard_excludes_unrelated_process_and_tracks_only_identical_orphan_pid(self):
        proc=Mock(pid=10); proc.poll.return_value=None
        rows={pid:{'ppid':ppid,'rss_kib':rss,'ps_cpu_percent':1,'cpu_seconds':2,'started':'original'}
              for pid,ppid,rss in [(10,1,50),(11,10,60),(99,1,100000)]}
        with tempfile.TemporaryDirectory() as directory:
            monitor=OwnedMonitor(Path(directory)/'host.jsonl',[proc],100)
            with patch('calibrate_parallel.process_rows',return_value=rows),patch('calibrate_parallel.host_pressure',return_value={}):
                self.assertEqual(monitor.sample()['rss_kib'],110)
                proc.terminate.assert_called_once()
                self.assertEqual(set(monitor.seen),{10,11})
                proc.poll.return_value=0
                orphan={11:{**rows[11],'ppid':1},99:rows[99]}
                with patch('calibrate_parallel.process_rows',return_value=orphan):
                    self.assertEqual(set(monitor.remaining()),{'11'})
                with patch('calibrate_parallel.process_rows',return_value={11:{**rows[11],'started':'reused'}}):
                    self.assertEqual(monitor.remaining(),{})
            stats=monitor.close()
            self.assertEqual(stats['peak_owned_rss_kib'],110)
            self.assertIsNotNone(stats['memory_guard'])

    def test_mixed_fixture_keeps_parent_budgets_and_disjoint_three_port_slots(self):
        config,plan=fixture(ROOT/'resource/simulation/paper-a100.json','mixed-four')
        self.assertEqual(config['models'],['resnet50','vgg19','resnet18','vgg16'])
        self.assertEqual(len(config['nodes']),2)
        counts={template:sum(row['count'] for row in plan['placements'] if row['template_name']==template)
                for template in ['parent_full','parent_half']}
        self.assertEqual(counts,{'parent_full':7,'parent_half':3})
        self.assertTrue(all(row['count']>0 for row in plan['placements']))
        ports=[]
        for index in range(4):
            slot,daemon=slot_configuration(config,15300,index)
            ports += [daemon]+[node['port'] for node in slot['nodes']]
        self.assertEqual(ports,list(range(15300,15312)))

    def test_cpu_time_and_batch_statistics_have_explicit_observation_units(self):
        self.assertAlmostEqual(cpu_seconds('02:03.5'),123.5)
        self.assertEqual(cpu_seconds('1-02:03:04'),93784)
        stats=batch_distribution({1:1,32:3})
        self.assertEqual(stats['observations'],4)
        self.assertEqual(stats['samples'],97)
        self.assertEqual(stats['mean'],24.25)
        self.assertEqual(stats['p99'],32)

    def make_run(self, root):
        out=root/'run'; (out/'scheduler').mkdir(parents=True); (out/'measurement').mkdir()
        origin=1700000000000
        tasks=[]; clients=[]; replay=[]
        for index,model in enumerate(['resnet50','vgg19'],1):
            request_id=str(uuid.UUID(int=index)); due=origin+index
            tasks.append({'task_id':request_id,'model':model,'batch_size':7,'received_at':due+5,
                          'completed_at':due+15,'status':'completed','error':None})
            clients.append({'request_id':request_id,'model':model,'batch_size':7,'planned_due_ms':due,
                            'completed_at_ms':due+20,'end_to_end_ms':20,'success':True,'failure_kind':None})
            replay.append([index/1000,model,7])
        dump(out/'manifest.json',{'arguments':{'warmup_seconds':0,'client_terminal_log':True},
                                  'experiment_allocation':{'resolved_plan_sha256':'plan'},'replay_sha256':'replay','code_identity':'code'})
        dump(out/'simulation.json',{'models':['resnet50','vgg19']})
        dump(out/'summary.json',{'client_sent_tasks':2,'client_failed_tasks':0,'offered_samples':14,
             'all_offered_received':True,'all_offered_samples_received':True,
             'terminal_source':'server_csv_after_flush','server_history_baseline':0,
             'all_requests_terminal':True,'pending_tasks':0,'received_tasks':2,'received_samples':14,
             'completed_tasks':2,'failed_tasks':0,'completed_samples':14,
             'mean_jct_ms':10,'p95_jct_ms':10,'p99_jct_ms':10,'max_send_lag_ms':None,
             'completed_samples_per_wall_second':14/30.1,
             'measurement_started_at_ms':origin,'measurement_finished_at_ms':origin+30000,
             'sender_exit_code':0,'sending_and_drain_seconds':30.1,'shutdown_errors':[]})
        dump(out/'shutdown.json',{'errors':[]}); dump(out/'tasks.json',tasks); dump(out/'replay.json',replay)
        dump(out/'measurement/final_summary.json',{'planned_origin_ms':origin,'elapsed_seconds':30,
             'sending_seconds':30,'pending_tasks':0,'sent_tasks':2,'planned_requests':2,'replay_requests':2,
             'sent_samples':14,'completed_tasks':2,'failed_tasks':0,'completed_samples':14,
             'transport':{'failed_by_reason':{},'models':{model:{'offered':1,'completed':1,'failed':0,
                    'slo_250_ms_hits':1,'slo_400_ms_hits':1} for model in ['resnet50','vgg19']}}})
        (out/'measurement/client-terminal.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in clients))
        collector=load_collector()
        with (out/'scheduler/task_terminal_log.csv').open('w',newline='') as target:
            writer=csv.DictWriter(target,fieldnames=collector.FIELDS); writer.writeheader()
            for task in tasks: writer.writerow({'model_name':task['model'],'received_at':task['received_at'],
                'processing_duration_ms':10,'batch_size':7,'task_id':task['task_id'],'status':'completed','error':''})
        timestamp=datetime.fromtimestamp((origin+1000)/1000,timezone.utc).isoformat()
        (out/'scheduler/worker_perf.jsonl.today').write_text(''.join(json.dumps({'timestamp':timestamp,
                             'fields':{'batch_size':batch}})+'\n' for batch in [6,8]))
        log=root/'runner.log'; log.write_text('normal exit\n')
        return out,log,collector

    def test_complete_post_shutdown_join_keeps_input_and_execution_batches_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            out,log,collector=self.make_run(Path(directory))
            original=(out/'summary.json').read_bytes()
            report=slot_report(out,0,log,collector)
            self.assertTrue(report['measurement_complete'],report.get('collection_error'))
            self.assertEqual(report['client_metrics']['input_request_batch']['histogram'],{'7':2})
            self.assertEqual(report['actual_worker_execution_batch']['histogram'],{'6':1,'8':1})
            self.assertEqual(report['actual_worker_execution_batch']['samples'],14)
            self.assertEqual(report['client_metrics']['all_offered_slo_percent']['400'],100)
            self.assertEqual((out/'summary.json').read_bytes(),original)
            drift=paired_drift(report,report)
            self.assertTrue(drift['same_plan']); self.assertTrue(drift['same_replay'])
            self.assertEqual(drift['mean_jct_ms']['difference'],0)
            journal=out/'measurement/client-terminal.jsonl'
            journal.write_text(journal.read_text()+journal.read_text().splitlines()[0]+'\n')
            invalid=slot_report(out,0,log,collector)
            self.assertFalse(invalid['measurement_complete'])
            self.assertIn('duplicate',invalid['collection_error'])
            self.assertFalse(paired_drift(invalid,report)['available'])


if __name__=='__main__': unittest.main()
