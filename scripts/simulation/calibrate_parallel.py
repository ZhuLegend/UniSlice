#!/usr/bin/env python3
"""Calibrate concurrent real mock suites; launch success never certifies JCT validity.

Build the four mock binaries once before invoking this pilot. Example:
  python scripts/simulation/calibrate_parallel.py --output simulation-runs/parallel-cv0
  python scripts/simulation/calibrate_parallel.py --output simulation-runs/parallel-cv10 --levels 1,4 --noise-cv .1 --repeats 3

Every slot has its own node/daemon ports, snapshots, logs and process trees. All
slots in a stage wait at a readiness barrier before their identical 30s replay.
CSV execution curves and normal dynamic execution batching remain unchanged.
Lifecycle waits default to the source values; explicit overrides are snapshotted.
The only automatic stop decisions concern launch, cleanup, timeout or owned RSS;
the author must assess the paired latency/arrival-lag/batch drift measurements.
"""
import argparse
from collections import Counter
import copy
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time

from long_analysis import log_events
from monitor import descendants
from run import ROOT, allocation_plan_identity, code_identity, dump, ensure_ports_free, percentile, single_phase_workload


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def fixture(config_path, name='full-two'):
    """Two bounded fixtures; full-two does not represent mixed-four's 10 workers."""
    config = copy.deepcopy(read_json(config_path))
    config['latency_csv'] = str((config_path.parent/config['latency_csv']).resolve())
    config['batching_config'] = str((config_path.parent/config['batching_config']).resolve())
    parent = next((node for node in config['nodes'] if node['kind']=='nvidia'
                   and node['resource'].get('ce')==7 and node['resource'].get('memory',0)>=70
                   and '1g.10gb' in node['share_targets']), None)
    if parent is None: raise ValueError('fixture requires a configured full parent supporting 1g.10gb')
    config['models'] = ['resnet50', 'vgg19']
    parent = dict(parent, name='parent', gpus=1, template_name='parent_full', share_targets=['1g.10gb'])
    config['nodes'] = [parent]
    config['batch_size_range'] = [1, 32]
    plan = {'schema_version': 1, 'placements': [
        {'template_name': 'parent_full', 'model': model, 'slice': '1g.10gb', 'count': count}
        for model,count in [('resnet50',3), ('vgg19',4)]]}
    if name == 'mixed-four':
        source=read_json(config_path)
        half=next((node for node in source['nodes'] if node['kind']=='nvidia'
                   and node['resource'].get('ce')==3 and node['resource'].get('memory',0)>=30
                   and '1g.10gb' in node['share_targets']),None)
        if half is None: raise ValueError('mixed-four fixture requires a configured half parent supporting 1g.10gb')
        config['models']=['resnet50','vgg19','resnet18','vgg16']
        config['nodes'].append(dict(half,name='half',gpus=1,template_name='parent_half',share_targets=['1g.10gb']))
        plan['placements']=[{'template_name':template,'model':model,'slice':'1g.10gb','count':count}
            for template,counts in [('parent_full',[('resnet50',2),('vgg19',2),('resnet18',2),('vgg16',1)]),
                                    ('parent_half',[('resnet50',1),('vgg19',1),('vgg16',1)])]
            for model,count in counts]
    elif name != 'full-two': raise ValueError('unknown calibration fixture')
    return config, plan


def slot_configuration(config, base_port, index):
    slot=copy.deepcopy(config); daemon_port=base_port+(len(config['nodes'])+1)*index
    for offset,node in enumerate(slot['nodes'],1): node['port']=daemon_port+offset
    return slot, daemon_port


def process_rows():
    # lstart distinguishes an owned PID from a later PID reuse during cleanup.
    text = subprocess.check_output(['ps','-axo','pid=,ppid=,rss=,pcpu=,time=,lstart='], text=True, timeout=5)
    rows = {}
    for line in text.splitlines():
        pid, ppid, rss, cpu, cpu_time, started = line.split(maxsplit=5)
        seconds = cpu_seconds(cpu_time)
        rows[int(pid)] = {'ppid': int(ppid), 'rss_kib': int(rss), 'ps_cpu_percent': float(cpu),
                         'cpu_seconds': seconds, 'started': started}
    return rows


def cpu_seconds(value):
    days, clock = value.split('-',1) if '-' in value else ('0',value)
    return int(days)*86400+sum(float(part)*60**index for index,part in enumerate(reversed(clock.split(':'))))


def host_pressure():
    """Record OS counters, separate from summed process RSS (shared pages repeat)."""
    try:
        if platform.system() == 'Darwin':
            text = subprocess.check_output(['vm_stat'], text=True, timeout=3)
            size = int(re.search(r'page size of (\d+) bytes',text).group(1))
            fields = {key.strip():int(value.rstrip('.')) for key,value in
                      (line.split(':',1) for line in text.splitlines()[1:] if ':' in line)}
            return {'source':'vm_stat','page_size_bytes':size,'counters':fields}
        path = Path('/proc/meminfo')
        if path.exists():
            return {'source':'/proc/meminfo','raw':path.read_text()}
        return {'source':None,'unavailable':True}
    except (OSError, ValueError, AttributeError, subprocess.SubprocessError) as exc:
        return {'error':str(exc)}


class OwnedMonitor:
    """Sample and stop only runner PIDs launched by this calibration stage."""
    def __init__(self, output, runners, max_rss_kib):
        self.log = output.open('w')
        self.runners, self.max_rss_kib = runners, max_rss_kib
        self.seen, self.cpu_max, self.samples = {}, {}, []
        self.guard = None

    def sample(self):
        event = {'timestamp_ms':round(time.time()*1000), 'host_pressure':host_pressure()}
        try:
            rows = process_rows()
            known = [pid for pid,started in self.seen.items() if pid in rows and rows[pid]['started']==started]
            selected = descendants(rows, [proc.pid for proc in self.runners if proc.poll() is None]+known)
            self.seen.update({pid:rows[pid]['started'] for pid in selected})
            # Known orphan parents remain roots for any later owned children;
            # reused PIDs never become roots.
            for pid in selected: self.cpu_max[pid] = max(self.cpu_max.get(pid,0),rows[pid]['cpu_seconds'])
            event.update(process_count=len(selected), rss_kib=sum(rows[pid]['rss_kib'] for pid in selected),
                         ps_cpu_percent=sum(rows[pid]['ps_cpu_percent'] for pid in selected),
                         processes={str(pid):rows[pid] for pid in sorted(selected)})
            if event['rss_kib'] > self.max_rss_kib and self.guard is None:
                self.guard = dict(event, limit_kib=self.max_rss_kib)
                dump(Path(self.log.name).with_name('memory_guard.json'),self.guard)
                self.stop_runners()
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            event['error'] = str(exc)
        self.samples.append({key:value for key,value in event.items() if key!='processes'})
        self.log.write(json.dumps(event)+'\n'); self.log.flush()
        return event

    def stop_runners(self):
        # SIGTERM enters run.py's own node/worker cleanup; no broad pkill/killall.
        for proc in self.runners:
            if proc.poll() is None:
                try: proc.terminate()
                except ProcessLookupError: pass

    def remaining(self):
        rows = process_rows()
        known=[pid for pid,started in self.seen.items() if pid in rows and rows[pid]['started']==started]
        return {str(pid):rows[pid] for pid in descendants(rows,known)}

    def close(self):
        self.log.close()
        rss = [row['rss_kib'] for row in self.samples if 'rss_kib' in row]
        cpu = [row['ps_cpu_percent'] for row in self.samples if 'ps_cpu_percent' in row]
        return {'samples':len(self.samples), 'sampling_errors':sum('error' in row for row in self.samples),
                'peak_owned_rss_kib':max(rss,default=None), 'mean_owned_rss_kib':sum(rss)/len(rss) if rss else None,
                'peak_ps_cpu_percent':max(cpu,default=None), 'mean_ps_cpu_percent':sum(cpu)/len(cpu) if cpu else None,
                'sampled_process_cpu_seconds_lower_bound':sum(self.cpu_max.values()), 'memory_guard':self.guard,
                'cpu_scope':'ps percent is each process lifetime average; CPU-seconds sum is a sampled lower bound.',
                'rss_scope':'Sum of owned runner/descendant RSS; shared mappings may be counted multiple times. '
                            'OS host memory counters are logged separately; unrelated processes are never signalled.'}


def load_collector():
    import terminal
    return terminal


def distribution(values):
    return {'n':len(values), 'mean':sum(values)/len(values) if values else None,
            'p95':percentile(values,.95), 'p99':percentile(values,.99)}


def batch_distribution(histogram):
    count = sum(histogram.values())
    batches = [batch for batch,n in histogram.items() for _ in range(n)]
    return {'observations':count, 'samples':sum(batch*n for batch,n in histogram.items()),
            **distribution(batches), 'histogram':{str(batch):n for batch,n in sorted(histogram.items())}}


def slot_report(out, exit_code, runner_log, collector):
    """Reuse the frozen complete client/server UUID join after daemon shutdown."""
    result = {'output':str(out), 'runner_exit_code':exit_code, 'measurement_complete':False,
              'original_runner_error':read_json(out/'error.json'), 'shutdown':read_json(out/'shutdown.json'),
              'client_final':read_json(out/'measurement/final_summary.json'),
              'original_runner_summary':read_json(out/'summary.json')}
    manifest = read_json(out/'manifest.json', {})
    result['plan_identity'] = manifest.get('experiment_allocation')
    result['replay_sha256'] = manifest.get('replay_sha256')
    result['code_identity'] = manifest.get('code_identity')
    try:
        tasks, summary, provenance = collector.audit(out, exit_code, runner_log)
        result.update(measurement_complete=True, terminal_collection=provenance, server_summary=summary)
        with (out/'measurement/client-terminal.jsonl').open() as journal:
            clients = [json.loads(line) for line in journal]
        successful = [row for row in clients if row['success']]
        result['client_metrics'] = {'offered_requests':len(clients), 'successful_requests':len(successful),
            'failed_requests':len(clients)-len(successful), 'pending_requests':0,
            'offered_samples':sum(row['batch_size'] for row in clients),
            'completed_samples':sum(row['batch_size'] for row in successful),
            'failure_percent':100*(len(clients)-len(successful))/len(clients),
            'failure_reasons':dict(Counter(row['failure_kind'] for row in clients if not row['success'])),
            'successful_end_to_end_ms':distribution([row['end_to_end_ms'] for row in successful]),
            'all_offered_slo_percent':{str(limit):100*sum(row['end_to_end_ms']<=limit for row in successful)/len(clients)
                                      for limit in [250,400]},
            'input_request_batch':batch_distribution(Counter(row['batch_size'] for row in clients))}
        final = result['client_final']
        start, end = final['planned_origin_ms'], final['planned_origin_ms']+final['elapsed_seconds']*1000
        result['client_window'] = {'start_ms':start, 'end_ms':end, 'sending_seconds':final['sending_seconds'],
                                  'sending_and_drain_seconds':final['elapsed_seconds']}
        hist = Counter()
        for path in sorted((out/'scheduler').glob('worker_perf.jsonl*')):
            for event in log_events(path):
                if summary['measurement_started_at_ms'] <= event['timestamp_ms'] <= summary['measurement_finished_at_ms']:
                    batch = event['batch_size']
                    if type(batch) is not int or not 1 <= batch <= 32: raise ValueError('invalid executed worker batch')
                    hist[batch] += 1
        result['actual_worker_execution_batch'] = batch_distribution(hist)
        if not hist: raise ValueError('no completed worker inference calls in the measurement window')
        if not summary['failed_tasks'] and result['actual_worker_execution_batch']['samples'] != summary['completed_samples']:
            raise ValueError('worker executed samples differ from fully successful original-request samples')
        result['worker_batch_scope'] = 'Completed scheduler-observed worker inference calls in the server measurement window, including drain; per-call weighting.'
        result['input_batch_scope'] = 'Original offered requests, including failures; per-request weighting. Input batches are separate from split/merged worker execution batches.'
        result['client_throughput_samples_per_second'] = summary['completed_samples']/final['elapsed_seconds'] if final['elapsed_seconds'] else None
    except (OSError, ValueError, KeyError, AssertionError, TypeError) as exc:
        result['collection_error'] = str(exc)
        result['measurement_complete'] = False
    dump(out/'calibration-terminal.json',result)
    return result


def paired_drift(result, serial):
    """Descriptive same-input comparisons, without significance/validity thresholds."""
    if not result['measurement_complete'] or not serial['measurement_complete']:
        return {'available':False, 'reason':'incomplete terminal or worker-batch collection'}
    delta = {'available':True, 'reference_output':serial['output'],
             'same_replay':result['replay_sha256']==serial['replay_sha256'],
             'same_plan':result['plan_identity']['resolved_plan_sha256']==serial['plan_identity']['resolved_plan_sha256'],
             'same_code':result['code_identity']==serial['code_identity'],
             'interpretation':'Paired planned input only; random worker IDs, routing and noise streams are not paired. '
                              'Observed drift has no automatic JCT-validity or statistical-significance verdict.'}
    for key in ['mean_jct_ms','p95_jct_ms','p99_jct_ms','max_send_lag_ms','completed_samples_per_wall_second']:
        value, reference = result['server_summary'].get(key), serial['server_summary'].get(key)
        delta[key] = {'serial':reference, 'observed':value, 'difference':value-reference if value is not None and reference is not None else None,
                      'relative_percent':100*(value/reference-1) if value is not None and reference else None}
    for key in ['mean','p95','p99']:
        value, reference = (row['client_metrics']['successful_end_to_end_ms'][key] for row in [result,serial])
        delta['client_'+key+'_ms'] = {'serial':reference,'observed':value,'difference':value-reference if value is not None and reference is not None else None}
    delta['failure_percentage_point_change'] = result['client_metrics']['failure_percent']-serial['client_metrics']['failure_percent']
    delta['slo_percentage_point_change'] = {limit:result['client_metrics']['all_offered_slo_percent'][limit]-serial['client_metrics']['all_offered_slo_percent'][limit] for limit in ['250','400']}
    for key in ['mean','p99']:
        value, reference = (row['actual_worker_execution_batch'][key] for row in [result,serial])
        delta['executed_batch_'+key] = {'serial':reference,'observed':value,'difference':value-reference if value is not None and reference is not None else None}
    return delta


def stage(args, root, config, plan_path, replay_path, rate, k, repeat, identity, collector):
    output = root/f'rate-{rate:g}'/f'k-{k}'/f'repeat-{repeat}'
    output.mkdir(parents=True)
    barrier = output/'measurement-release.json'
    runners, logs, runs = [], [], []
    started, last_sample = time.monotonic(), -math.inf
    monitor = OwnedMonitor(output/'host_metrics.jsonl',runners,args.max_host_memory_gib*1024**2)
    error, release_ms = None, None
    def tick():
        nonlocal last_sample
        if time.monotonic()-last_sample >= args.sample_seconds:
            event=monitor.sample(); last_sample=time.monotonic()
            if 'error' in event: raise RuntimeError('owned process monitor failed; RSS guard cannot be verified')
        if monitor.guard: raise RuntimeError('owned RSS guard fired; host-limited calibration stage')
        if time.monotonic()-started > args.stage_timeout: raise TimeoutError('calibration stage wall timeout')
    try:
        port_width=len(config['nodes'])+1
        ensure_ports_free(list(range(args.base_port,args.base_port+port_width*k)))
        for index in range(k):
            slot,daemon_port=slot_configuration(config,args.base_port,index)
            config_path = output/f'slot-{index+1}-config.json'; dump(config_path,slot)
            out = output/f'slot-{index+1}'; runner_log = output/f'slot-{index+1}-runner.log'
            command = [sys.executable,str(ROOT/'scripts/simulation/run.py'),'--skip-build',
                '--config',str(config_path),'--output',str(out),'--port',str(daemon_port),
                '--seconds',str(args.seconds),'--warmup-seconds','0','--rate',str(rate),'--seed',str(args.seed),
                '--replay',str(replay_path),'--noise-cv',str(args.noise_cv),'--client-terminal-log',
                '--experiment-fixed-allocation',str(plan_path),'--measurement-barrier',str(barrier),
                '--barrier-timeout',str(args.ready_timeout+60),'--ready-timeout',str(args.ready_timeout),
                '--schedule-interval','5','--max-host-memory-gib',str(args.runner_memory_gib)]
            if args.drain_seconds is not None: command += ['--drain-seconds',str(args.drain_seconds)]
            log = runner_log.open('w'); logs.append(log)
            runners.append(subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
            runs.append((out,runner_log,command))
        ready_deadline = time.monotonic()+args.ready_timeout
        while True:
            tick()
            if any(proc.poll() is not None for proc in runners): raise RuntimeError('runner exited before the all-ready barrier')
            ready = []
            for out,_,_ in runs:
                try: ready.append(read_json(out/'calibration-ready.json'))
                except (OSError, ValueError): ready.append(None)
            if all(ready):
                if any(row['code_identity']!=identity for row in ready): raise RuntimeError('runner source or binaries differ from the frozen calibration identity')
                if any(row['experiment_allocation']['resolved_plan_sha256']!=allocation_plan_identity(read_json(plan_path)) for row in ready):
                    raise RuntimeError('runner resolved allocation differs from the calibration plan')
                if len({row['replay_sha256'] for row in ready})!=1: raise RuntimeError('runner replay identities differ')
                release_ms=round(time.time()*1000)
                temporary=barrier.with_suffix('.tmp'); dump(temporary,{'released_at_ms':release_ms,'slots':k})
                temporary.replace(barrier)
                break
            if time.monotonic()>=ready_deadline: raise TimeoutError('all-ready barrier timeout')
            time.sleep(.2)
        while any(proc.poll() is None for proc in runners):
            tick(); time.sleep(.2)
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        monitor.stop_runners()
    finally:
        # Give each runner its existing graceful cleanup budget. Only kill its
        # own session if the runner itself cannot exit; record that failure.
        cleanup_deadline = time.monotonic()+config.get('control_timeout_ms',300000)/1000+45
        for proc in runners:
            try: proc.wait(timeout=max(.1,cleanup_deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                error = (error+'; ' if error else '')+f'runner {proc.pid} cleanup timeout'
                try: os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                proc.wait()
        for log in logs: log.close()
        monitor.sample()
        try: remaining = monitor.remaining()
        except (OSError, ValueError, subprocess.SubprocessError) as exc: remaining={'unverified':str(exc)}
        host = monitor.close()
    wall = time.monotonic()-started
    slots = [slot_report(out,proc.returncode,log,collector) for proc,(out,log,_) in zip(runners,runs)]
    origins = [row['client_window']['start_ms'] for row in slots if 'client_window' in row]
    result = {'rate_samples_per_second_per_model':rate,'concurrency':k,'repeat':repeat,'noise_cv':args.noise_cv,
        'stage_wall_seconds':wall,'barrier_release_ms':release_ms,'launch_ready':release_ms is not None,
        'measurement_complete':len(slots)==k and all(row['measurement_complete'] for row in slots),
        'all_owned_processes_gone':not remaining,'remaining_owned_processes':remaining,'host':host,
        'client_origin_spread_ms':max(origins)-min(origins) if len(origins)==k else None,
        'error':error,'slots':slots,'commands':[command for _,_,command in runs],
        'jct_validity':'Not automatically assessed; inspect descriptive same-trace serial drift.'}
    dump(output/'stage-summary.json',result)
    return result, output


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'resource/simulation/paper-a100.json')
    parser.add_argument('--fixture',choices=['full-two','mixed-four'],default='full-two',help='7-worker one-parent or 10-worker full+half-parent stress fixture')
    parser.add_argument('--levels',default='1,2,4,8',help='strictly increasing comma-separated levels, beginning with 1; add 16 explicitly')
    parser.add_argument('--rates',help='constant samples/s/model; defaults full-two:100,300 / mixed-four:100,150')
    parser.add_argument('--repeats',type=int,default=1,help='use additional repeats near a selected concurrency boundary')
    parser.add_argument('--seconds',type=int,default=30)
    parser.add_argument('--seed',type=int,default=20260923)
    parser.add_argument('--noise-cv',type=float,default=0,help='0 isolates host timing drift; use .1 in a separate calibration')
    parser.add_argument('--create-delay-ms',type=int,help='explicit static/calibration creation wait override; snapshots record the exact value')
    parser.add_argument('--destroy-delay-ms',type=int,help='explicit static/calibration cleanup wait override; snapshots record the exact value')
    parser.add_argument('--model-load-ms',type=int,help='simulated model initialization wait; use 0 for static allocation JCT comparisons')
    parser.add_argument('--model-unload-ms',type=int,help='simulated cleanup wait, excluded from measured request JCT')
    parser.add_argument('--base-port',type=int,default=15300)
    parser.add_argument('--max-host-memory-gib',type=float,default=12)
    parser.add_argument('--runner-memory-gib',type=float,default=4)
    parser.add_argument('--sample-seconds',type=float,default=2)
    parser.add_argument('--ready-timeout',type=int,default=900)
    parser.add_argument('--stage-timeout',type=int,default=1800)
    parser.add_argument('--drain-seconds',type=int,help='optional runner drain override; incomplete draining is always reported')
    args=parser.parse_args()
    if args.rates is None: args.rates='100,300' if args.fixture=='full-two' else '100,150'
    try: levels=[int(value) for value in args.levels.split(',')]; rates=[float(value) for value in args.rates.split(',')]
    except ValueError: parser.error('levels and rates must be numeric comma-separated lists')
    if not levels or levels[0]!=1 or levels!=sorted(set(levels)) or args.base_port<1024:
        parser.error('levels must increase from 1 and use unprivileged valid ports')
    if not rates or any(not math.isfinite(value) or value<=0 for value in rates) or len(set(rates))!=len(rates): parser.error('rates must be unique, finite and positive')
    if not 0<=args.noise_cv<=1 or not math.isfinite(args.noise_cv): parser.error('noise CV must be in [0,1]')
    if min(args.repeats,args.seconds,args.ready_timeout,args.stage_timeout)<=0 or any(not math.isfinite(value) or value<=0 for value in [args.max_host_memory_gib,args.runner_memory_gib,args.sample_seconds]): parser.error('counts, timings and memory limits must be positive')
    if args.drain_seconds is not None and args.drain_seconds<=0: parser.error('drain timeout must be positive')
    if any(value is not None and value<0 for value in [args.create_delay_ms,args.destroy_delay_ms,args.model_load_ms,args.model_unload_ms]): parser.error('lifecycle overrides cannot be negative')
    config,plan=fixture(args.config.resolve(),args.fixture)
    if args.create_delay_ms is not None: config['create_delay_ms']=args.create_delay_ms
    if args.destroy_delay_ms is not None: config['destroy_delay_ms']=args.destroy_delay_ms
    lifecycle=config.setdefault('model_lifecycle',{'load_ms':1000,'unload_ms':100,'overrides':[]})
    if args.model_load_ms is not None:
        lifecycle['load_ms']=args.model_load_ms
        for override in lifecycle.get('overrides',[]): override['load_ms']=args.model_load_ms
    if args.model_unload_ms is not None:
        lifecycle['unload_ms']=args.model_unload_ms
        for override in lifecycle.get('overrides',[]): override['unload_ms']=args.model_unload_ms
    if args.base_port+(len(config['nodes'])+1)*max(levels)>65535: parser.error('fixture slot ports exceed 65535')
    root=args.output.resolve(); root.mkdir(parents=True,exist_ok=False)
    inputs=root/'inputs'; inputs.mkdir()
    plan_path=inputs/'allocation.json'; dump(plan_path,plan)
    dump(inputs/'fixture.json',config)
    identity=code_identity(); collector=load_collector()
    dump(root/'manifest.json',{'arguments':{key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
        'code_identity':identity,'fixture':config,'plan':plan,'plan_sha256':allocation_plan_identity(plan),
        'started_unix_seconds':time.time(),'platform':platform.platform(),'logical_cpus':os.cpu_count(),
        'design':'Same planned replay and fixed placement per rate. Normal adaptive execution batching. '
                 'No automatic latency-validity threshold; startup feasibility and timing drift are reported separately.'})
    report={'stages':[],'finished':False,'stopped_reason':None,'automatic_jct_validity_assessment':False}
    def terminate_calibration(signum, _frame):
        raise KeyboardInterrupt(f'calibration interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,terminate_calibration)
    try:
        for rate in rates:
            _,events=single_phase_workload(config['models'],rate,args.seconds,args.seed,(1,32))
            replay_path=inputs/f'replay-rate-{rate:g}.json'; dump(replay_path,events)
            serial={}
            for k in levels:
                for repeat in range(1,args.repeats+1):
                    print(f'Calibration rate={rate:g} samples/s/model K={k} repeat={repeat} CV={args.noise_cv:g}',flush=True)
                    result,output=stage(args,root,config,plan_path,replay_path,rate,k,repeat,identity,collector)
                    if k==1: serial[repeat]=result['slots'][0] if result['slots'] else {'measurement_complete':False}
                    for slot in result['slots']: slot['paired_serial_drift']=paired_drift(slot,serial[repeat])
                    dump(output/'stage-summary.json',result)
                    report['stages'].append(result); dump(root/'summary.json',report)
                    if result['error'] or not result['all_owned_processes_gone'] or not result['launch_ready'] or not result['measurement_complete']:
                        raise RuntimeError(f'calibration stopped after stage safety/launch/terminal-integrity failure: {output}')
        report['finished']=True
    except BaseException as exc:
        report['stopped_reason']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['finished_unix_seconds']=time.time(); dump(root/'summary.json',report)


if __name__=='__main__': main()
