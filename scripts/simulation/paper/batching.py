"""Bounded strict-fixed tuning, then fresh serial paired validation; no retries."""
from collections import defaultdict
from pathlib import Path
import gzip
import hashlib
import itertools
import json
import os
import random
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback

from run import ROOT,code_identity,ensure_ports_free
from calibrate_parallel import OwnedMonitor,process_rows,slot_configuration
from .batching_audit import audit,expected_censor_exit,metrics
from .queue import sample,remaining_owned
SRC=ROOT
A=S=None


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def local_pairs(pair):
    return sorted(itertools.product(*[sorted({max(1, min(32, b+d)) for d in [-1,0,1]}) for b in pair]))


def calibration_passes(mean_ratio, p99_ratio, slo_difference, limits):
    return (limits['mean_ratio_range'][0] <= mean_ratio <= limits['mean_ratio_range'][1]
        and limits['p99_ratio_range'][0] <= p99_ratio <= limits['p99_ratio_range'][1]
        and abs(slo_difference) <= limits['slo400_absolute_difference_pp'])


def fixed_batch_control(targets, destination, transport):
    if transport=='environment':
        return [], {'HSGS_EXPERIMENT_FIXED_BATCHES':json.dumps(targets,sort_keys=True)}
    if transport!='cli':raise ValueError('unknown fixed-batch control interface')
    save(destination,targets)
    return ['--experiment-fixed-batches',str(destination)], {}


def choose(data):
    common = set.intersection(*[{r['request_id'] for r in row['audit']['requests'] if r['observed_complete']} for row in data])
    assert common, 'search has no jointly completed original requests'
    grouped = defaultdict(list)
    for row in data:
        grouped[tuple(row['fixed_targets'][m] for m in ['resnet50','vgg19'])].append(row)
    candidates = []
    for pair, group in sorted(grouped.items()):
        value = {'pair':list(pair), 'repeats':len(group),
            'common_completed_mean_jct_ms':statistics.mean(statistics.mean(r['server_jct_ms'] for r in row['audit']['requests'] if r['request_id'] in common) for row in group)}
        value.update({key:statistics.mean(row['audit']['metrics']['all'][key] for row in group)
            for key in ['slo400_percent','completion_percent','censored','all_offered_capped_e2e_mean_ms','successful_server_mean_ms']})
        candidates.append(value)
    best = min(candidates, key=lambda v:(v['common_completed_mean_jct_ms'],v['all_offered_capped_e2e_mean_ms'],v['pair']))
    offered = len(data[0]['audit']['requests'])
    return {'selected':best, 'candidates':candidates, 'common_request_ids':sorted(common),
        'common_requests':len(common), 'offered_requests_per_run':offered, 'common_coverage_percent':100*len(common)/offered}


def archive_large_logs(run, heartbeat):
    run.relative_to(S)
    paths = [run/'scheduler.log', run/'parent.log']+list(run.glob('scheduler/task_tracking.jsonl*'))+list(run.glob('scheduler/worker_perf.jsonl*'))
    archived = []
    for path in paths:
        if not path.is_file() or path.name.endswith('.gz') or path.stat().st_size < 128*1024:
            continue
        heartbeat()
        raw_sha, raw_size = sha(path), path.stat().st_size
        dest = Path(str(path)+'.gz')
        assert not dest.exists()
        tmp = dest.with_name(dest.name+'.tmp')
        with path.open('rb') as source, tmp.open('wb') as handle:
            with gzip.GzipFile(filename='', mode='wb', fileobj=handle, compresslevel=6, mtime=0) as output:
                while chunk := source.read(1024*1024):
                    output.write(chunk)
                    heartbeat()
        with gzip.open(tmp,'rb') as handle:
            expanded_sha = hashlib.file_digest(handle,'sha256').hexdigest()
        assert expanded_sha == raw_sha
        tmp.replace(dest)
        archived.append({'raw_path':str(path), 'raw_size':raw_size, 'raw_sha256':raw_sha,
            'gzip_path':str(dest), 'gzip_size':dest.stat().st_size, 'gzip_sha256':sha(dest), 'decompressed_sha256':expanded_sha})
        path.unlink()
    save(run.parent/'log-archives.json', archived)


class Pipeline:
    def __init__(self, root):
        global A,S
        A=S=root
        assert not (A/'process-status.json').exists(), 'one-time launch; do not duplicate'
        self.protocol = read(S/'protocol.json')
        self.identity = read(A/'experiment-identity.json')
        self.state = {'state':'running','pipeline_pid':os.getpid(),'completed_stages':[], 'completed_measurements':0,
            'started_at_ms':round(time.time()*1000),'error':None}
        self.results = []
        self.active = {}
        self.runners = []
        self.monitor = None
        self.awake = None

    def checkpoint(self, **changes):
        self.state.update(changes, updated_at_ms=round(time.time()*1000))
        save(A/'process-status.json',self.state)
        save(S/'suite-status.json',self.state)

    def verify(self):
        assert code_identity(SRC,Path(self.protocol['target'])/'debug') == self.protocol['runtime_identity']
        assert all(sha(Path(p)) == h for p,h in self.identity['files'].items())
        assert shutil.disk_usage(S).free > 2*1024**3, 'free disk reserve below2GiB'

    def job(self, case, name, seed, pair=None, calibration_mode=None):
        return {'case':case,'id':name,'runtime_seed':seed,'mode':'fixed' if pair is not None else 'dynamic',
            'fixed_targets':dict(zip(['resnet50','vgg19'],pair)) if pair is not None else None,
            'calibration_mode':calibration_mode}

    def queue(self, label, jobs, concurrency):
        self.verify()
        folder = S/label
        folder.mkdir(parents=True, exist_ok=False)
        save(folder/'jobs.json',jobs)
        self.runners = []
        self.active = {}
        self.monitor = OwnedMonitor(folder/'owned-host.jsonl',self.runners,4*1024**2)
        last_sample = -float('inf')
        data, free, index = [], list(range(concurrency)), 0
        def heartbeat(force=False):
            nonlocal last_sample
            if force or time.monotonic()-last_sample >= 2:
                sample(self.monitor,self.active)
                last_sample = time.monotonic()
                assert shutil.disk_usage(S).free > 2*1024**3, 'free disk reserve below2GiB'
                self.checkpoint(stage=label,stage_completed=len(data),stage_expected=len(jobs),concurrency=concurrency,
                    active=[{'runner_pid':slot['process'].pid,'job':slot['job'],'output':str(slot['out'])} for slot in self.active.values()])
        try:
            while self.active or index<len(jobs):
                heartbeat()
                for slot_index, slot in list(self.active.items()):
                    proc, job = slot['process'], slot['job']
                    if proc.poll() is None:
                        assert time.monotonic()-slot['started'] < self.protocol['cases'][job['case']]['seconds']+100, 'hard trial timeout'
                        continue
                    heartbeat(True)
                    slot['log'].close()
                    remaining = remaining_owned(slot,process_rows())
                    save(slot['folder']/'exit.json',{'returncode':proc.returncode,'remaining_owned':remaining})
                    assert not remaining, 'owned descendants remain after runner exit'
                    case = self.protocol['cases'][job['case']]
                    certificate = audit(slot['out'],case,job['mode'],job['fixed_targets'],self.protocol['runtime_identity'],job['runtime_seed'])
                    assert expected_censor_exit(proc.returncode,certificate), 'unexpected runner failure'
                    row = {**job,'output':str(slot['out']),'runner_exit_code':proc.returncode,'audit':certificate,
                        'measurement_origin':'new_measurement','stage':label,'concurrency':concurrency}
                    save(slot['folder']/'audit.json',row)
                    heartbeat(True)
                    archive_large_logs(slot['out'],heartbeat)
                    data.append(row)
                    self.results.append(row)
                    with (folder/'results.jsonl').open('a') as handle: handle.write(json.dumps(row)+'\n')
                    with (S/'all-results.jsonl').open('a') as handle: handle.write(json.dumps(row)+'\n')
                    self.runners.remove(proc)
                    del self.active[slot_index]
                    free.append(slot_index)
                    self.state['completed_measurements'] += 1
                    heartbeat(True)
                while free and index<len(jobs):
                    slot_index, job = free.pop(0), jobs[index]
                    index += 1
                    trial = folder/f'trial-{index:04d}-{job["id"]}'
                    trial.mkdir()
                    inp = S/job['case']/'inputs'
                    config, port = slot_configuration(read(inp/f'configuration-{job["mode"]}.json'),35300,slot_index)
                    ensure_ports_free([port]+[node['port'] for node in config['nodes']])
                    save(trial/'configuration.json',config)
                    out = trial/'run'
                    cmd = [sys.executable,str(SRC/'scripts/simulation/run.py'),'--skip-build','--config',str(trial/'configuration.json'),
                        '--output',str(out),'--port',str(port),'--seconds',str(self.protocol['cases'][job['case']]['seconds']),
                        '--warmup-seconds','0','--rate','75','--seed','20260923','--random-seed',str(job['runtime_seed']),
                        '--replay',str(inp/'replay.json'),'--model-rates',str(inp/'model-rates.json'),'--noise-cv','.1',
                        '--client-terminal-log','--ready-timeout','180','--schedule-interval','5','--max-host-memory-gib','4',
                        '--experiment-fixed-allocation',str(inp/'allocation.json'),'--drain-seconds','10']
                    env = dict(os.environ,CARGO_TARGET_DIR=self.protocol['target'],PYTHONDONTWRITEBYTECODE='1',GIT_OPTIONAL_LOCKS='0')
                    env.pop('HSGS_EXPERIMENT_FIXED_BATCHES',None)
                    if job['mode']=='fixed':
                        flags,values=fixed_batch_control(job['fixed_targets'],trial/'fixed-batches.json',self.protocol['fixed_batch_control'])
                        cmd+=flags;env.update(values)
                    log = (trial/'runner.log').open('x')
                    proc = subprocess.Popen(cmd,cwd=SRC,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
                    self.runners.append(proc)
                    self.active[slot_index]={'process':proc,'job':job,'folder':trial,'out':out,'started':time.monotonic(),'log':log,'owned':{}}
                    save(trial/'command.json',{'command':cmd,'job':job,'runner_pid':proc.pid,'fixed_targets':job['fixed_targets'],'fixed_batch_control':self.protocol['fixed_batch_control']})
                    heartbeat(True)
                if self.active: time.sleep(.05)
            heartbeat(True)
            remaining = self.monitor.remaining()
            assert not remaining
            host = self.monitor.close()
            self.monitor = None
            assert not host['sampling_errors']
            save(folder/'queue-audit.json',{'status':'passed','measurements':len(data),'concurrency':concurrency,'host':host,'remaining_owned':remaining})
            self.state['completed_stages'].append(label)
            self.checkpoint(active=[])
            return data
        except BaseException:
            self.checkpoint(state='needs_attention',error=traceback.format_exc())
            if self.monitor:
                self.monitor.stop_runners()
                for proc in self.runners:
                    if proc.poll() is None: proc.wait(timeout=340)
                self.monitor.sample()
                save(folder/'failure-owned-processes.json',self.monitor.remaining())
                self.monitor.close()
                self.monitor = None
            for slot in self.active.values():
                if not slot['log'].closed: slot['log'].close()
            raise

    def calibrate(self):
        seed = self.protocol['calibration_seed']
        modes = [('high','dynamic',None),('high','fixed-high',(16,8)),('medium','fixed-medium',(4,4))]
        serial = self.queue('calibration/serial',[self.job(c,n,seed,p,n) for c,n,p in modes],1)
        parallel = self.queue('calibration/parallel',[self.job(c,f'{n}-{i}',seed,p,n) for i,(c,n,p) in enumerate((modes*3)[:8])],self.protocol['tuning_concurrency'])
        comparisons = []
        for reference in serial:
            same = [r for r in parallel if r['calibration_mode']==reference['calibration_mode']]
            ratios = {key:statistics.mean(r['audit']['metrics']['all'][key] for r in same)/reference['audit']['metrics']['all'][key]
                for key in ['successful_server_mean_ms','successful_server_p99_ms']}
            serial_slo = reference['audit']['metrics']['all']['slo400_percent']
            parallel_slo = statistics.mean(r['audit']['metrics']['all']['slo400_percent'] for r in same)
            difference = parallel_slo-serial_slo
            comparisons.append({'mode':reference['calibration_mode'],'ratios':ratios,
                'serial_slo400_percent':serial_slo, 'parallel_slo400_percent':parallel_slo, 'slo400_difference_pp':difference,
                'passed':calibration_passes(ratios['successful_server_mean_ms'],ratios['successful_server_p99_ms'],difference,self.protocol['calibration_screen'])})
        screen = {'passed':all(r['passed'] for r in comparisons),'comparisons':comparisons,'thresholds':self.protocol['calibration_screen'],
            'scope':'Engineering interference screen, not statistical equivalence; final validation serial; no automatic retry/concurrency fallback.'}
        save(S/'concurrency-calibration.json',screen)
        assert screen['passed'], 'calibration failed; stop before search'

    def search(self, case):
        coarse_seed = self.protocol['coarse_seed']
        pairs = list(itertools.product(self.protocol['coarse_targets'],repeat=2))
        random.Random(f'{case}-coarse-{coarse_seed}').shuffle(pairs)
        coarse = self.queue(case+'/coarse',[self.job(case,f'b{a}-{b}',coarse_seed,(a,b)) for a,b in pairs],self.protocol['tuning_concurrency'])
        coarse_selection = choose(coarse)
        save(S/case/'coarse-selection.json',coarse_selection)
        pairs = local_pairs(coarse_selection['selected']['pair'])
        jobs = []
        for seed in self.protocol['local_seeds']:
            order = pairs.copy()
            random.Random(f'{case}-local-{seed}').shuffle(order)
            jobs.extend(self.job(case,f'b{a}-{b}-s{seed}',seed,(a,b)) for a,b in order)
        local = self.queue(case+'/local',jobs,self.protocol['tuning_concurrency'])
        selected = choose(local)
        selected.update(coarse_winner=coarse_selection['selected']['pair'],local_pairs=[list(p) for p in pairs],
            selection_seeds=self.protocol['local_seeds'],coarse_seed=coarse_seed,
            scope='Best observed in finite local refinement around coarse winner; not exhaustive or globally optimal.')
        save(S/case/'selection.json',selected)
        return selected['selected']['pair']

    def validate(self, case, pair):
        jobs = []
        for i,seed in enumerate(self.protocol['validation_seeds']):
            arms = [self.job(case,f'dynamic-s{seed}',seed),self.job(case,f'fixed-s{seed}',seed,pair)]
            jobs.extend(arms if i%2==0 else list(reversed(arms)))
        return self.queue(case+'/validation',jobs,1)

    def run(self):
        self.verify()
        self.checkpoint(stage='starting')
        self.awake = subprocess.Popen(['/usr/bin/caffeinate','-is','-w',str(os.getpid())],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,close_fds=True) if sys.platform=='darwin' else None
        try:
            self.calibrate()
            selections = {case:self.search(case) for case in self.protocol['cases']}
            validation = []
            for case,pair in selections.items(): validation.extend(self.validate(case,pair))
            assert len(validation)==40 and len(self.results)<=self.protocol['new_trials_max']
            self.verify()
            save(S/'selected-targets.json',selections)
            save(S/'validation-results.json',validation)
            self.checkpoint(state='measurements_completed_and_audited',stage='awaiting_independent_analysis',active=[],finished_at_ms=round(time.time()*1000))
            save(A/'followup-workflow-status.json',{'phase':'measurements_completed_and_audited','output':str(S),
                'new_measurements':len(self.results),'remaining':['Analyze the audited validation records']})
        except BaseException:
            self.checkpoint(state='needs_attention',error=traceback.format_exc())
            raise
        finally:
            if self.awake is not None:
                self.awake.terminate()
                self.awake.wait()

