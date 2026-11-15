#!/usr/bin/env python3
"""Continuously refill bounded simulator slots; this is not an exhaustive-search solver.

Jobs are streamed JSONL records: {"id":"c0001","allocation":{...}} or
{"id":"greedy","policy":"greedy"}. All jobs in one queue share a frozen
config, replay, rate and duration. A slot is reused only after full draining,
normal shutdown, owned-process checks and terminal auditing.
"""
import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts/simulation'))
from calibrate_parallel import OwnedMonitor, load_collector, process_rows, slot_configuration, slot_report
from monitor import descendants
from run import code_identity, dump, ensure_ports_free, model_rates, single_phase_workload, validate_allocation_plan


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def jobs(path, config):
    """Keep memory bounded by streaming plans rather than creating a million futures."""
    with path.open() as source:
        sequence = 0
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or set(row)-{'id', 'policy', 'allocation', 'runtime_seed'}:
                raise ValueError(f'invalid job schema on line {line_number}')
            runtime_seed = row.get('runtime_seed')
            if 'runtime_seed' in row and (type(runtime_seed) is not int or not 0 <= runtime_seed < 2**64):
                raise ValueError(f'invalid runtime seed on line {line_number}')
            name = row.get('id', f'c{sequence+1:08d}')
            if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', name):
                raise ValueError(f'invalid job id on line {line_number}')
            policy = row.get('policy', 'fixed')
            if policy == 'fixed':
                plan = validate_allocation_plan(row.get('allocation'), config)
                if set(plan) != {'schema_version', 'placements'}:
                    raise ValueError(f'invalid allocation schema on line {line_number}')
                if {placement['model'] for placement in plan['placements']} != set(config['models']):
                    raise ValueError(f'all models require resources on line {line_number}')
            elif policy == 'greedy' and 'allocation' not in row:
                plan = None
            else:
                raise ValueError(f'invalid allocation policy on line {line_number}')
            sequence += 1
            job = {'sequence': sequence, 'id': name, 'policy': policy, 'allocation': plan}
            if runtime_seed is not None:
                job['runtime_seed'] = runtime_seed
            yield job


def static_config(path):
    config = json.loads(path.read_text())
    for key in ['latency_csv', 'batching_config']:
        config[key] = str((path.parent/config[key]).resolve())
    config['create_delay_ms'] = config['destroy_delay_ms'] = 0
    lifecycle = config.setdefault('model_lifecycle', {'overrides': []})
    lifecycle['load_ms'] = lifecycle['unload_ms'] = 0
    for override in lifecycle.get('overrides', []):
        override['load_ms'] = override['unload_ms'] = 0
    return config


def track_owned(active, rows):
    for slot in active.values():
        known = [pid for pid, started in slot['owned'].items()
                 if pid in rows and rows[pid]['started'] == started]
        roots = known + ([slot['process'].pid] if slot['process'].poll() is None else [])
        selected = descendants(rows, roots)
        slot['owned'].update({pid: rows[pid]['started'] for pid in selected})


def remaining_owned(slot, rows):
    known = [pid for pid, started in slot['owned'].items()
             if pid in rows and rows[pid]['started'] == started]
    return {str(pid): rows[pid] for pid in descendants(rows, known)}


def sample(monitor, active, check_guard=True):
    # Retire dead PID metadata on each sample so a long queue stays bounded.
    old_seen, old_cpu = dict(monitor.seen), dict(monitor.cpu_max)
    event = monitor.sample()
    monitor.queue_sample_count = getattr(monitor, 'queue_sample_count', 0)+1
    monitor.queue_sampling_errors = getattr(monitor, 'queue_sampling_errors', 0)+int('error' in event)
    if 'error' in event:
        raise RuntimeError('owned process monitor failed; memory guard cannot be verified')
    rows = {int(pid): value for pid, value in event['processes'].items()}
    track_owned(active, rows)
    retired = sum(value for pid, value in old_cpu.items()
                  if pid not in rows or rows[pid]['started'] != old_seen[pid])
    monitor.seen = {pid: row['started'] for pid, row in rows.items()}
    monitor.cpu_max = {pid: max(row['cpu_seconds'], old_cpu.get(pid, 0))
                       if old_seen.get(pid) == row['started'] else row['cpu_seconds']
                       for pid, row in rows.items()}
    # Preserve the monitor's aggregate totals without retaining every sample.
    if len(monitor.samples) > 2:
        previous, latest = monitor.samples[-2:]
        first = monitor.samples[0]
        first['rss_kib'] = max(first.get('rss_kib', 0), previous.get('rss_kib', 0))
        first['ps_cpu_percent'] = max(first.get('ps_cpu_percent', 0), previous.get('ps_cpu_percent', 0))
        monitor.samples[:] = [first, latest]
    if monitor.guard and check_guard:
        raise RuntimeError('owned RSS guard fired; queue stopped')
    return retired


def command(args, config_path, out, daemon_port, plan_path, runtime_seed=None):
    result = [sys.executable, str(ROOT/'scripts/simulation/run.py'), '--skip-build',
        '--config', str(config_path), '--output', str(out), '--port', str(daemon_port),
        '--seconds', str(args.seconds), '--warmup-seconds', '0', '--rate', str(args.rate),
        '--seed', str(args.seed), '--replay', str(args.replay), '--noise-cv', str(args.noise_cv),
        '--client-terminal-log', '--ready-timeout', str(args.ready_timeout), '--schedule-interval', '5',
        '--max-host-memory-gib', str(args.runner_memory_gib)]
    result += ['--experiment-fixed-allocation', str(plan_path)] if plan_path else ['--experiment-freeze-greedy']
    if getattr(args,'model_rates',None):
        result += ['--model-rates',str(args.model_rates)]
    if args.drain_seconds is not None:
        result += ['--drain-seconds', str(args.drain_seconds)]
    if runtime_seed is not None:
        result += ['--random-seed', str(runtime_seed)]
    return result


def execute(args, config, identity):
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    inputs = root/'inputs'; inputs.mkdir()
    shutil.copy2(args.jobs, inputs/'jobs.jsonl')
    shutil.copy2(args.replay, inputs/'replay.json')
    args.replay = inputs/'replay.json'
    dump(inputs/'configuration.json', config)
    expected = {'csv_sha256': digest(Path(config['latency_csv'])),
                'batching_sha256': digest(Path(config['batching_config'])),
                'replay_sha256': digest(args.replay)}
    if getattr(args,'model_rates',None):
        shutil.copy2(args.model_rates,inputs/'model-rates.json')
        args.model_rates = inputs/'model-rates.json'
        expected['model_rates_sha256'] = digest(args.model_rates)
    dump(root/'manifest.json', {'arguments': {key: str(value) if isinstance(value, Path) else value
                                             for key, value in vars(args).items()},
        'code_identity': identity, 'queue_source_sha256': digest(Path(__file__)),
        'jobs_sha256': digest(inputs/'jobs.jsonl'), 'replay_sha256': digest(args.replay),
        'configuration': config, 'profile_identity': expected,
        'design': 'Streaming bounded slots; immediate refill after complete drain, shutdown and audit. '
                  'All slots share one frozen trace. No synchronized cohort barrier or online pruning.'})
    source = iter(jobs(inputs/'jobs.jsonl', config))
    free = deque(range(args.concurrency))
    active, runners, logs = {}, [], []
    monitor = OwnedMonitor(root/'host_metrics.jsonl', runners, args.max_host_memory_gib*1024**2)
    collector = load_collector()
    started, last_sample, changed = time.monotonic(), -math.inf, time.monotonic()
    occupied_seconds, retired_cpu = 0.0, 0.0
    completed, failed_trials, launched, best = 0, 0, 0, None
    exhausted, error = False, None
    events = (root/'queue-events.jsonl').open('w')
    results = (root/'results.jsonl').open('w')

    def record(event, **values):
        nonlocal changed, occupied_seconds
        now = time.monotonic()
        occupied_seconds += (now-changed)*len(active)
        changed = now
        events.write(json.dumps({'event': event, 'elapsed_seconds': now-started,
            'timestamp_ms': round(time.time()*1000), 'occupied_slots_before_event': len(active), **values})+'\n')
        events.flush()

    def save_summary(finished=False, host=None, remaining=None):
        wall = time.monotonic()-started
        dump(root/'summary.json', {'finished': finished, 'error': error,
            'input_exhausted': exhausted, 'launched_trials': launched, 'completed_trials': completed,
            'complete_trials_with_application_failures': failed_trials,
            'concurrency_limit': args.concurrency, 'wall_seconds': wall,
            'occupied_slot_seconds': occupied_seconds,
            'slot_occupancy_fraction': occupied_seconds/(args.concurrency*wall) if wall else 0,
            'occupancy_scope': 'Slot reserved from launch through drain, shutdown and audit; includes startup/cleanup. '
                               'Finite queue tail and startup reduce utilization.',
            'best_complete_successful_trial': best, 'host': host,
            'remaining_owned_processes': remaining,
            'scope': 'Minimum observed among successfully completed queued trials only; no assertion of exhaustive coverage or expected-JCT optimality.'})

    try:
        while active or not exhausted:
            if time.monotonic()-last_sample >= args.sample_seconds:
                retired_cpu += sample(monitor, active)
                last_sample = time.monotonic()
            for index, slot in list(active.items()):
                proc = slot['process']
                if proc.poll() is None:
                    if time.monotonic()-slot['started'] > args.trial_timeout:
                        raise TimeoutError(f'trial {slot["job"]["sequence"]} wall timeout')
                    continue
                slot['log'].close()
                logs.remove(slot['log'])
                leftover = remaining_owned(slot, process_rows())
                if leftover:
                    dump(slot['folder']/'leftover-processes.json', leftover)
                    raise RuntimeError('owned processes survive runner exit; slot cannot be reused')
                report = slot_report(slot['out'], proc.returncode, slot['runner_log'], collector)
                report['all_owned_processes_gone'] = True
                dump(slot['out']/'calibration-terminal.json', report)
                row = {'sequence': slot['job']['sequence'], 'id': slot['job']['id'],
                       'slot': index, 'policy': slot['job']['policy'],
                       'trial_wall_seconds': time.monotonic()-slot['started'], **report}
                if 'runtime_seed' in slot['job']:
                    row['runtime_seed'] = slot['job']['runtime_seed']
                results.write(json.dumps(row)+'\n'); results.flush()
                if not report['measurement_complete']:
                    raise RuntimeError(f'incomplete trial {slot["job"]["sequence"]}: {report.get("collection_error")}')
                manifest = json.loads((slot['out']/'manifest.json').read_text())
                if manifest['code_identity'] != identity or any(manifest[key] != value for key, value in expected.items()):
                    raise RuntimeError('runtime source/binary/profile identity changed during queue execution')
                if 'runtime_seed' in slot['job']:
                    randomness = manifest.get('randomness', {})
                    if randomness.get('root_seed') != slot['job']['runtime_seed'] or randomness.get('schema_version') != 1:
                        raise RuntimeError('runner did not honor the explicit runtime seed')
                    if digest(slot['out']/'random-inputs.json') != randomness.get('inputs_sha256'):
                        raise RuntimeError('external random-input manifest hash mismatch')
                if report['server_summary']['scheduler']['successful_schedules'] != 1:
                    raise RuntimeError('fixed allocation was applied more than once')
                completed += 1
                failures = report['client_metrics']['failed_requests']
                failed_trials += bool(failures)
                if not failures and (best is None or report['server_summary']['mean_jct_ms'] < best['mean_jct_ms']):
                    best = {'sequence': row['sequence'], 'id': row['id'], 'output': row['output'],
                            'mean_jct_ms': report['server_summary']['mean_jct_ms'], 'policy': row['policy']}
                # Record the occupied period before retiring the slot.
                record('completed', slot=index, sequence=row['sequence'])
                del active[index]; runners.remove(proc); free.append(index)
                save_summary()
            while free and not exhausted:
                try:
                    job = next(source)
                except StopIteration:
                    exhausted = True
                    record('input_exhausted')
                    break
                index = free.popleft()
                slot_config, port = slot_configuration(config, args.base_port, index)
                ensure_ports_free([port]+[node['port'] for node in slot_config['nodes']])
                folder = root/f'trial-{job["sequence"]:08d}-{job["id"]}'; folder.mkdir()
                config_path = folder/'configuration.json'; dump(config_path, slot_config)
                plan_path = folder/'allocation.json' if job['allocation'] is not None else None
                if plan_path:
                    dump(plan_path, job['allocation'])
                out = folder/'run'; runner_log = folder/'runner.log'; log = runner_log.open('w')
                cmd = command(args, config_path, out, port, plan_path, job.get('runtime_seed'))
                logs.append(log)
                proc = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
                runners.append(proc)
                record('launching', slot=index, sequence=job['sequence'], id=job['id'],runner_pid=proc.pid)
                active[index] = {'job': job, 'process': proc, 'log': log, 'runner_log': runner_log,
                    'out': out, 'folder': folder, 'started': time.monotonic(), 'owned': {}}
                dump(folder/'job.json', {'job': job, 'slot': index, 'runner_pid':proc.pid,'command': cmd})
                launched += 1
            if active:
                time.sleep(.05)
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        record('stopped', reason=error)
        monitor.stop_runners()
    finally:
        # No new work is taken after a signal/guard/error. Existing runners get
        # their normal node/worker cleanup budget before this process exits.
        deadline = time.monotonic()+config.get('control_timeout_ms', 300000)/1000+45
        for proc in list(runners):
            try:
                proc.wait(timeout=max(.1, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                error = (error+'; ' if error else '')+f'runner {proc.pid} cleanup timeout'
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
        for log in logs:
            if not log.closed:
                log.close()
        try:
            retired_cpu += sample(monitor, active, check_guard=False)
            remaining = monitor.remaining()
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            remaining = {'unverified': str(exc)}
        host = monitor.close()
        host['sampled_process_cpu_seconds_lower_bound'] += retired_cpu
        host['samples'] = getattr(monitor, 'queue_sample_count', 0)
        host['sampling_errors'] = getattr(monitor, 'queue_sampling_errors', 0)
        host.pop('mean_owned_rss_kib', None)
        host.pop('mean_ps_cpu_percent', None)
        host['sample_summary_scope'] = 'Bounded in-memory peak summary; complete sequential samples in host_metrics.jsonl. Only peak RSS/CPU and CPU-seconds are interpreted.'
        record('finished' if not error else 'aborted')
        save_summary(finished=not error and exhausted and completed==launched and not remaining,
                     host=host, remaining=remaining)
        events.close(); results.close()
        source.close()
    if error or remaining:
        raise RuntimeError(error or 'owned-process cleanup cannot be verified')
    return json.loads((root/'summary.json').read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['jobs', 'config', 'replay', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--base-port', type=int, default=17300)
    parser.add_argument('--seconds', type=int, default=30)
    parser.add_argument('--rate', type=float, required=True, help='initial samples/second/model, matching the replay')
    parser.add_argument('--model-rates',type=Path,help='optional explicit model-to-samples/s JSON; overrides uniform initial demand')
    parser.add_argument('--seed', type=int, default=20260923)
    parser.add_argument('--noise-cv', type=float, default=.1)
    parser.add_argument('--max-host-memory-gib', type=float, default=12)
    parser.add_argument('--runner-memory-gib', type=float, default=4)
    parser.add_argument('--sample-seconds', type=float, default=2)
    parser.add_argument('--ready-timeout', type=int, default=900)
    parser.add_argument('--trial-timeout', type=int, default=1800)
    parser.add_argument('--drain-seconds', type=int)
    args = parser.parse_args()
    for name in ['jobs', 'config', 'replay', 'output']:
        setattr(args, name, getattr(args, name).resolve())
    if min(args.concurrency, args.seconds, args.ready_timeout, args.trial_timeout) <= 0:
        parser.error('concurrency and durations must be positive')
    if any(not math.isfinite(value) or value <= 0 for value in
           [args.rate, args.max_host_memory_gib, args.runner_memory_gib, args.sample_seconds]):
        parser.error('rates, memory limits and sampling period must be finite and positive')
    if not math.isfinite(args.noise_cv) or not 0 <= args.noise_cv <= 1:
        parser.error('noise CV must be in [0,1]')
    if args.drain_seconds is not None and args.drain_seconds <= 0:
        parser.error('drain timeout must be positive')
    config = static_config(args.config)
    if args.base_port < 1024 or args.base_port+(len(config['nodes'])+1)*args.concurrency > 65536:
        parser.error('slot port range must fit unprivileged TCP ports')
    rates = model_rates(config['models'],args.rate,args.model_rates)
    single_phase_workload(config['models'], rates, args.seconds, args.seed, (1,32), json.loads(args.replay.read_text()))
    def stop(signum, _frame):
        raise KeyboardInterrupt(f'queue interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    result = execute(args, config, code_identity())
    print(json.dumps({key: result[key] for key in ['finished','completed_trials','wall_seconds','slot_occupancy_fraction','best_complete_successful_trial']}, indent=2))


if __name__ == '__main__':
    main()
