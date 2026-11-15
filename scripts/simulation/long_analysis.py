"""Post-run stability and resource audit; never changes the running scheduler."""
import argparse
import bisect
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import re


def timestamp_ms(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp() * 1000


def quantile(sorted_values, q):
    return sorted_values[max(0, math.ceil(q * len(sorted_values)) - 1)] if sorted_values else None


def task_metrics(tasks):
    times = sorted(t['completed_at'] - t['received_at'] for t in tasks if t['status'] == 'completed')
    count = len(tasks)
    return {'received': count, 'completed': len(times),
            'failed': sum(t['status'] == 'failed' for t in tasks),
            'pending': sum(t['status'] in ('received', 'processing') for t in tasks),
            'slo_percent': {str(limit): 100 * bisect.bisect_right(times, limit) / count if count else None for limit in (250, 400)},
            'p50_ms': quantile(times, .5), 'p95_ms': quantile(times, .95), 'p99_ms': quantile(times, .99),
            'max_ms': times[-1] if times else None, 'over_5s': len(times) - bisect.bisect_right(times, 5000)}


def log_events(path):
    if not path.exists():
        return
    with path.open() as source:
        for line in source:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if 'timestamp' in row and 'fields' in row:
                yield {'timestamp_ms': timestamp_ms(row['timestamp']), **row['fields']}


def resource_audit(run, config, start, end):
    violations, profile_events, initial_profiles, runtime_creates, runtime_destroys = [], [], Counter(), Counter(), Counter()
    paired, pending_lifecycle, reserved_total, released_total = 0, {}, 0, 0
    nodes, workers = {}, {}
    profile_timeline_valid = True
    for node in config['nodes']:
        name = node['name']
        active, used, peak, created, destroyed = {}, Counter(), Counter(), Counter(), Counter()
        for e in log_events(run / f'{name}.log'):
            event, when = e.get('event', ''), e['timestamp_ms']
            if event in ('resource_reserved', 'resource_released'):
                reservation = e['reservation_id']
                amount = {key: int(value) for key, value in re.findall(r'(ce|memory|ai_core|ai_cpu): (\d+)', e['amount'])}
                if event == 'resource_reserved':
                    reserved_total += 1
                    if reservation in active:
                        violations.append(f'{name}: duplicate reservation {reservation}')
                    active[reservation] = amount
                    used.update(amount)
                else:
                    released_total += 1
                    prior = active.pop(reservation, None)
                    if prior != amount:
                        violations.append(f'{name}: unmatched resource release {reservation}')
                    used.subtract(amount)
                for resource, capacity in node['resource'].items():
                    peak[resource] = max(peak[resource], used[resource])
                    if not 0 <= used[resource] <= capacity * node['gpus']:
                        violations.append(f'{name}: {resource} capacity violation at {when}')
            worker_id = e.get('id', e.get('worker_id'))
            if event.endswith('_started') and event.startswith(('create_', 'destroy_', 'model_load_', 'model_unload_')):
                key = (name, worker_id, event[:-8])
                if key in pending_lifecycle:
                    violations.append(f'{name}: repeated lifecycle start {key}')
                pending_lifecycle[key] = e
            if event.endswith('_finished') and event.startswith(('create_', 'destroy_', 'model_load_', 'model_unload_')):
                prior = pending_lifecycle.pop((name, worker_id, event[:-9]), None)
                if prior is None:
                    violations.append(f'{name}: unpaired {event} {worker_id}')
                else:
                    paired += 1
                    if when - prior['timestamp_ms'] < prior['delay_ms'] - 2:
                        violations.append(f'{name}: lifecycle delay shorter than configured {event} {worker_id}')
            if event == 'create_started':
                profile = e['slice']
                if (name, worker_id) in workers:
                    profile_timeline_valid = False
                    violations.append(f'{name}: repeated worker id; allocation timeline ambiguous {worker_id}')
                workers[(name, worker_id)] = profile
                created[profile] += 1
                profile_events.append((when, name, profile, 1))
                if when < start:
                    initial_profiles[profile] += 1
                elif when <= end:
                    runtime_creates[profile] += 1
            elif event == 'destroy_started':
                if start <= when <= end:
                    runtime_destroys[workers.get((name, worker_id), 'unknown')] += 1
            elif event == 'worker_released':
                profile = workers.pop((name, worker_id), None)
                if profile is None:
                    profile_timeline_valid = False
                    violations.append(f'{name}: release without create {worker_id}')
                else:
                    destroyed[profile] += 1
                    profile_events.append((when, name, profile, -1))
        if active or any(used.values()):
            violations.append(f'{name}: unreleased capacity at shutdown')
        nodes[name] = {'created': dict(created), 'released': dict(destroyed), 'peak_reserved': dict(peak), 'remaining_reserved': dict(used)}
    if pending_lifecycle:
        violations.append(f'{len(pending_lifecycle)} lifecycle operations lack completion')
    if workers:
        profile_timeline_valid = False
        violations.append(f'{len(workers)} workers lack release events')
    profiles = sorted({profile for _, _, profile, _ in profile_events})
    counts, points, node_points = Counter(), [], []
    per_node = {n['name']: Counter() for n in config['nodes']}
    for when, name, profile, delta in sorted(profile_events):
        if when > end:
            break
        if when >= start and not points:
            points.append([0, *[counts[p] for p in profiles]])
        counts[profile] += delta
        per_node[name][profile] += delta
        if when >= start:
            t = (when - start) / 1000
            points.append([t, *[counts[p] for p in profiles]])
            node_points.append([t, name, profile, delta])
    if not points:
        points.append([0, *[counts[p] for p in profiles]])
    points.append([(end-start)/1000, *[counts[p] for p in profiles]])
    return {'violations': violations, 'timing_pairs_checked': paired, 'reservations': reserved_total, 'releases': released_total,
            'initial_profiles': dict(initial_profiles), 'runtime_creates': dict(runtime_creates), 'runtime_destroys': dict(runtime_destroys),
            'nodes': nodes, 'profiles': profiles,
            'profile_timeline_valid': profile_timeline_valid,
            'profile_points': points if profile_timeline_valid else [],
            'profile_changes': node_points if profile_timeline_valid else []}


def analyze(run, summary=None, config=None, tasks=None, workload=None):
    read = lambda filename: json.loads((run / filename).read_text())
    summary = summary or read('summary.json')
    config = config or read('simulation.json')
    tasks = tasks if tasks is not None else read('tasks.json')
    workload = workload or read('workload.json')
    start, end = summary['measurement_started_at_ms'], summary['measurement_finished_at_ms']
    phases, recovery = [], []
    for index, phase in enumerate(workload['phases']):
        lo = phase['start_seconds']
        hi = workload['phases'][index+1]['start_seconds'] if index+1 < len(workload['phases']) else workload['duration_seconds']
        selected = [t for t in tasks if lo*1000 <= t['received_at']-start < hi*1000]
        transient = [t for t in selected if t['received_at']-start < (lo+120)*1000]
        steady = [t for t in selected if t['received_at']-start >= (lo+120)*1000]
        phases.append({'name': phase['name'], 'label': phase.get('label', phase['name']), 'start_seconds': lo, 'end_seconds': hi,
                       'rates': phase['rates'], **task_metrics(selected), 'first_120s': task_metrics(transient), 'after_120s': task_metrics(steady),
                       'models': {m: task_metrics([t for t in selected if t['model'] == m]) for m in config['models']}})
        if index:
            buckets = [task_metrics([t for t in selected if (lo+offset)*1000 <= t['received_at']-start < (lo+offset+10)*1000]) for offset in range(0, int(hi-lo), 10)]
            row = {'switch_seconds': lo, 'label': phase.get('label', phase['name'])}
            for threshold in ('250', '400'):
                first = next((i for i in range(len(buckets)-2) if all(b['received'] and b['slo_percent'][threshold] >= 99 for b in buckets[i:i+3])), None)
                row[f'first_30s_at_99pct_{threshold}_starts_after_s'] = first*10 if first is not None else None
            recovery.append(row)
    cycles = [json.loads(line) for line in (run/'scheduler/schedule_events.jsonl').open()]
    runtime = [c for c in cycles if not c['initial'] and c['started_at_ms'] < start + workload['duration_seconds']*1000]
    apply = sorted(c['apply_ms']/1000 for c in runtime)
    gaps = [e for e in log_events(run/'scheduler/lifecycle_events.jsonl') if e.get('event') == 'model_queue_without_worker' and start <= e['timestamp_ms'] <= end]
    samples = []
    monitor_errors = []
    if (run/'host_metrics.jsonl').exists():
        for line in (run/'host_metrics.jsonl').open():
            row = json.loads(line)
            if 'error' in row:
                monitor_errors.append(row)
            elif start <= row['timestamp_ms'] <= start + workload['duration_seconds']*1000:
                samples.append([(row['timestamp_ms']-start)/1000, row['rss_kib']/1024, row['ps_cpu_percent'], row['process_count']])
    rss = [r[1] for r in samples]
    result = {'run': run.name, 'workload': workload['name'], 'multiplier': workload['multiplier'],
              'overall': task_metrics(tasks), 'models': {m: task_metrics([t for t in tasks if t['model']==m]) for m in config['models']},
              'phases': phases, 'recovery': recovery, 'resource_audit': resource_audit(run, config, start, end),
              'scheduling': {'cycles': len(runtime), 'failures': sum(not c['success'] for c in runtime),
                             'apply_p50_s': quantile(apply,.5), 'apply_p95_s': quantile(apply,.95), 'apply_max_s': max(apply, default=None),
                             'no_worker_queue_events': len(gaps)},
              'host': {'points': samples, 'errors': monitor_errors, 'peak_rss_mib': max(rss, default=None),
                       'first_rss_mib': rss[0] if rss else None, 'last_rss_mib': rss[-1] if rss else None,
                       'note': 'Host process RSS, including retained request records and runner replay. CPU is ps pcpu summed across processes; not GPU utilization.'},
              'client': {'sent': summary['client_sent_tasks'], 'failed': summary['client_failed_tasks'], 'max_send_lag_ms': summary.get('max_send_lag_ms')},
              'timing': {key: summary.get(key) for key in ['startup_seconds','sending_seconds','sending_and_drain_seconds','shutdown_seconds','total_wall_seconds']}}
    (run/'long_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = analyze(args.run.resolve())
    print(json.dumps({k:v for k,v in result.items() if k not in ('models','phases','resource_audit','host')}, indent=2))
    print('resource violations:', result['resource_audit']['violations'])
