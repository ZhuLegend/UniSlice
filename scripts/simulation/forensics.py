"""Read-only log diagnostics, including runs without a final HTTP task snapshot.

Internal queue records are not HTTP receive/completion records. In particular,
this module does not manufacture a full-run SLO denominator for aborted runs.
"""
import argparse
import bisect
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re

from long_analysis import log_events, resource_audit, timestamp_ms


def histogram_quantile(counts, q):
    target = math.ceil(sum(counts.values()) * q)
    if not target:
        return None
    seen = 0
    for value, count in sorted(counts.items()):
        seen += count
        if seen >= target:
            return value


def queue_timings(paths, task_statuses=None):
    records, counts = {}, Counter()
    queued, execution, total = Counter(), Counter(), Counter()
    tail_count = tail_queue = tail_total = 0
    first = last = None
    for path in sorted(paths):
        for e in log_events(path):
            event, task, when = e.get('event'), e.get('task_id'), e['timestamp_ms']
            # Producer's integer clock makes arrival / assignment intervals comparable.
            when = e.get('timestamp_ms', when)
            counts[event] += 1
            row = records.setdefault(task, [None, None, None])
            if event == 'task_arrived':
                first = when if first is None else min(first, when)
                last = when if last is None else max(last, when)
                if row[0] is not None:
                    counts['repeated_arrival'] += 1
                row[0] = min(row[0], when) if row[0] is not None else when
            elif event == 'task_assigned':
                # First claim; a retry would not be mistaken for pure queue waiting.
                row[1] = min(row[1], when) if row[1] is not None else when
            elif event == 'compute_completed':
                row[2] = max(row[2], when) if row[2] is not None else when
    # Concurrent writers may emit assignment before the corresponding arrival
    # line. Join by task ID first instead of interpreting file order as causality.
    for arrived, assigned, completed in records.values():
        if completed is None:
            continue
        if arrived is None or assigned is None:
            counts['completion_without_matching_arrival_assignment'] += 1
            continue
        duration, wait, after_claim = completed - arrived, assigned - arrived, completed - assigned
        if min(duration, wait, after_claim) < 0:
            counts['negative_clock_interval'] += 1
            continue
        total[round(duration)] += 1
        execution[round(after_claim)] += 1
        queued[round(wait)] += 1
        if duration > 5000:
            tail_count += 1
            tail_queue += wait
            tail_total += duration
    unfinished = [task for task, row in records.items() if row[0] is not None and row[2] is None]
    unfinished_statuses = Counter((task_statuses or {}).get(task, 'unknown') for task in unfinished)
    return {'events': dict(counts), 'unmatched_active_at_log_end': len(unfinished),
            'unfinished_compute_http_statuses': dict(unfinished_statuses),
            'first_queue_arrival_ms': first, 'last_queue_arrival_ms': last,
            'matched_completed': sum(total.values()),
            'queue_wait_p99_ms': histogram_quantile(queued, .99),
            'after_assignment_p99_ms': histogram_quantile(execution, .99),
            'queue_to_compute_p99_ms': histogram_quantile(total, .99),
            'over_5s_count': tail_count,
            'over_5s_queue_fraction': tail_queue / tail_total if tail_total else None,
            'note': 'Internal queue to compute events only; not end-to-end HTTP latency or full-run SLO. Missing compute completion includes explicitly failed/cancelled HTTP requests; it does not alone prove orphaned work.'}


def model_binding_audit(node_paths, performance_paths):
    """Compare completed batches with physical model load/unload events."""
    transitions = {}
    for path in node_paths:
        for event in log_events(path):
            kind = event.get('event')
            if kind not in ('model_load_finished', 'model_unload_started'):
                continue
            worker = event.get('worker_id', event.get('id'))
            transitions.setdefault(worker, []).append((event['timestamp_ms'],
                event.get('model') if kind == 'model_load_finished' else None))
    timeline = {}
    for worker, changes in transitions.items():
        changes.sort(key=lambda row: row[0])
        timeline[worker] = ([row[0] for row in changes], [row[1] for row in changes])
    counts, examples = Counter(), []
    for path in performance_paths:
        for event in log_events(path):
            if 'model_name' not in event or 'worker_id' not in event:
                continue
            counts['records_checked'] += 1
            worker, when, model = event['worker_id'], event['timestamp_ms'], event['model_name']
            if worker not in timeline:
                category, physical = 'unknown_worker', None
            else:
                times, models = timeline[worker]
                index = bisect.bisect_right(times, when) - 1
                physical = models[index] if index >= 0 else None
                if physical == model:
                    counts['matched'] += 1
                    continue
                # JSON timestamps from different processes have sub-ms edges.
                # Keep boundary cases separate instead of silently passing them.
                edge = any(abs(times[i] - when) <= 2 and
                           (models[i] == model or (i > 0 and models[i-1] == model))
                           for i in (index, index+1) if 0 <= i < len(times))
                category = 'boundary_ambiguous' if edge else ('not_loaded' if physical is None else 'model_mismatch')
            counts[category] += 1
            if len(examples) < 10:
                examples.append({'worker_id': worker, 'timestamp_ms': when,
                                 'reported_model': model, 'physical_model': physical,
                                 'category': category})
    return {'counts': dict(counts), 'examples': examples,
            'note': 'Batch completion log versus physical model lifecycle. This checks recorded model ownership, not tensor correctness; <=2ms transition edges are reported separately.'}


ERRORS = {'bind_address_in_use': 'Address already in use',
          'cannot_assign_address': "Can't assign requested address",
          'worker_readiness_deadline': 'worker readiness deadline exceeded',
          'connection_reset': 'Connection reset by peer',
          'connection_refused': 'Connection refused',
          'rpc_connect_timeout': 'Timeout connecting to RPC peer'}
STAMP = re.compile(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z')


def raw_lines(path):
    with path.open(errors='replace') as source:
        yield from source


def error_evidence(paths, origin):
    counts, examples = Counter(), {}
    for path in paths:
        nearby = None
        for line_number, line in enumerate(raw_lines(path), 1):
            match = STAMP.search(line)
            if match:
                nearby = match[0]
            for category, needle in ERRORS.items():
                if needle not in line:
                    continue
                counts[category] += 1
                example = {'file': path.name, 'line': line_number,
                           'timestamp': match[0] if match else None,
                           'preceding_timestamp': nearby,
                           'seconds_after_origin': (timestamp_ms(match[0])-origin)/1000 if match else None,
                           'text': line.strip()[:600]}
                saved = examples.setdefault(category, [])
                saved.append(example)
                saved.sort(key=lambda r: r['seconds_after_origin'] if r['seconds_after_origin'] is not None else float('inf'))
                del saved[3:]
    return {'counts_of_matching_log_lines': dict(counts), 'first_examples': examples,
            'note': 'Counts are log lines, not distinct failed requests. Undated lines have no event time; preceding_timestamp is only context, not an onset estimate.'}


def analyze(run):
    read = lambda name: json.loads((run/name).read_text())
    startup, config = read('startup.json'), read('simulation.json')
    summary = read('summary.json') if (run/'summary.json').exists() else None
    task_statuses = {task['task_id']: task['status'] for task in read('tasks.json')} if (run/'tasks.json').exists() else None
    origin = summary['measurement_started_at_ms'] if summary else startup['ready_at_ms']
    host = [json.loads(line) for line in raw_lines(run/'host_metrics.jsonl')] if (run/'host_metrics.jsonl').exists() else []
    host = [r for r in host if 'rss_kib' in r]
    last = max((r['timestamp_ms'] for r in host), default=(summary or {}).get('measurement_finished_at_ms', startup['ready_at_ms']))
    cycles = [json.loads(line) for line in raw_lines(run/'scheduler/schedule_events.jsonl')]
    runtime = [c for c in cycles if not c['initial']]
    csv_counts = Counter()
    terminal_log = run/'scheduler/task_terminal_log.csv'
    if not terminal_log.exists():
        terminal_log = run/'scheduler/task_completion_log.csv'
    for row in csv.DictReader(raw_lines(terminal_log)):
        csv_counts[row['model_name']] += 1
    result = {'run': run.name, 'origin_ms': origin,
              'origin_kind': 'sender measurement start' if summary else 'startup ready; sender start unavailable',
              'has_final_task_snapshot': (run/'tasks.json').exists(),
              'has_client_final_summary': (run/'measurement/final_summary.json').exists(),
              'runtime_schedules': len(runtime), 'runtime_schedule_failures': sum(not c['success'] for c in runtime),
              'runtime_apply_seconds': [c['apply_ms']/1000 for c in runtime],
              'queue': queue_timings(run.glob('scheduler/task_tracking.jsonl.*'), task_statuses),
              'model_binding': model_binding_audit(
                  [run/f'{node["name"]}.log' for node in config['nodes']],
                  run.glob('scheduler/worker_perf.jsonl.*')),
              'csv_terminal_rows_by_model': dict(csv_counts),
              'csv_note': 'New terminal logs include task ID, status and error; legacy completion logs lack them. Neither captures active requests.',
              'errors': error_evidence(sorted(run.glob('*.log')), origin),
              'host': {'available': bool(host),
                       'peak_rss_gib': max((r['rss_kib'] for r in host), default=0)/1024**2 if host else None,
                       'max_processes': max((r['process_count'] for r in host), default=None),
                       'peak_ps_cpu_percent': max((r['ps_cpu_percent'] for r in host), default=None),
                       'first_sample_ms': min((r['timestamp_ms'] for r in host), default=None),
                       'last_sample_ms': max((r['timestamp_ms'] for r in host), default=None)},
              'resource_audit': resource_audit(run, config, origin, last)}
    (run/'forensics.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    for run in parser.parse_args().runs:
        result = analyze(run.resolve())
        print(run, json.dumps({k: result[k] for k in ('queue', 'runtime_schedules', 'runtime_schedule_failures', 'host')}, ensure_ascii=False), flush=True)
