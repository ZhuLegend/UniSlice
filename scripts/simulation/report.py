#!/usr/bin/env python3
"""Build an offline, interactive timeline from a completed simulation run."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import tomllib

HERE = Path(__file__).resolve().parent


def timestamp_ms(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp() * 1000


def slice_units(kind, profile):
    # These are compute-resource units, not an equivalence of GPU performance.
    if kind == 'nvidia' and re.fullmatch(r'[12347]g\.(10|20|40|80)gb', profile):
        return int(profile.split('g.')[0])
    if kind == 'huawei' and profile in {'vir05_1c_16g', 'vir10_3c_32g'}:
        return {'vir05_1c_16g': 1, 'vir10_3c_32g': 2}[profile]
    raise ValueError(f'No verified resource-unit mapping for {kind}/{profile}')


def resource_timeline(node, models, log_path, start_ms, end_ms):
    events = []
    for line in log_path.read_text().splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        fields = item.get('fields', {})
        if fields.get('event') in {'create_started', 'model_loaded', 'model_unloaded', 'worker_released', 'model_unload_finished'}:
            events.append((timestamp_ms(item['timestamp']), fields))
    events.sort(key=lambda row: row[0])
    if not any(f['event'] == 'model_loaded' for _, f in events):
        raise ValueError(f'{log_path}: model binding events are missing; rerun with allocation logging enabled')
    workers = {}
    points = []
    units_per_card = 7 if node['kind'] == 'nvidia' else 4
    capacity = node['gpus'] * (node['resource']['ce'] if node['kind'] == 'nvidia' else node['resource']['ai_core'] / 5)
    changes = []

    def values():
        totals = [0] * (len(models) + 1)
        for worker in workers.values():
            owner = worker['model']
            index = models.index(owner) if owner in models else len(models)
            totals[index] += worker['units']
        if sum(totals) > capacity + 1e-8:
            raise ValueError(f'{node["name"]}: logged reservations exceed configured compute capacity')
        return totals

    for when, fields in events:
        if when > end_ms:
            break
        if when >= start_ms and not points:
            points.append([0, *values()])
        event, worker_id = fields['event'], fields.get('id', fields.get('worker_id'))
        if event == 'create_started':
            if worker_id in workers:
                raise ValueError(f'duplicate create event for worker {worker_id}')
            workers[worker_id] = {'units': slice_units(node['kind'], fields['slice']), 'model': None}
        elif worker_id not in workers:
            raise ValueError(f'{event} without reservation for worker {worker_id}')
        elif event == 'model_loaded':
            workers[worker_id]['model'] = fields['model']
        elif event in {'model_unloaded', 'model_unload_finished'}:
            workers[worker_id]['model'] = None
        elif event == 'worker_released':
            del workers[worker_id]
        if when >= start_ms:
            t = round((when - start_ms) / 1000, 6)
            value = values()
            if points[-1][1:] != value:
                points.append([t, *value])
                changes.append({'time': t, 'event': event, 'model': fields.get('model')})
    if not points:
        points.append([0, *values()])
    points.append([round((end_ms - start_ms) / 1000, 6), *values()])
    return {'name': node['name'], 'kind': node['kind'], 'capacity': capacity,
            'units_per_card': units_per_card,
            'unit': '1g.10gb' if node['kind'] == 'nvidia' else 'vir05_1c_16g',
            'points': points, 'changes': changes}


def resource_timelines(config, models, run, start, end):
    """Keep request metrics usable when failed worker creation leaves an incomplete trace.

    Never publish a partial-node sum as the full resource pool. The reservation
    ledger audit remains separate from the worker/model allocation timeline.
    """
    resources, warnings = [], []
    for node in config['nodes']:
        try:
            resources.append(resource_timeline(node, models, run / f'{node["name"]}.log', start, end))
        except ValueError as exc:
            warnings.append(f'{node["name"]}: {exc}')
    return ([], warnings) if warnings else (resources, [])


def build_data(run):
    config = json.loads((run / 'simulation.json').read_text())
    summary = json.loads((run / 'summary.json').read_text())
    manifest = json.loads((run / 'manifest.json').read_text())
    scheduler_config = tomllib.loads((run / 'scheduler.toml').read_text())
    client_path = run / 'measurement/final_summary.json'
    client = json.loads(client_path.read_text()) if client_path.exists() else None
    role_path = run / 'analysis-role.json'
    analysis_role = json.loads(role_path.read_text()) if role_path.exists() else None
    tasks = json.loads((run / 'tasks.json').read_text())
    models = config['models']
    start = summary.get('measurement_started_at_ms')
    end = summary.get('measurement_finished_at_ms')
    if start is None or end is None:
        raise ValueError('Run lacks precise measurement boundaries; rerun with the current runner')
    rows = []
    for task in tasks:
        received = (task['received_at'] - start) / 1000
        finished = (task['completed_at'] - start) / 1000 if task['completed_at'] is not None else None
        if not 0 <= received <= (end - start) / 1000:
            raise ValueError('Task lies outside the measurement interval')
        status = {'completed': 1, 'failed': 2, 'received': 0, 'processing': 0}[task['status']]
        rows.append([models.index(task['model']), received, finished, task['batch_size'], status])
    schedules = []
    path = run / 'scheduler/scheduler_stats_log.csv'
    if path.exists():
        for row in csv.DictReader(path.open()):
            finish = (int(row['timestamp']) - start) / 1000
            begin = finish - int(row['cycle_duration_ms']) / 1000
            if finish >= 0 and begin <= (end - start) / 1000:
                schedules.append([max(0, begin), finish, row['success'] == 'true'])
    workload_path = run / 'workload.json'
    workload = json.loads(workload_path.read_text()) if workload_path.exists() else None
    detailed = []
    path = run / 'scheduler/schedule_events.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            row['start'] = (row['started_at_ms'] - start) / 1000
            row['end'] = (row['finished_at_ms'] - start) / 1000
            detailed.append(row)
    lifecycle = []
    for node in config['nodes']:
        for line in (run / f'{node["name"]}.log').read_text().splitlines():
            try: event = json.loads(line)
            except json.JSONDecodeError: continue
            fields = event.get('fields', {})
            if fields.get('event') in {'create_started', 'create_finished', 'destroy_started', 'destroy_finished', 'model_load_started', 'model_load_finished', 'model_unload_started', 'model_unload_finished'}:
                lifecycle.append({'time': (timestamp_ms(event['timestamp']) - start) / 1000,
                                  'node': node['name'], **fields})
    scheduler_log = run / 'scheduler/lifecycle_events.jsonl'
    if scheduler_log.exists():
        for line in scheduler_log.read_text().splitlines():
            try: event = json.loads(line)
            except json.JSONDecodeError: continue
            fields = event.get('fields', {})
            if fields.get('event') in {'worker_drain_started', 'worker_drain_finished', 'model_queue_without_worker'}:
                lifecycle.append({'time': (timestamp_ms(event['timestamp']) - start) / 1000, 'node': fields.get('node_id'), **fields})
    lifecycle.sort(key=lambda e: e['time'])
    resources, resource_warnings = resource_timelines(config, models, run, start, end)
    # One plot per accelerator kind, preserving individual node timelines in raw_resources.
    grouped = []
    for kind in sorted({r['kind'] for r in resources}):
        group = [r for r in resources if r['kind'] == kind]
        if len(group) == 1:
            grouped.extend(group)
            continue
        times = sorted({p[0] for r in group for p in r['points']})
        cursors = [0] * len(group)
        points = []
        for t in times:
            totals = [0] * (len(models) + 1)
            for i, r in enumerate(group):
                while cursors[i]+1 < len(r['points']) and r['points'][cursors[i]+1][0] <= t:
                    cursors[i] += 1
                totals = [a+b for a,b in zip(totals, r['points'][cursors[i]][1:])]
            points.append([t, *totals])
        grouped.append({**group[0], 'name': f'{kind} · {len(group)} 个资源节点',
                        'capacity': sum(r['capacity'] for r in group), 'points': points,
                        'changes': [c for r in group for c in r['changes']]})
    diagnostics = None
    if workload and workload.get('name') == 'long-cycle-multi-slice':
        from long_analysis import analyze
        diagnostics = analyze(run, summary, config, tasks, workload)
    return {'observability': {'model_lifecycle': any(e['event'] == 'model_load_finished' for e in lifecycle), 'drain': scheduler_log.exists()}, 'run': run.name, 'workload': workload, 'schedule_events': detailed, 'lifecycle': lifecycle, 'models': models, 'tasks': rows,
            'duration': (end - start) / 1000, 'sending_seconds': summary['sending_seconds'],
            'started_at': datetime.fromtimestamp(start / 1000).astimezone().isoformat(timespec='seconds'),
            'summary': summary, 'client': client, 'arguments': manifest['arguments'], 'runtime': manifest.get('runtime', {}), 'config': config,
            'revision': manifest['revision'], 'csv_sha256': manifest['csv_sha256'],
            'batching_policy': scheduler_config['batch_extractor_strategy'],
            'diagnostics': diagnostics, 'analysis_role': analysis_role,
            'resource_warnings': resource_warnings,
            'schedules': schedules,
            'resources': grouped, 'raw_resources': resources}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--compare', type=Path, action='append', default=[])
    args = parser.parse_args()
    data = [build_data(run.resolve()) for run in [args.run, *args.compare]]
    # Escape script terminators in arbitrary model/run names.
    serialized = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    page = (HERE / 'report.html').read_text().replace('__REPORT_DATA__', serialized)
    page = page.replace('__METRICS_JS__', (HERE / 'timeline_metrics.js').read_text())
    output = args.output or args.run / 'report.html'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page)
    print(output.resolve())


if __name__ == '__main__':
    main()
