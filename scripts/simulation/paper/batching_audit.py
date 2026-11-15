"""Strict extraction experiment; finite-stream censoring is audited explicitly."""
from collections import Counter
from pathlib import Path
import csv
import gzip
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import tomllib
import traceback
import uuid

from run import code_identity, percentile
from calibrate_parallel import OwnedMonitor
from long_analysis import timestamp_ms


def read(path): return json.loads(path.read_text())
def text_file(path):
    if path.exists(): return path.read_text()
    with gzip.open(str(path)+'.gz', 'rt') as handle: return handle.read()

def rows(path): return [json.loads(x) for x in text_file(path).splitlines() if x.strip()]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def log_events(path):
    opener = gzip.open if path.name.endswith('.gz') else open
    with opener(path, 'rt') as source:
        for line in source:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if 'timestamp' in row and 'fields' in row:
                yield {'timestamp_ms':timestamp_ms(row['timestamp']), **row['fields']}


def save(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def metrics(data, cap=5000):
    assert data
    good = [x for x in data if x['observed_complete']]
    return {'offered': len(data), 'completed': len(good), 'censored': len(data) - len(good),
        'completion_percent': 100 * len(good) / len(data),
        'slo400_percent': 100 * sum(x['observed_complete'] and x['end_to_end_ms'] <= 400 for x in data) / len(data),
        'successful_server_mean_ms': statistics.mean(x['server_jct_ms'] for x in good) if good else None,
        'successful_server_p99_ms': percentile([x['server_jct_ms'] for x in good], .99) if good else None,
        'all_offered_capped_e2e_mean_ms': statistics.mean(min(x['end_to_end_ms'], cap) if x['observed_complete'] else cap for x in data)}


def fixed_sample_accounting(offered, executed, targets):
    result = {}
    for model, total in offered.items():
        done = executed[model]
        target = targets[model]
        assert 0 <= done <= total and done % target == 0
        remaining, modulus = total-done, total % target
        assert remaining >= modulus
        result[model] = {'offered_samples':total, 'executed_samples':done,
            'remaining_samples':remaining, 'unfillable_modulus_samples':modulus,
            'excess_over_modulus_samples':remaining-modulus,
            'classification':'only_modulus_remaining' if remaining==modulus else 'remaining_beyond_modulus_may_include_inflight_or_backlog'}
    return result


def audit(run, case, mode, fixed, identity, seed):
    manifest = read(run / 'manifest.json')
    assert manifest['code_identity'] == identity
    assert manifest['replay_sha256'] == case['source_input_sha256']['replay.json']
    assert manifest['randomness']['root_seed'] == seed
    assert manifest['randomness']['inputs_sha256'] == sha(run / 'random-inputs.json')
    assert tomllib.loads((run / 'scheduler.toml').read_text())['batch_extractor_strategy'] == case['extractors'][mode]
    shutdown = read(run / 'shutdown.json'); assert not shutdown['errors']
    client = rows(run / 'measurement/client-terminal.jsonl')
    final, summary = read(run / 'measurement/final_summary.json'), read(run / 'summary.json')
    replay = read(run / 'replay.json'); assert len(client) == len(replay) == case['offered_requests']
    assert final['sent_tasks'] == len(replay) and final['planned_requests'] == len(replay)
    assert final['sent_samples'] == case['offered_samples']
    assert summary['scheduler']['successful_schedules'] == 1
    expected = {str(uuid.UUID(int=(20260923 << 64) | i)) for i in range(len(replay))}
    assert {c['request_id'] for c in client} == expected
    arrivals = {}
    for event in log_events(next(run.glob('scheduler/task_tracking.jsonl*'))):
        if event.get('event') == 'task_arrived':
            assert event['task_id'] not in arrivals
            arrivals[event['task_id']] = event
    assert set(arrivals) == expected
    terminal_rows = read(run / 'tasks.json')
    terminal = {t['task_id']: t for t in terminal_rows}
    assert len(terminal) == len(terminal_rows)
    assert set(terminal) <= expected
    assert len(terminal) + summary['pending_tasks'] == len(replay)
    start = final['planned_origin_ms']
    cutoff = start + 1000 * (case['seconds'] + 10)
    clients_by_id = {c['request_id']: c for c in client}
    for t in terminal_rows:
        if t['status'] == 'failed':
            assert clients_by_id[t['task_id']]['failure_kind'] == 'drain_cancelled'
            assert t['completed_at'] >= cutoff - 50, 'server failure before observation ended'
    result = []
    for c in client:
        i = uuid.UUID(c['request_id']).int & ((1 << 64) - 1)
        due, model, batch = replay[i]
        assert (c['model'], c['batch_size']) == (model, batch)
        assert abs(c['planned_due_ms'] - start - 1000 * due) <= 1
        assert (arrivals[c['request_id']]['model_id'], arrivals[c['request_id']]['batch_size']) == (model, batch)
        assert c['failure_kind'] in (None, 'drain_cancelled')
        good = c['success'] and c['completed_at_ms'] <= cutoff
        if good:
            t = terminal[c['request_id']]; assert t['status'] == 'completed'
            duration = t['completed_at'] - t['received_at']
        else:
            assert c['end_to_end_ms'] >= 5000
            duration = None
        window = 'low_before' if due < 30 else 'high' if due < 60 else 'low_after'
        result.append({'request_id': c['request_id'], 'model': model, 'phase': window,
            'observed_complete': good, 'server_jct_ms': duration, 'end_to_end_ms': c['end_to_end_ms']})
    cancelled = sum(c['failure_kind'] == 'drain_cancelled' for c in client)
    assert final['pending_at_drain'] == cancelled
    host = rows(run / 'host_metrics.jsonl')
    assert host and not any('error' in h for h in host)
    gap = max(b['timestamp_ms'] - a['timestamp_ms'] for a, b in zip(host, host[1:]))
    assert gap <= 5000
    assert host[0]['timestamp_ms'] <= start
    assert host[-1]['timestamp_ms'] + 2500 >= max(c['completed_at_ms'] for c in client)
    policies = []
    import re
    for line in text_file(run / 'scheduler.log').splitlines():
        if '"batch_policy"' in line:
            m = re.search(r'worker_id: ([0-9a-f-]+), model: ([A-Za-z0-9_-]+), fixed_target_batch: (\d+)', line)
            assert m, 'unparsed batch-policy record'
            policies.append((m[2], int(m[3])))
    counts = Counter(model for model, _ in policies)
    placement = read(run / 'experiment-allocation-input.json')
    assert counts == Counter({p['model']: p['count'] for p in placement['placements']})
    assert all(batch == (fixed[model] if mode == 'fixed' else 0) for model, batch in policies)
    calls = []; executed = Counter()
    for path in run.glob('scheduler/worker_perf.jsonl*'):
        for e in log_events(path):
            assert 1 <= e['batch_size'] <= 32
            if mode == 'fixed': assert e['batch_size'] == fixed[e['model_name']], 'underfilled fixed call'
            executed[e['model_name']] += e['batch_size']
            calls.append(e)
    offered = Counter()
    for _, model, batch in replay: offered[model] += batch
    assert all(executed[m] <= offered[m] for m in offered)
    residual = fixed_sample_accounting(offered, executed, fixed) if mode == 'fixed' else None
    scope = {'all': metrics(result)}
    if case['seconds'] == 90:
        scope.update({p: metrics([r for r in result if r['phase'] == p]) for p in ['low_before', 'high', 'low_after']})
    return {'fixed_sample_accounting': residual, 'metrics': scope, 'requests': result, 'cancelled_at_observation_end': cancelled,
        'server_pending_at_stop': summary['pending_tasks'], 'host_max_gap_ms': gap,
        'offered_samples_by_model': dict(offered), 'executed_samples_by_model': dict(executed),
        'actual_call_batch_histogram': dict(Counter(e['batch_size'] for e in calls)),
        'raw_runner_error': read(run / 'error.json') if (run / 'error.json').exists() else None}


def expected_censor_exit(code, certificate):
    if code == 0:
        return certificate['cancelled_at_observation_end'] == 0
    message = (certificate['raw_runner_error'] or {}).get('error', '')
    return code == 1 and certificate['cancelled_at_observation_end'] > 0 and message in [
        'measurement stopped with pending requests; partial results were preserved',
        'measurement finished with explicit failed requests; complete records were preserved']

