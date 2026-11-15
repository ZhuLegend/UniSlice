#!/usr/bin/env python3
"""Coverage and measurement audit for the full-parent runtime flow tests."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts/simulation'))
from run import allocation_plan_identity, dump
from long_analysis import log_events


def read(path):
    return json.loads(path.read_text())


def compact(row):
    server,client=row['server_summary'],row['client_metrics']
    return {'id':row['id'],'policy':row['policy'],'output':row['output'],
        'mean_jct_ms':server['mean_jct_ms'],'p95_jct_ms':server['p95_jct_ms'],
        'p99_jct_ms':server['p99_jct_ms'],'failure_percent':client['failure_percent'],
        'slo_400_percent':client['all_offered_slo_percent']['400'],
        'successful_samples_per_second':row['client_throughput_samples_per_second'],
        'sending_and_drain_seconds':row['client_window']['sending_and_drain_seconds'],
        'trial_wall_seconds':row['trial_wall_seconds'],
        'max_send_lag_ms':server['max_send_lag_ms'],
        'worker_batch_mean':row['actual_worker_execution_batch']['mean'],
        'resolved_plan':read(Path(row['output'])/row['plan_identity']['resolved_path'])['plan']}


def audit(folder):
    summary,manifest=read(folder/'summary.json'),read(folder/'manifest.json')
    jobs=[json.loads(line) for line in (folder/'inputs/jobs.jsonl').read_text().splitlines()]
    rows=[json.loads(line) for line in (folder/'results.jsonl').read_text().splitlines()]
    assert summary['finished'] and summary['input_exhausted'] and not summary['error']
    assert not summary['remaining_owned_processes'] and not summary['host']['memory_guard']
    assert summary['completed_trials']==summary['launched_trials']==len(rows)==len(jobs)
    assert len({job['id'] for job in jobs})==len(jobs)
    assert {row['sequence'] for row in rows}==set(range(1,len(jobs)+1))
    expected=read(folder/'inputs/replay.json')
    model_requests=dict(Counter(event[1] for event in expected))
    rates=read(folder/'inputs/model-rates.json')
    for row in rows:
        job=jobs[row['sequence']-1]
        assert row['id']==job['id'] and row['measurement_complete'] and row['all_owned_processes_gone']
        assert row['code_identity']==manifest['code_identity'] and row['replay_sha256']==manifest['replay_sha256']
        server,client=row['server_summary'],row['client_metrics']
        assert server['scheduler']['successful_schedules']==1
        assert server['pending_tasks']==client['pending_requests']==0
        assert client['offered_requests']==len(expected)
        assert client['offered_samples']==sum(event[2] for event in expected)
        assert {model:counts['offered'] for model,counts in server['client_transport']['models'].items()}==model_requests
        run=Path(row['output']); identity=read(run/'manifest.json')
        assert all(identity[key]==value for key,value in manifest['profile_identity'].items())
        assert read(run/'workload.json')['initial_rates']==rates
        if not client['failed_requests']:
            assert row['runner_exit_code']==0 and not row['shutdown']['errors']
            assert row['actual_worker_execution_batch']['samples']==server['completed_samples']==client['offered_samples']
        if job.get('allocation'):
            assert row['plan_identity']['resolved_plan_sha256']==allocation_plan_identity(job['allocation'])
    successful=[row for row in rows if not row['client_metrics']['failed_requests']]
    assert successful
    best=min(successful,key=lambda row:row['server_summary']['mean_jct_ms'])
    greedy=next((row for row in rows if row['id']=='greedy'),None)
    result={'queue':str(folder),'completed':len(rows),'complete_application_failure_trials':sum(bool(row['client_metrics']['failed_requests']) for row in rows),
        'requests':sum(row['client_metrics']['offered_requests'] for row in rows),
        'samples':sum(row['client_metrics']['offered_samples'] for row in rows),
        'concurrency':summary['concurrency_limit'],'wall_seconds':summary['wall_seconds'],
        'trials_per_second':len(rows)/summary['wall_seconds'],
        'slot_occupancy_fraction':summary['slot_occupancy_fraction'],
        'peak_owned_rss_gib':summary['host']['peak_owned_rss_kib']/1024**2,
        'trial_wall_seconds':{'mean':statistics.mean(row['trial_wall_seconds'] for row in rows),'max':max(row['trial_wall_seconds'] for row in rows)},
        'max_send_lag_ms':max(row['server_summary']['max_send_lag_ms'] for row in rows),
        'best':compact(best),'greedy':compact(greedy) if greedy else None,
        'best_mean_jct_reduction_vs_greedy_percent':100*(1-best['server_summary']['mean_jct_ms']/greedy['server_summary']['mean_jct_ms']) if greedy and not greedy['client_metrics']['failed_requests'] else None,
        'scope':'Complete finite-trace observed mean server JCT, request weighted. Failed candidates excluded from winners. No expected-JCT or global dynamic-policy optimum claim.',
        'input_sha256':{name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in ['manifest.json','summary.json','results.jsonl']}}
    if greedy:
        domain={allocation_plan_identity(job['allocation']) for job in jobs if job.get('allocation')}
        result['greedy_in_enumerated_domain']=greedy['plan_identity']['resolved_plan_sha256'] in domain
    dump(folder/'coverage-audit.json',result)
    return result,rows


def worker_diagnostics(rows):
    events=[]
    tasks=[]
    for row in rows:
        run=Path(row['output'])
        for node in read(run/'simulation.json')['nodes']:
            events.extend(event for event in log_events(run/f"{node['name']}.log") if event.get('event')=='inference')
        tasks.extend(read(run/'tasks.json'))
    assert sum(event['batch'] for event in events)==sum(row['client_metrics']['completed_samples'] for row in rows)
    groups={}
    for model in sorted({event['model'] for event in events}):
        selected=[event for event in events if event['model']==model]
        completed=[task for task in tasks if task['model']==model and task['status']=='completed']
        groups[model]={'calls':len(selected),'samples':sum(event['batch'] for event in selected),
            'mean_execution_batch':statistics.mean(event['batch'] for event in selected),
            'mean_worker_elapsed_ms':1000*statistics.mean(event['elapsed_seconds'] for event in selected),
            'mean_profile_ms':1000*statistics.mean(event['profile_seconds'] for event in selected),
            'mean_request_jct_ms':statistics.mean(task['completed_at']-task['received_at'] for task in completed)}
    overshoot=sorted(1000*(event['elapsed_seconds']-event['sampled_seconds']) for event in events)
    return {'models':groups,'wake_over_sampled_ms':{'n':len(overshoot),'mean':statistics.mean(overshoot),
        'p99':overshoot[max(0,math.ceil(.99*len(overshoot))-1)],'max':max(overshoot)},
        'scope':'Worker call weighted timing and per-model request weighted JCT; no per-request latency decomposition.'}

