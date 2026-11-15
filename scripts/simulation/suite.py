#!/usr/bin/env python3
"""Run paper-aligned load multipliers sequentially and build the comparison page."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from run import code_identity, http_request_timeout_seconds

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--multipliers', nargs='+', type=float, default=[1, 4, 8])
    parser.add_argument('--config', type=Path, default=ROOT/'resource/simulation/paper-a100.json')
    parser.add_argument('--long-cycle', action='store_true')
    parser.add_argument('--phase-seconds', type=int)
    parser.add_argument('--schedule-interval', type=int, default=15)
    parser.add_argument('--http2-max-streams', type=int, default=65536)
    parser.add_argument('--max-in-flight', type=int, default=262144)
    parser.add_argument('--drain-seconds', type=int, help='default: HTTP request deadline + 20 seconds')
    parser.add_argument('--seed', type=int, default=20260923)
    parser.add_argument('--input-batch-min', type=int, default=1)
    parser.add_argument('--input-batch-max', type=int, default=32)
    parser.add_argument('--noise-cv', type=float, default=0.1)
    parser.add_argument('--noise-seed', type=int)
    parser.add_argument('--max-host-memory-gib', type=float, default=4)
    parser.add_argument('--skip-build', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--resume', action='store_true', help='reuse fully finished runs with matching experiment settings')
    args = parser.parse_args()
    if args.drain_seconds is None:
        args.drain_seconds = http_request_timeout_seconds(json.loads(args.config.read_text())) + 20
    if args.phase_seconds is None:
        args.phase_seconds = 360 if args.long_cycle else 60
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    started = time.monotonic()
    started_unix = time.time()
    if args.resume:
        previous = json.loads((out/'suite_status.json').read_text())
        manifests = [json.loads(p.read_text()) for p in out.glob('*/manifest.json')]
        started_unix = previous.get('started_unix_seconds', min((m['started_unix_seconds'] for m in manifests), default=started_unix))
    status = {'multipliers': args.multipliers, 'completed': [], 'running': None, 'started_unix_seconds': started_unix}
    runs = []
    failed = False
    for index, multiplier in enumerate(args.multipliers):
        name = f'{multiplier:g}x'
        status['running'] = name
        (out/'suite_status.json').write_text(json.dumps(status, indent=2))
        command = [sys.executable, str(HERE/'run.py'), '--config', str(args.config.resolve()),
                   '--paper-multiplier', str(multiplier), '--phase-seconds', str(args.phase_seconds),
                   '--schedule-interval', str(args.schedule_interval), '--seed', str(args.seed),
                   '--http2-max-streams', str(args.http2_max_streams),
                   '--max-in-flight', str(args.max_in_flight),
                   '--input-batch-min', str(args.input_batch_min), '--input-batch-max', str(args.input_batch_max),
                   '--noise-cv', str(args.noise_cv), '--max-host-memory-gib', str(args.max_host_memory_gib),
                   '--drain-seconds', str(args.drain_seconds), '--output', str(out/name)]
        if args.noise_seed is not None: command += ['--noise-seed', str(args.noise_seed)]
        if args.long_cycle: command.append('--long-cycle')
        if args.skip_build or index > 0: command.append('--skip-build')
        if args.offline: command.append('--offline')
        if (out/name).exists():
            if not args.resume or not (out/name/'shutdown.json').exists() or not (out/name/'summary.json').exists():
                raise RuntimeError(f'{name} exists but is not a fully finished run; preserve and inspect it before resuming')
            manifest = json.loads((out/name/'manifest.json').read_text())
            if not manifest.get('code_identity') or manifest['code_identity'] != code_identity():
                raise RuntimeError(f'{name}: saved sources or executables differ; use a new suite directory')
            saved = manifest['arguments']
            expected = {'paper_multiplier':multiplier,'phase_seconds':args.phase_seconds,'schedule_interval':args.schedule_interval,'seed':args.seed,'long_cycle':args.long_cycle}
            expected.update(http2_max_streams=args.http2_max_streams, max_in_flight=args.max_in_flight)
            expected.update(input_batch_min=args.input_batch_min, input_batch_max=args.input_batch_max,
                            noise_cv=args.noise_cv, noise_seed=args.noise_seed,
                            max_host_memory_gib=args.max_host_memory_gib)
            expected['drain_seconds'] = args.drain_seconds
            if any(saved.get(key, False if key=='long_cycle' else None) != value for key,value in expected.items()):
                raise RuntimeError(f'{name}: saved workload differs from requested resume settings')
            requested = json.loads(args.config.read_text())
            requested['execution_noise'] = {'cv': args.noise_cv,
                'seed': args.noise_seed if args.noise_seed is not None else requested.get('execution_noise', {}).get('seed', args.seed)}
            snapshot = json.loads((out/name/'simulation.json').read_text())
            for field, hash_key in [('latency_csv','csv_sha256'),('batching_config','batching_sha256')]:
                source = args.config.resolve().parent/requested[field]
                if hashlib.sha256(source.read_bytes()).hexdigest() != manifest.get(hash_key):
                    raise RuntimeError(f'{name}: saved {field} contents differ from requested source')
            for key in (snapshot.keys() | requested.keys()) - {'latency_csv','batching_config'}:
                if snapshot.get(key) != requested.get(key):
                    raise RuntimeError(f'{name}: saved {key} differs from requested configuration')
            summary = json.loads((out/name/'summary.json').read_text())
            result_code = int(bool(summary['failed_tasks'] or summary['pending_tasks'] or summary['client_failed_tasks'] or summary.get('shutdown_errors') or summary['sender_exit_code']))
            print(f'Reusing completed {name}: exit {result_code}', flush=True)
        else:
            print(f'Starting {name}', flush=True)
            with (out/f'{name}.log').open('w') as log:
                result_code = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        failed |= result_code != 0
        status['completed'].append({'run': name, 'exit_code': result_code})
        (out/'suite_status.json').write_text(json.dumps(status, indent=2))
        if (out/name/'summary.json').exists():
            runs.append(out/name)
            report = [sys.executable, str(HERE/'report.py'), str(runs[0]), '--output', str(out/'index.html')]
            for run in runs[1:]: report += ['--compare', str(run)]
            rendered = subprocess.run(report, cwd=ROOT)
            status['completed'][-1]['report_exit_code'] = rendered.returncode
            failed |= rendered.returncode != 0
        print(f'Finished {name}: exit {result_code}', flush=True)
    status.update(running=None, elapsed_seconds=time.time()-started_unix, resumed_execution_seconds=time.monotonic()-started)
    (out/'suite_status.json').write_text(json.dumps(status, indent=2))
    return int(failed)


if __name__ == '__main__':
    sys.exit(main())
