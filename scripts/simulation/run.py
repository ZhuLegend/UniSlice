#!/usr/bin/env python3
"""Run the real scheduler/queues against CSV-driven virtual nodes (Python 3.11+)."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import tomllib
import urllib.request
from workload import paper_workload, long_cycle_workload, phase_events, MAX_REPLAY_EVENTS
from monitor import ProcessMonitor
from terminal import read_terminal_tasks
from random_inputs import resolve_random_inputs

ROOT = Path(__file__).resolve().parents[2]

def source_fingerprint(root=ROOT):
    """Hash execution sources, including different edits to the same dirty file."""
    paths = [root/'Cargo.toml', root/'Cargo.lock']
    suffixes = {'.rs', '.py', '.js', '.toml', '.yaml', '.yml', '.sh', '.c', '.h', '.cpp'}
    for directory in ('crates', 'scripts', '.github'):
        paths.extend(p for p in (root/directory).rglob('*') if p.is_file() and p.suffix in suffixes)
    paths.extend(root.glob('Dockerfile*'))
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(str(path.relative_to(root)).encode() + b'\0')
        digest.update(path.read_bytes() if path.exists() else b'<missing>')
        digest.update(b'\0')
    return digest.hexdigest()

def code_identity(root=ROOT, binaries=None):
    binaries_dir = binaries or Path(os.environ.get('CARGO_TARGET_DIR', root/'target')).resolve()/'debug'
    binaries = {}
    for name in ('mock_virtual', 'mock_vgpu', 'scheduler_daemon', 'task_sender'):
        with (binaries_dir/name).open('rb') as binary:
            binaries[name] = hashlib.file_digest(binary, 'sha256').hexdigest()
    return {'source_sha256': source_fingerprint(root), 'binary_sha256': binaries}


def git_metadata(root=ROOT):
    result = {}
    for key, args in [('revision', ['rev-parse', 'HEAD']), ('worktree', ['status', '--short'])]:
        try:
            proc = subprocess.run(['git', *args], cwd=root, capture_output=True, text=True)
            result[key] = proc.stdout.strip() if proc.returncode == 0 else None
        except FileNotFoundError:
            result[key] = None
    return result


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")

def validate_fixed_batches(value, config):
    if not isinstance(value, dict) or set(value) != set(config['models']):
        raise ValueError('fixed batches must cover exactly the configured models')
    if any(type(batch) is not int or not 1 <= batch <= 32 for batch in value.values()):
        raise ValueError('fixed batch targets must be integers in [1,32]')
    return value

def get(url, timeout=3):
    # This runner only controls its own loopback daemon; bypass OS/environment proxies.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=timeout) as response:
        return json.load(response)

def wait_until(check, processes, timeout, label):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for proc in processes:
            if proc.poll() is not None:
                raise RuntimeError(f"{label}: process exited with {proc.returncode}; inspect run logs")
        try:
            if check():
                return
        except (OSError, ValueError):
            pass
        time.sleep(0.2)
    raise TimeoutError(f"{label}: exceeded {timeout}s")

def port_open(port):
    with socket.create_connection(("127.0.0.1", port), timeout=0.5):
        return True

def ensure_ports_free(ports):
    sockets = []
    try:
        for port in ports:
            sock = socket.socket()
            sockets.append(sock)
            # Match the server's restart behavior after TCP TIME_WAIT. listen()
            # still rejects an active listener; SO_REUSEPORT is intentionally absent.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
            sock.listen(1)
    finally:
        for sock in sockets:
            sock.close()

def toml_inline(value):
    """Serialize the scalar/list/table subset used by batching strategies."""
    if isinstance(value, dict):
        return '{ ' + ', '.join(f'{json.dumps(k)} = {toml_inline(v)}' for k, v in value.items()) + ' }'
    if isinstance(value, list):
        return '[' + ', '.join(map(toml_inline, value)) + ']'
    if isinstance(value, (str, bool, int, float)):
        return json.dumps(value, allow_nan=False)
    raise ValueError(f'unsupported batching configuration value: {value!r}')

def scheduler_config(config, rate, config_dir=ROOT/'resource/simulation'):
    rates = rate if isinstance(rate, dict) else {m: rate for m in config["models"]}
    source = config_dir/Path(config['batching_config'])
    batching = tomllib.loads(source.read_text())['batch_extractor_strategy']
    lines = [
        f'batch_extractor_strategy = {toml_inline(batching)}',
        f'batch_size_range = {json.dumps(config["batch_size_range"])}',
        '[init_arrival_rate]',
        *[f'{json.dumps(model)} = {float(rates[model])}' for model in config['models']],
    ]
    templates = {}
    for node in config['nodes']:
        template = node['template_name']
        if not template:
            raise ValueError('template_name must identify a shared scheduling pool')
        definition = {key: node[key] for key in ('kind', 'resource', 'share_targets')}
        if template in templates and templates[template] != definition:
            raise ValueError(f'inconsistent device template {template}')
        templates[template] = definition
    for template, node in templates.items():
        name = json.dumps(template)
        kind = 'MockNvidia' if node['kind'] == 'nvidia' else 'MockHuawei'
        resource = ', '.join(f'{json.dumps(k)} = {v}' for k, v in node['resource'].items())
        lines += [f'[device_combinations.{name}]', f'type = "{kind}"',
                  f'config.share_targets = {json.dumps(node["share_targets"])}',
                  f'config.resource_template = {{ {resource} }}']
    for node in config['nodes']:
        lines += [f'[remote_nodes.{json.dumps(node["name"])}]', 'ip = "127.0.0.1"',
                  f'port = {node["port"]}', f'template_name = {json.dumps(node["template_name"])}']
    return '\n'.join(lines) + '\n'

def prediction_csv_paths(path, config):
    if path is None:
        return {}
    paths = json.loads(path.read_text())
    templates = {node['template_name'] for node in config['nodes']}
    if not isinstance(paths, dict) or any(template not in templates or not isinstance(csv, str)
                                          or not csv for template, csv in paths.items()):
        raise ValueError('prediction CSVs must map configured template names to file paths')
    paths = {template: (path.resolve().parent/csv).resolve() for template, csv in paths.items()}
    for csv in paths.values():
        if not csv.is_file():
            raise ValueError(f'prediction CSV not found: {csv}')
    return paths

def sender_config(config, url, log_dir, rate, rate_max, batch, cycle):
    low, high = (batch, batch) if isinstance(batch, int) else batch
    lines = [f'[sender]\ndaemon_url = {json.dumps(url)}\nlog_dir = {json.dumps(str(log_dir))}\nstats_interval = 1\ncycle_duration = {cycle}',
             '[metrics]\nrate_window_size = 60\nlatency_window_size = 100',
             '[rate_change]\nstrategy = "sudden"']
    for model in config['models']:
        minimum = rate[model] if isinstance(rate,dict) else rate
        maximum = rate_max[model] if isinstance(rate_max,dict) else rate_max
        lines += [f'[[models]]\nname = {json.dumps(model)}\nrate_range = [{float(minimum)}, {float(maximum)}]\nbatch_size_range = [{low}, {high}]']
    return '\n'.join(lines) + '\n'

def percentile(values, quantile):
    return sorted(values)[min(len(values) - 1, math.ceil(quantile * len(values)) - 1)] if values else None

def http_request_timeout_seconds(config):
    # Let the application return its deadline result before cancelling the
    # HTTP/2 stream. Keep client cancellation later than application expiry.
    return math.ceil(config.get('inference_timeout_ms', 300000) / 1000) + 10


def validate_replay(events, models, seconds, batch_range):
    """Accept the task_sender [offset_seconds, model, input_batch] schema."""
    if not isinstance(events, list) or not events or len(events) > MAX_REPLAY_EVENTS:
        raise ValueError('replay must contain 1 to 1,000,000 events')
    last = -1.0
    for row in events:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError('replay rows must be [offset_seconds, model, batch]')
        offset, model, batch = row
        if (type(offset) not in (int, float) or not math.isfinite(offset)
                or not 0 <= offset < seconds or offset < last):
            raise ValueError('replay offsets must be sorted, finite, and within the sending window')
        if not isinstance(model, str) or model not in models:
            raise ValueError('replay model is not configured')
        if type(batch) is not int or not batch_range[0] <= batch <= batch_range[1]:
            raise ValueError('replay input batch is outside the configured range')
        last = offset
    return events


def model_rates(models, rate, path=None):
    rates = json.loads(path.read_text()) if path else rate
    rates = rates if isinstance(rates,dict) else {model:rates for model in models}
    if not isinstance(rates,dict) or set(rates) != set(models) or any(
        isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value <= 0
        for value in rates.values()):
        raise ValueError('model rates must contain every configured model with a positive finite samples/s value')
    return {model:float(rates[model]) for model in sorted(models)}


def single_phase_workload(models, rate, seconds, seed, batch_range, events=None):
    rates = model_rates(models,rate)
    if events is None:
        events = sorted(phase_events(models, rates, 0, seconds, seed, batch_range, MAX_REPLAY_EVENTS),
                        key=lambda row: (row[0], row[1]))
    validate_replay(events, models, seconds, batch_range)
    return {'name': 'single-phase-replay', 'seed': seed, 'duration_seconds': seconds,
            'initial_rates': rates, 'slo_ms': [250, 400], 'batch_size_range': list(batch_range),
            'rate_unit': 'samples/s', 'phases': [{'start_seconds': 0, 'name': 'constant', 'rates': rates}],
            'provenance': 'Static input with precise replay offsets; configured rates initialize the scheduler. '
                          'External replay offsets are authoritative; rates are not inferred from realized arrivals.'}, events


def validate_allocation_plan(plan, config):
    if not isinstance(plan, dict) or type(plan.get('schema_version')) is not int or plan['schema_version'] != 1:
        raise ValueError('allocation plan requires schema_version 1')
    placements = plan.get('placements')
    if not isinstance(placements, list) or not placements:
        raise ValueError('allocation plan requires nonempty placements')
    templates = {node['template_name']: node for node in config['nodes']}
    seen = set()
    for row in placements:
        if not isinstance(row, dict) or set(row) != {'template_name', 'model', 'slice', 'count'}:
            raise ValueError('invalid allocation placement fields')
        template, model, profile, count = (row[key] for key in ['template_name', 'model', 'slice', 'count'])
        if (not isinstance(template, str) or template not in templates or model not in config['models']
                or profile not in templates[template]['share_targets'] or type(count) is not int or count <= 0):
            raise ValueError('allocation placement requires configured template/model/slice and a positive integer count')
        key = (template, model, profile)
        if key in seen:
            raise ValueError('duplicate allocation placement')
        seen.add(key)
    # Parent-budget feasibility remains the daemon's authoritative validation.
    return plan


def allocation_plan_identity(plan):
    """Semantic identity is distinct from the SHA of an exact input snapshot."""
    normalized = {'schema_version': plan['schema_version'], 'placements': sorted(plan['placements'],
                  key=lambda row: (row['template_name'], row['model'], row['slice']))}
    return hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def resolved_allocation(path, mode, config, expected_plan=None):
    record = json.loads(path.read_text())
    if record.get('mode') != mode or record.get('applied') is not True:
        raise ValueError('experimental allocation was not applied in the requested mode')
    validate_allocation_plan(record.get('plan'), config)
    if expected_plan is not None and allocation_plan_identity(record['plan']) != allocation_plan_identity(expected_plan):
        raise ValueError('resolved allocation differs from the fixed input plan')
    return record


def wait_measurement_barrier(path, out, processes, timeout, ready, monitor=None):
    """Publish readiness, then wait for an atomically published shared release file."""
    if path.exists():
        raise ValueError('measurement barrier already released; use a new barrier path')
    dump(out/'calibration-ready.json', ready)
    def released():
        if monitor and monitor.limit_exceeded.is_set():
            raise RuntimeError('host memory guard triggered before barrier release')
        return path.exists()
    wait_until(released, processes, timeout, 'measurement barrier')
    dump(out/'calibration-release.json', {'barrier_path': str(path), 'observed_at_ms': round(time.time()*1000),
                                         'release': json.loads(path.read_text())})

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    workload_group = parser.add_mutually_exclusive_group()
    workload_group.add_argument('--paper-multiplier', type=float, help='paper SLO/workload preset, e.g. 1, 4, 8, 16, 32')
    workload_group.add_argument('--replay', type=Path, help='precise task_sender replay JSON; single phase, --seconds sending window')
    workload_group.add_argument('--single-phase', action='store_true', help='seeded constant-rate replay, input batch 1–32 by default')
    parser.add_argument('--long-cycle', action='store_true', help='six phases / five switches, with the configured load multiplier')
    parser.add_argument('--phase-seconds', type=int)
    parser.add_argument('--seed', type=int, default=20260923)
    parser.add_argument('--config', type=Path, default=ROOT/'resource/simulation/heterogeneous.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--skip-build', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--port', type=int, default=5300)
    parser.add_argument('--seconds', type=int, default=30)
    parser.add_argument('--warmup-seconds', type=int, default=5)
    parser.add_argument('--rate', type=float, default=100, help='samples/s per model')
    parser.add_argument('--model-rates', type=Path, help='JSON model-to-samples/s map for single-phase/replay input and initial scheduler demand')
    parser.add_argument('--rate-max', type=float, help='optional changing sample rate upper bound')
    parser.add_argument('--batch', type=int, default=3, help='fixed input batch; internal batches remain adaptive')
    parser.add_argument('--input-batch-min', type=int, default=1, help='minimum request batch for workload presets')
    parser.add_argument('--input-batch-max', type=int, default=32, help='maximum request batch for workload presets')
    parser.add_argument('--noise-cv', type=float, help='execution-time CV, default config value or 0.1; 0 disables variation')
    parser.add_argument('--noise-seed', type=int, help='default config value or workload seed')
    parser.add_argument('--random-seed', type=int, help='root replicate seed for explicit noise, routing and batching inputs; defaults to noise seed')
    parser.add_argument('--random-worker-slots', type=int, default=64, help='explicit per-parent/profile lifetime worker seed budget')
    parser.add_argument('--random-model-loads', type=int, default=16, help='explicit seed budget for genuine loads per model/worker')
    parser.add_argument('--prediction-csvs', type=Path, help='JSON map of template_name to predictor CSV; worker execution profiles remain unchanged')
    parser.add_argument('--routing-snapshot-seconds', type=float, help='passive routing snapshot interval, including sending and drain')
    parser.add_argument('--client-terminal-log', action='store_true', help='persist each client outcome, including failures and planned arrival')
    parser.add_argument('--max-host-memory-gib', type=float, default=4, help='aggregate simulation RSS guard; safely stop above this limit')
    parser.add_argument('--cycle-seconds', type=int, default=15)
    parser.add_argument('--schedule-interval', type=int, default=60)
    parser.add_argument('--drain-seconds', type=int, help='default: HTTP request deadline + 20 seconds')
    parser.add_argument('--ready-timeout', type=int, default=900)
    allocation_group = parser.add_mutually_exclusive_group()
    allocation_group.add_argument('--experiment-fixed-allocation', type=Path, help='experimental fixed placement JSON (daemon validates parent budgets)')
    allocation_group.add_argument('--experiment-freeze-greedy', action='store_true', help='hold the first ready greedy allocation for this experiment')
    parser.add_argument('--experiment-fixed-batches', type=Path, help='mock-only model-to-fixed-batch JSON; requires a fixed allocation and SplitAndSatisfy extraction')
    parser.add_argument('--measurement-barrier', type=Path, help='wait after readiness/warmup until this shared JSON release file exists')
    parser.add_argument('--barrier-timeout', type=float, default=900, help='maximum seconds waiting for the measurement barrier')
    parser.add_argument('--http-protocol', choices=['http1', 'http2'], default='http2')
    parser.add_argument('--http-connections', type=int, default=4)
    parser.add_argument('--http2-max-streams', type=int, default=65536)
    parser.add_argument('--max-in-flight', type=int, default=262144)
    parser.add_argument('--create-delay-ms', type=int, help='explicit development override; saved in snapshot')
    parser.add_argument('--destroy-delay-ms', type=int, help='explicit development override; saved in snapshot')
    args = parser.parse_args()
    if min(args.http_connections, args.http2_max_streams, args.max_in_flight) <= 0:
        parser.error('transport capacities must be positive')
    if not math.isfinite(args.barrier_timeout) or args.barrier_timeout <= 0:
        parser.error('barrier timeout must be finite and positive')
    if args.long_cycle and args.paper_multiplier is None:
        parser.error('--long-cycle requires --paper-multiplier')
    if args.phase_seconds is None:
        args.phase_seconds = 360 if args.long_cycle else 60
    def terminate_run(signum, _frame):
        raise KeyboardInterrupt(f'run interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, terminate_run)
    if args.seconds <= 0 or args.warmup_seconds < 0 or min(args.cycle_seconds, args.schedule_interval) <= 0 or (args.drain_seconds is not None and args.drain_seconds <= 0) or args.rate <= 0 or not math.isfinite(args.rate):
        parser.error('durations and rate must be positive (warmup may be zero)')
    rate_max = args.rate if args.rate_max is None else args.rate_max
    if not math.isfinite(rate_max) or rate_max < args.rate:
        parser.error('--rate-max must be finite and >= --rate')
    config = json.loads(args.config.read_text())
    if args.model_rates and (not (args.single_phase or args.replay) or args.rate_max is not None):
        parser.error('--model-rates requires --single-phase/--replay and cannot be combined with --rate-max')
    rates = model_rates(config['models'],args.rate,args.model_rates)
    plan_bytes = args.experiment_fixed_allocation.read_bytes() if args.experiment_fixed_allocation else None
    fixed_plan = validate_allocation_plan(json.loads(plan_bytes), config) if plan_bytes is not None else None
    fixed_batches = validate_fixed_batches(json.loads(args.experiment_fixed_batches.read_text()), config) if args.experiment_fixed_batches else None
    if fixed_batches is not None:
        strategy = tomllib.loads((args.config.resolve().parent/Path(config['batching_config'])).read_text())['batch_extractor_strategy']
        if fixed_plan is None or strategy != {'type': 'SplitAndSatisfy'}:
            parser.error('fixed batches require --experiment-fixed-allocation and SplitAndSatisfy extraction')
    prediction_paths = prediction_csv_paths(args.prediction_csvs, config)
    if args.routing_snapshot_seconds is not None and (
        not math.isfinite(args.routing_snapshot_seconds) or args.routing_snapshot_seconds <= 0
    ):
        parser.error('routing snapshot interval must be finite and positive')
    noise = config.get('execution_noise', {})
    cv = args.noise_cv if args.noise_cv is not None else noise.get('cv', 0.1)
    noise_seed = args.noise_seed if args.noise_seed is not None else noise.get('seed', args.seed)
    if not math.isfinite(cv) or not 0 <= cv <= 1 or not 0 <= noise_seed < 2**64:
        parser.error('noise CV must be in [0,1] and noise seed must fit u64')
    if not math.isfinite(args.max_host_memory_gib) or args.max_host_memory_gib <= 0:
        parser.error('host memory guard must be finite and positive')
    config['execution_noise'] = {'cv': cv, 'seed': noise_seed}
    random_seed = args.random_seed if args.random_seed is not None else noise_seed
    random_inputs = resolve_random_inputs(config, random_seed, args.random_worker_slots, args.random_model_loads)
    if args.drain_seconds is None:
        args.drain_seconds = http_request_timeout_seconds(config) + 20
    workload, events = (None, None)
    initial_rates = args.rate
    if args.paper_multiplier is not None:
        generator = long_cycle_workload if args.long_cycle else paper_workload
        if not 1 <= args.input_batch_min <= args.input_batch_max <= 32:
            parser.error('input batch range must be within [1,32]')
        args.batch = (args.input_batch_min, args.input_batch_max)
        workload, events = generator(config['models'], args.paper_multiplier, args.phase_seconds, args.seed, args.batch)
        args.seconds = workload['duration_seconds']
        args.warmup_seconds = 0
        args.cycle_seconds = args.phase_seconds
        initial_rates = workload['initial_rates']
    elif args.single_phase or args.replay:
        if not 1 <= args.input_batch_min <= args.input_batch_max <= 32:
            parser.error('input batch range must be within [1,32]')
        args.batch = (args.input_batch_min, args.input_batch_max)
        replay_bytes = args.replay.read_bytes() if args.replay else None
        replay_events = json.loads(replay_bytes) if replay_bytes is not None else None
        workload, events = single_phase_workload(config['models'], rates, args.seconds, args.seed, args.batch, replay_events)
        args.warmup_seconds = 0
        initial_rates = workload['initial_rates']
    config.setdefault('model_lifecycle', {'load_ms': 1000, 'unload_ms': 100, 'overrides': []})
    config['latency_csv'] = str((args.config.resolve().parent/config['latency_csv']).resolve())
    for name in ['create_delay_ms', 'destroy_delay_ms']:
        override = getattr(args, name)
        if override is not None:
            if override < 0: parser.error('lifecycle delays cannot be negative')
            config[name] = override
    if args.long_cycle and args.phase_seconds * 1000 <= 10 * (config['create_delay_ms'] + config['destroy_delay_ms']):
        parser.error('long-cycle phases must exceed ten times the combined slice creation + destruction delay')
    input_range = (args.batch, args.batch) if isinstance(args.batch, int) else args.batch
    if not config['batch_size_range'][0] <= input_range[0] <= input_range[1] <= config['batch_size_range'][1]:
        parser.error('--batch outside configured batch range')
    out = (args.output or ROOT/'simulation-runs'/time.strftime('%Y%m%d-%H%M%S')).resolve()
    # Resolve and validate before launching anything, then preserve the exact source.
    scheduler_text = scheduler_config(config, initial_rates, args.config.resolve().parent)
    batching_source = (args.config.resolve().parent/config['batching_config']).resolve()
    out.mkdir(parents=True, exist_ok=False)
    dump(out/'random-inputs.json', random_inputs)
    (out/'batching-source.toml').write_bytes(batching_source.read_bytes())
    config['batching_config'] = str(out/'batching-source.toml')
    dump(out/'simulation.json', config)
    (out/'scheduler.toml').write_text(scheduler_text)
    if workload:
        dump(out/'workload.json', workload)
        if args.replay: (out/'replay.json').write_bytes(replay_bytes)
        else: dump(out/'replay.json', events)
    allocation_mode = 'fixed' if fixed_plan is not None else 'frozen_greedy' if args.experiment_freeze_greedy else None
    allocation = {'mode': allocation_mode}
    if fixed_plan is not None:
        (out/'experiment-allocation-input.json').write_bytes(plan_bytes)
        allocation.update(input_path=str(args.experiment_fixed_allocation.resolve()),
                          snapshot_path='experiment-allocation-input.json',
                          input_sha256=hashlib.sha256(plan_bytes).hexdigest(),
                          plan_sha256=allocation_plan_identity(fixed_plan))
    source_before_build = source_fingerprint()
    dump(out/'manifest.json', {
        'arguments': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        'csv_sha256': hashlib.sha256(Path(config['latency_csv']).read_bytes()).hexdigest(),
        'batching_source': str(batching_source),
        'batching_sha256': hashlib.sha256((out/'batching-source.toml').read_bytes()).hexdigest(),
        **git_metadata(),
        'started_unix_seconds': time.time(), 'execution_noise':config['execution_noise'],
        'randomness': {'schema_version': 1, 'root_seed': random_seed,
                       'inputs_sha256': hashlib.sha256((out/'random-inputs.json').read_bytes()).hexdigest(),
                       'worker_slots': args.random_worker_slots, 'model_loads': args.random_model_loads,
                       'seed_resolution': 'external SHA-256 labeled final seeds; UUIDs excluded',
                       'noise_seed_precedence': '--random-seed controls final noise streams; --noise-seed supplies the root only when --random-seed is absent',
                       'route_draw': 'SplitMix64 keyed by zero-based original replay request ordinal',
                       'controller_draw': 'SplitMix64 per logical extraction/controller sequence',
                       'noise_draw': 'StdRng lognormal draws; final per-worker/model/generation seeds injected'},
        'prediction_csvs': {template: {'source': str(path),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for template, path in prediction_paths.items()},
        'model_rates_sha256': hashlib.sha256(args.model_rates.read_bytes()).hexdigest() if args.model_rates else None,
        'experiment_allocation': allocation,
        'replay_sha256': hashlib.sha256((out/'replay.json').read_bytes()).hexdigest() if workload else None,
        'runtime': {'worker_transport':'independent_process_rpc', 'tokio_worker_threads':2,
                    'client_request_timeout_seconds':http_request_timeout_seconds(config),
                    'http_protocol': args.http_protocol, 'data_connections': args.http_connections,
                    'max_streams_per_connection': args.http2_max_streams, 'sender_max_in_flight': args.max_in_flight},
    })
    print(f'Run output: {out}', flush=True)
    if not args.skip_build:
        common = ['cargo','build','--locked'] + (['--offline'] if args.offline else [])
        subprocess.run(common+['-p','compute_node','--features','mock,vgpu,bin_help','--bin','mock_virtual','--bin','mock_vgpu'],cwd=ROOT,check=True)
        subprocess.run(common+['-p','scheduler','--features','scheduler_daemon,task_sender,mock,huawei,nvidia','--bin','scheduler_daemon','--bin','task_sender'],cwd=ROOT,check=True)
    binaries = Path(os.environ.get('CARGO_TARGET_DIR', ROOT/'target')).resolve()/'debug'
    manifest = json.loads((out/'manifest.json').read_text())
    manifest['code_identity'] = code_identity(binaries=binaries)
    if manifest['code_identity']['source_sha256'] != source_before_build:
        raise RuntimeError('sources changed during build; use a new run directory after finishing edits')
    dump(out/'manifest.json', manifest)
    processes, files, groups = [], [], []
    if prediction_paths:
        (out/'predictions').mkdir()
        snapshots = {}
        for template, path in prediction_paths.items():
            snapshot = out/'predictions'/f'{len(snapshots)}.csv'
            snapshot.write_bytes(path.read_bytes())
            snapshots[template] = str(snapshot.relative_to(out))
        dump(out/'prediction-csvs.json', snapshots)
    if args.model_rates:
        (out/'model-rates.json').write_bytes(args.model_rates.read_bytes())
    node_processes = []
    monitor = ProcessMonitor(out/'host_metrics.jsonl', processes, interval=2,
                             max_rss_kib=args.max_host_memory_gib * 1024**2)
    env = dict(os.environ, RUST_LOG='info,allocation=info,simulation=info', TOKIO_WORKER_THREADS='2')
    env.pop('HSGS_EXPERIMENT_FIXED_BATCHES', None)
    if fixed_batches is not None:
        env['HSGS_EXPERIMENT_FIXED_BATCHES'] = json.dumps(fixed_batches, sort_keys=True)
        dump(out/'experiment-fixed-batches.json', fixed_batches)
        manifest['fixed_batch_experiment'] = {'targets': fixed_batches, 'extraction': 'SplitAndSatisfy'}
        dump(out/'manifest.json', manifest)
    env['HSGS_EXPERIMENT_RANDOM_INPUTS'] = str(out/'random-inputs.json')
    # Bound host thread count independently of the virtual GPU count.
    run_started = time.monotonic()
    def launch(name, command):
        log = (out/f'{name}.log').open('w')
        files.append(log)
        proc = subprocess.Popen([str(v) for v in command],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(proc); groups.append(proc.pid)
        return proc
    def send_phase(name, seconds):
        path = out/f'{name}.toml'
        path.write_text(sender_config(config,url,out/name,rates if args.model_rates else args.rate,
                                     rates if args.model_rates else rate_max,args.batch,args.cycle_seconds))
        replay_args = ['--replay', out/'replay.json'] if workload and name == 'measurement' else []
        if replay_args:
            replay_args += ['--replay-request-id-seed', args.seed]
        terminal_args = ['--terminal-log', out/name/'client-terminal.jsonl'] if args.client_terminal_log else []
        proc = launch(name,[binaries/'task_sender','--config',path,'--send-batch-size-only','--duration-secs',seconds,'--drain-timeout-secs',args.drain_seconds,'--request-timeout-secs',http_request_timeout_seconds(config),
            '--http-protocol',args.http_protocol,'--http-connections',args.http_connections,'--max-in-flight',args.max_in_flight]+replay_args+terminal_args)
        deadline = time.monotonic() + seconds + args.drain_seconds + 30
        routing_log = (out/'routing_weights.jsonl').open('w') if args.routing_snapshot_seconds and name == 'measurement' else None
        next_snapshot = time.monotonic()
        try:
            while proc.poll() is None:
                if routing_log and time.monotonic() >= next_snapshot:
                    try:
                        snapshot = get(url+'/routing/weights', timeout=2)
                    except (OSError, ValueError) as exc:
                        snapshot = {'timestamp_ms': round(time.time()*1000), 'error': str(exc)}
                    routing_log.write(json.dumps(snapshot) + '\n')
                    routing_log.flush()
                    next_snapshot = time.monotonic() + args.routing_snapshot_seconds
                if monitor.limit_exceeded.is_set():
                    raise RuntimeError('host memory guard triggered; run is host-limited, not a GPU capacity result')
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(proc.args, seconds+args.drain_seconds+30)
                try: proc.wait(timeout=1)
                except subprocess.TimeoutExpired: pass
            if monitor.limit_exceeded.is_set():
                raise RuntimeError('host memory guard triggered; run is host-limited, not a GPU capacity result')
            status = proc.returncode
        except subprocess.TimeoutExpired:
            snapshot = out/name/'progress.json'
            dump(out/'sender-timeout.json', {'phase': name, 'last_progress': json.loads(snapshot.read_text()) if snapshot.exists() else None})
            raise
        finally:
            if routing_log: routing_log.close()
        return status
    try:
        ensure_ports_free([args.port]+[n['port'] for n in config['nodes']])
        if monitor: monitor.start()
        for node in config['nodes']:
            node_processes.append(launch(node['name'],[binaries/'mock_virtual','--simulation-config',out/'simulation.json','--node',node['name'],'--bin-worker-path',binaries/'mock_vgpu']))
        wait_until(lambda: all(port_open(n['port']) for n in config['nodes']),processes,30,'virtual node startup')
        url = f'http://127.0.0.1:{args.port}'
        allocation_args = ['--experiment-fixed-allocation', out/'experiment-allocation-input.json'] if fixed_plan is not None else ['--experiment-freeze-greedy'] if allocation_mode else []
        prediction_args = ['--prediction-csvs', out/'prediction-csvs.json'] if prediction_paths else []
        scheduler_proc = launch('scheduler',[binaries/'scheduler_daemon','--host','127.0.0.1','--port',args.port,'--config',out/'scheduler.toml','--simulation-config',out/'simulation.json','--log-dir',out/'scheduler','--interval',args.schedule_interval,'--http2-max-streams',args.http2_max_streams]+allocation_args+prediction_args)
        resolved_path = out/'scheduler/experiment-allocation-resolved.json'
        wait_until(lambda: get(url+'/scheduler/stats').get('successful_schedules',0) >= 1 and
                   (not allocation_mode or resolved_path.exists()),processes,args.ready_timeout,'initial allocation')
        if allocation_mode:
            record = resolved_allocation(resolved_path, allocation_mode, config, fixed_plan)
            allocation.update(resolved_path='scheduler/experiment-allocation-resolved.json',
                              resolved_sha256=hashlib.sha256(resolved_path.read_bytes()).hexdigest(),
                              resolved_plan_sha256=allocation_plan_identity(record['plan']), applied_at_ms=record.get('applied_at_ms'))
            manifest['experiment_allocation'] = allocation
            dump(out/'manifest.json', manifest)
        ready_ms = round(time.time()*1000)
        startup_seconds = time.monotonic()-run_started
        dump(out/'startup.json', {'ready_at_ms': ready_ms, 'startup_seconds': startup_seconds, 'initial_rates': initial_rates})
        print(f'All initial slices/models ready after {startup_seconds:.2f}s.',flush=True)
        if args.warmup_seconds:
            if send_phase('warmup',args.warmup_seconds): raise RuntimeError('warmup did not drain successfully')
        if args.measurement_barrier:
            wait_measurement_barrier(args.measurement_barrier.resolve(), out, processes, args.barrier_timeout,
                                     {'runner_pid': os.getpid(), 'ready_at_ms': round(time.time()*1000),
                                      'url': url, 'experiment_allocation': allocation,
                                      'replay_sha256': manifest['replay_sha256'], 'code_identity': manifest['code_identity']}, monitor)
        baseline = get(url+'/stats')['total_tasks']
        started_ms = round(time.time() * 1000)
        start = time.monotonic()
        status = send_phase('measurement',args.seconds)
        client = json.loads((out/'measurement/final_summary.json').read_text())
        started_ms = client.get('started_at_ms', started_ms)
        elapsed = time.monotonic()-start
        finished_ms = round(time.time()*1000)
        totals = get(url+'/stats')
        scheduler_stats = get(url+'/scheduler/stats')
        # The daemon flushes its terminal CSV on graceful shutdown. Read it only
        # after exit, so neither the in-memory history cap nor buffering truncates it.
        scheduler_proc.terminate()
        scheduler_proc.wait(timeout=config.get('control_timeout_ms', 300000)/1000 + 30)
        if scheduler_proc.returncode != 0:
            raise RuntimeError('scheduler did not flush normally')
        all_tasks = read_terminal_tasks(out/'scheduler/task_terminal_log.csv')
        if len(all_tasks) != totals['completed_tasks'] + totals['failed_tasks']:
            raise RuntimeError('server terminal CSV and lifetime counters disagree')
        tasks = all_tasks[baseline:]
        completed = [t for t in tasks if t['status']=='completed']
        latencies = [t['completed_at']-t['received_at'] for t in completed]
        dump(out/'tasks.json',tasks)
        summary = {'measurement_started_at_ms':started_ms,
            'startup_seconds':startup_seconds, 'ready_at_ms':ready_ms,
            'total_wall_seconds':time.monotonic()-run_started,
            'client_sent_tasks':client['sent_tasks'],
            'client_failed_tasks':client['failed_tasks'],
            'client_transport':client.get('transport'),
            'slo_scope':client.get('slo_scope'),
            'max_send_lag_ms':client.get('max_send_lag_ms'),
            'measurement_finished_at_ms':finished_ms,
            'sending_seconds':args.seconds,'sending_and_drain_seconds':elapsed,
            'received_tasks':len(tasks),'completed_tasks':len(completed),
            'failed_tasks':sum(t['status']=='failed' for t in tasks),
            'pending_tasks':totals['total_tasks']-len(all_tasks),
            'terminal_source':'server_csv_after_flush', 'server_history_baseline':baseline,
            'offered_samples':client['sent_samples'],
            'offered_samples_per_second':client['sent_samples']/args.seconds,
            'received_samples':sum(t['batch_size'] for t in tasks),
            'completed_samples':sum(t['batch_size'] for t in completed),
            'completed_samples_per_wall_second':sum(t['batch_size'] for t in completed)/elapsed,
            'mean_jct_ms':sum(latencies)/len(latencies) if latencies else None,
            'p50_jct_ms':percentile(latencies,.5),'p95_jct_ms':percentile(latencies,.95),'p99_jct_ms':percentile(latencies,.99),
            'sender_exit_code':status,'scheduler':scheduler_stats}
        summary['all_offered_received'] = client['sent_tasks'] == len(tasks)
        summary['all_offered_samples_received'] = client['sent_samples'] == summary['received_samples']
        summary['all_requests_terminal'] = not (summary['pending_tasks'] or client['pending_tasks'])
        if workload:
            summary['slo_offered'] = {str(threshold): {model: {
                'hits':client['transport']['models'].get(model, {}).get(f'slo_{threshold}_ms_hits',0),
                'offered':client['transport']['models'].get(model, {}).get('offered',0)} for model in config['models']} for threshold in workload['slo_ms']}
            summary['slo'] = {str(threshold): {model: {
                'hits':sum(t['status']=='completed' and t['completed_at']-t['received_at'] <= threshold for t in tasks if t['model']==model),
                'received':sum(t['model']==model for t in tasks)} for model in config['models']} for threshold in workload['slo_ms']}
        dump(out/'summary.json',summary)
        print(json.dumps(summary,indent=2),flush=True)
        if status or summary['pending_tasks'] or summary['failed_tasks'] or client['failed_tasks'] or client['sent_tasks'] != len(tasks):
            if not summary['all_requests_terminal']:
                raise RuntimeError('measurement stopped with pending requests; partial results were preserved')
            if not summary['all_offered_received']:
                raise RuntimeError('offered and received request counts differ; complete available records were preserved')
            if summary['failed_tasks'] or client['failed_tasks']:
                raise RuntimeError('measurement finished with explicit failed requests; complete records were preserved')
            raise RuntimeError('sender exited with an error; local and remote records were preserved')
        if not completed: raise RuntimeError('no successful inference requests; inspect node and scheduler logs')
    except BaseException as exc:
        dump(out/'error.json',{'error':str(exc)})
        raise
    finally:
        # Ask node parents to stop first: signalling their process groups would
        # kill workers before model unload/slice destruction can finish.
        shutdown_started = time.monotonic()
        for proc in reversed(processes):
            if proc.poll() is None:
                proc.terminate()
        deadline = time.monotonic() + config.get('control_timeout_ms', 300000) / 1000 + 30
        shutdown_errors = []
        for proc in processes:
            try:
                proc.wait(timeout=max(0.1, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                shutdown_errors.append(f'process {proc.pid} did not finish cleanup')
                try: os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                proc.wait()
        for proc in node_processes:
            if proc.returncode != 0:
                shutdown_errors.append(f'node process {proc.pid} exited with {proc.returncode}')
        # Reap any surviving children after parents have completed their cleanup.
        for pgid in groups:
            try: os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError: pass
        shutdown = {'shutdown_seconds': time.monotonic()-shutdown_started, 'errors': shutdown_errors}
        dump(out/'shutdown.json', shutdown)
        if (out/'summary.json').exists():
            summary['shutdown_seconds'] = shutdown['shutdown_seconds']
            summary['total_wall_seconds'] = time.monotonic()-run_started
            summary['shutdown_errors'] = shutdown_errors
            dump(out/'summary.json', summary)
        if shutdown_errors:
            dump(out/'error.json', {'error': '; '.join(shutdown_errors)})

        for log in files: log.close()
        if monitor and monitor.thread.ident is not None: monitor.stop()
        if shutdown_errors: raise RuntimeError('; '.join(shutdown_errors))

if __name__ == '__main__':
    main()
