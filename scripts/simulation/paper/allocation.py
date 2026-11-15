from collections import defaultdict
from pathlib import Path
import copy,json,math,signal,statistics,time
from run import allocation_plan_identity
from .archive import sha
from .domain import write_jobs,SLICE
from .suite import Suite,read,save
SEARCH_SEEDS=tuple(range(2026100501,2026100506))
VALIDATION_SEEDS=tuple(range(2026100511,2026100521))

def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def search_jobs(candidates, seed, repeat):
    return [{'id': f"{row['id']}-r{repeat}", 'allocation': row['allocation'],
             'runtime_seed': seed} for row in candidates]


def validation_jobs(greedy, selected, seeds=VALIDATION_SEEDS):
    # A shared plan has one measurement, not two labels with independent noise.
    plans = {'greedy': greedy}
    if allocation_plan_identity(greedy) != allocation_plan_identity(selected):
        plans['reference'] = selected
    return [{'id': f'{label}-v{repeat}', 'allocation': plan, 'runtime_seed': seed}
            for repeat, seed in enumerate(seeds, 1) for label, plan in plans.items()]


def measures(rows):
    jct = [row['server_summary']['mean_jct_ms'] for row in rows]
    valid_jct = [value for value in jct if value is not None]
    assert all(math.isfinite(value) and value >= 0 for value in valid_jct), 'invalid JCT observation'
    client = [row['client_metrics'] for row in rows]
    assert all(value is not None or item['failed_requests'] for value, item in zip(jct, client)), 'successful run lacks JCT'
    tails = {name: [row['server_summary'][name] for row in rows if row['server_summary'][name] is not None]
             for name in ['p95_jct_ms', 'p99_jct_ms']}
    return {'repeats': len(rows), 'per_repeat_jct_ms': jct,
            'mean_jct_ms': statistics.mean(valid_jct) if valid_jct else None,
            'sample_sd_jct_ms': statistics.stdev(valid_jct) if len(valid_jct) > 1 else 0,
            'min_jct_ms': min(valid_jct) if valid_jct else None, 'max_jct_ms': max(valid_jct) if valid_jct else None,
            'mean_per_run_p95_ms': statistics.mean(tails['p95_jct_ms']) if tails['p95_jct_ms'] else None,
            'mean_per_run_p99_ms': statistics.mean(tails['p99_jct_ms']) if tails['p99_jct_ms'] else None,
            'mean_slo_400_percent': statistics.mean(row['all_offered_slo_percent']['400'] for row in client),
            'mean_successful_samples_per_second': statistics.mean(row['client_throughput_samples_per_second'] for row in rows),
            'failed_requests': sum(row['failed_requests'] for row in client),
            'offered_requests': sum(row['offered_requests'] for row in client),
            'mean_failure_percent': statistics.mean(row['failure_percent'] for row in client),
            'all_repeats_successful': all(not row['failed_requests'] for row in client)}


def select(candidates, rows, greedy_plan, seeds=SEARCH_SEEDS):
    expected = {row['id']: row['allocation'] for row in candidates}
    assert allocation_plan_identity(greedy_plan) in {allocation_plan_identity(plan) for plan in expected.values()}, 'greedy outside candidate domain'
    grouped = defaultdict(dict)
    for row in rows:
        base, repeat = row['id'].rsplit('-r', 1)
        repeat = int(repeat)
        assert base in expected and 1 <= repeat <= len(seeds), 'unknown search job'
        assert row['runtime_seed'] == seeds[repeat-1], 'search seed mismatch'
        assert row['plan_identity']['resolved_plan_sha256'] == allocation_plan_identity(expected[base])
        assert repeat not in grouped[base], 'duplicate candidate replicate'
        grouped[base][repeat] = row
    assert set(grouped) == set(expected), 'incomplete candidate coverage'
    table = []
    for name, plan in expected.items():
        assert set(grouped[name]) == set(range(1, len(seeds)+1)), 'incomplete repeated candidate'
        stats = measures([grouped[name][repeat] for repeat in range(1, len(seeds)+1)])
        table.append({'id': name, 'plan_sha256': allocation_plan_identity(plan), 'resolved_plan': plan, **stats})
    successful = [row for row in table if row['all_repeats_successful']]
    assert successful, 'no candidate completes all repeats without application failure'
    selected = min(successful, key=lambda row: (row['mean_jct_ms'], row['plan_sha256']))
    greedy = next(row for row in table if row['plan_sha256'] == allocation_plan_identity(greedy_plan))
    if greedy['all_repeats_successful']:
        assert selected['mean_jct_ms'] <= greedy['mean_jct_ms']
    return {'selected': selected, 'greedy': greedy, 'candidates': table,
            'candidate_count': len(table), 'candidate_trials': len(rows),
            'application_failure_candidates': sum(not row['all_repeats_successful'] for row in table),
            'selection_scope': 'Minimum five-repeat mean among complete, failure-free static plans; empirical finite-trace reference, not certified expected-JCT optimality.'}


def validate(rows, greedy, selected, seeds=VALIDATION_SEEDS):
    same = allocation_plan_identity(greedy) == allocation_plan_identity(selected)
    labels = {'greedy'} if same else {'greedy', 'reference'}
    grouped = defaultdict(dict)
    plans = {'greedy': greedy, 'reference': selected}
    for row in rows:
        label, repeat = row['id'].rsplit('-v', 1)
        repeat = int(repeat)
        assert label in labels and 1 <= repeat <= len(seeds)
        assert row['runtime_seed'] == seeds[repeat-1], 'validation seed mismatch'
        assert row['plan_identity']['resolved_plan_sha256'] == allocation_plan_identity(plans[label])
        assert repeat not in grouped[label]
        grouped[label][repeat] = row
    assert set(grouped) == labels
    summaries = {}
    for label, group in grouped.items():
        assert set(group) == set(range(1, len(seeds)+1))
        summaries[label] = measures([group[repeat] for repeat in range(1, len(seeds)+1)])
    if same:
        summaries['reference'] = copy.deepcopy(summaries['greedy'])
    differences = [a-b if a is not None and b is not None else None
                   for a, b in zip(summaries['greedy']['per_repeat_jct_ms'],
                                   summaries['reference']['per_repeat_jct_ms'])]
    successful = all(value['all_repeats_successful'] for value in summaries.values())
    return {'same_allocation': same, 'unique_plans_measured': len(labels), **summaries,
            'paired_greedy_minus_reference_ms': differences,
            'paired_mean_difference_ms': statistics.mean(differences) if successful else None,
            'paired_sample_sd_ms': (statistics.stdev(differences) if len(differences) > 1 else 0) if successful else None,
            'jct_gap_percent': 100*(summaries['greedy']['mean_jct_ms']/summaries['reference']['mean_jct_ms']-1) if successful else None,
            'scope': 'Ten prespecified independent validation repeats after selection; no reselection. Identical allocations share measurements. Descriptive repeat means/SD; mean per-run percentiles are not pooled percentiles.'}


def run(root, resume, concurrency):
    protocol = read(root/'protocol.json')
    assert concurrency == protocol['concurrency'], 'concurrency differs from frozen protocol'
    assert protocol['search_seeds'] == list(SEARCH_SEEDS) and protocol['validation_seeds'] == list(VALIDATION_SEEDS)
    suite = Suite(root, resume)
    if not resume:
        suite.manifest['concurrency_gate'] = protocol['concurrency_gate']
        frozen = [root/'protocol.json', root/'build-identity.json', root/'provenance.json',
                  *root.glob('*/inputs/*.csv'), *root.glob('concurrency-calibration.json')]
        frozen += list((Path(__file__).resolve().parents[3]/'scripts/simulation').glob('*.py'))
        suite.manifest['frozen_files'].update({str(path.resolve()): sha(path) for path in frozen})
        save(suite.manifest_path, suite.manifest)
        suite.verify()
    suite.state.update(status='running', error=None)
    suite.checkpoint()
    def interrupt(signum, _frame):
        raise KeyboardInterrupt(f'repeated suite interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    try:
        for name in protocol['scenarios']:
            inputs = root/name/'inputs'
            suite.queue(name, 'native-check', inputs/'native-check-jobs.jsonl', 1)
            native = jsonl(root/name/'native-check/results.jsonl')
            assert len(native) == len(SEARCH_SEEDS)
            for repeat, row in enumerate(sorted(native, key=lambda row: row['sequence'])):
                assert row['runtime_seed'] == SEARCH_SEEDS[repeat]
                assert not row['client_metrics']['failed_requests'], 'native control has application failures'
            hashes = {row['plan_identity']['resolved_plan_sha256'] for row in native}
            assert len(hashes) == 1, 'native greedy plan changes across runtime seeds'
            greedy = read(Path(native[0]['output'])/native[0]['plan_identity']['resolved_path'])['plan']
            candidates = jsonl(inputs/'nominal-jobs.jsonl')
            assert allocation_plan_identity(greedy) in {allocation_plan_identity(row['allocation']) for row in candidates}
            rows = []
            certificates = []
            for repeat in range(1, len(SEARCH_SEEDS)+1):
                label = f'search-r{repeat}'
                certificates.append(suite.queue(name, label, inputs/f'chunks/{label}.jsonl', concurrency, True))
                rows.extend(jsonl(root/name/label/'results.jsonl'))
            search = select(candidates, rows, greedy)
            save(root/name/'search-selection.json', search)
            jobs = root/name/'validation-jobs.jsonl'
            expected_jobs = validation_jobs(greedy, search['selected']['resolved_plan'])
            if jobs.exists():
                assert jsonl(jobs) == expected_jobs, 'frozen validation plan changed'
            else:
                write_jobs(jobs, expected_jobs)
            suite.queue(name, 'independent-validation', jobs, 1)
            validation = validate(jsonl(root/name/'independent-validation/results.jsonl'), greedy, search['selected']['resolved_plan'])
            result = {'name': name, 'coverage': read(inputs/'coverage.json'), 'rates': read(inputs/'model-rates.json'),
                      'search': search, 'validation': validation, 'native_checks': measures(native),
                      'search_wall_seconds': sum(row['wall_seconds'] for row in certificates),
                      'native_plan_sha256': allocation_plan_identity(greedy), 'allowed_slices': [SLICE]}
            save(root/name/'scenario-result.json', result)
            suite.state.setdefault('scenario_results', {})[name] = str(root/name/'scenario-result.json')
            suite.checkpoint()
        results = [read(root/name/'scenario-result.json') for name in protocol['scenarios']]
        save(root/'suite-results.json', {'status': 'complete', 'protocol': protocol, 'scenarios': results,
             'metric_scope': 'Original-request weighted server JCT, frozen 30s mixed-batch arrivals plus complete drain. Failures reported; any failed search repeat excludes its plan from the failure-free reference.',
             'optimal_scope': 'Minimum prespecified repeated empirical mean over all 416 static 1g.10gb plans; independent validation cannot certify expected or dynamic-policy optimality.'})
        lines = ['# E2 repeated search and independent validation', '',
                 '| Case | Plans | Search greedy ms | Search reference ms | Validation greedy ms | Validation reference ms | Same allocation |',
                 '|---|---:|---:|---:|---:|---:|---|']
        for row in results:
            search, validation = row['search'], row['validation']
            values = [search['greedy']['mean_jct_ms'], search['selected']['mean_jct_ms'],
                      validation['greedy']['mean_jct_ms'], validation['reference']['mean_jct_ms']]
            cells = ' | '.join('-' if value is None else f'{value:.2f}' for value in values)
            lines.append(f"| {row['name']} | {search['candidate_count']} | {cells} | {validation['same_allocation']} |")
        lines += ['', 'Five search seeds and ten independent validation seeds were fixed before execution. Identical plans share the same comparative measurements. Validation does not reselect the reference.',
                  'JSON includes repeat values, sample SD, failure counts and per-run tail/SLO/throughput statistics. Tail means are not pooled percentiles. Any validation failures invalidate an unqualified mean-JCT comparison.']
        (root/'suite-report.md').write_text('\n'.join(lines)+'\n')
        suite.state.update(status='complete', active=None, error=None, finished_at_ms=round(time.time()*1000))
        suite.checkpoint()
    except BaseException as error:
        suite.stop()
        suite.state.update(status='stopped', error=f'{type(error).__name__}: {error}')
        suite.checkpoint()
        raise

