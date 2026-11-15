import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest

from paper import queue as run_queue
from run import allocation_plan_identity
from paper.allocation import SEARCH_SEEDS, VALIDATION_SEEDS, search_jobs, select, validate, validation_jobs


def plan(count):
    return {'schema_version': 1, 'placements': [
        {'template_name': 'full', 'model': 'resnet50', 'slice': '1g.10gb', 'count': count},
        {'template_name': 'full', 'model': 'vgg19', 'slice': '1g.10gb', 'count': 1}]}


def measured(job, jct, failed=0):
    return {'id': job['id'], 'runtime_seed': job['runtime_seed'],
            'plan_identity': {'resolved_plan_sha256': allocation_plan_identity(job['allocation'])},
            'server_summary': {'mean_jct_ms': jct, 'p95_jct_ms': jct*2, 'p99_jct_ms': jct*3},
            'client_metrics': {'failed_requests': failed, 'offered_requests': 100,
                               'failure_percent': failed, 'all_offered_slo_percent': {'400': 99}},
            'client_throughput_samples_per_second': 10}


class RepeatedTest(unittest.TestCase):
    def candidates(self):
        return [{'id': 'c00001', 'allocation': plan(1)}, {'id': 'c00002', 'allocation': plan(2)}]

    def rows(self):
        candidates = self.candidates()
        # Candidate 1 has the best single run but candidate 2 the best mean.
        values = [(1, 20), (50, 20), (50, 20), (50, 20), (50, 20)]
        return [measured(job, values[repeat-1][index])
                for repeat, seed in enumerate(SEARCH_SEEDS, 1)
                for index, job in enumerate(search_jobs(candidates, seed, repeat))]

    def test_selection_uses_full_common_seed_mean_and_greedy_domain(self):
        result = select(self.candidates(), self.rows(), plan(1))
        self.assertEqual(result['selected']['id'], 'c00002')
        self.assertEqual(result['selected']['mean_jct_ms'], 20)
        self.assertEqual(result['greedy']['mean_jct_ms'], 40.2)
        self.assertEqual(result['candidate_trials'], 10)
        self.assertTrue(set(SEARCH_SEEDS).isdisjoint(VALIDATION_SEEDS))

    def test_any_failed_search_repeat_excludes_plan_without_hiding_failure(self):
        rows = self.rows()
        rows[1]['client_metrics']['failed_requests'] = 1
        result = select(self.candidates(), rows, plan(1))
        self.assertEqual(result['selected']['id'], 'c00001')
        self.assertEqual(result['application_failure_candidates'], 1)
        self.assertEqual(result['candidates'][1]['failed_requests'], 1)

    def test_incomplete_duplicate_or_wrong_seed_search_cannot_select(self):
        original = self.rows()
        for rows in [original[:-1], original+[original[0]]]:
            with self.assertRaises(AssertionError):
                select(self.candidates(), rows, plan(1))
        rows = copy.deepcopy(original)
        rows[0]['runtime_seed'] += 1
        with self.assertRaisesRegex(AssertionError, 'seed mismatch'):
            select(self.candidates(), rows, plan(1))
        with self.assertRaisesRegex(AssertionError, 'outside candidate domain'):
            select(self.candidates(), original, plan(3))

    def test_equal_allocations_share_validation_values_and_exact_zero_gap(self):
        jobs = validation_jobs(plan(1), plan(1))
        self.assertEqual(len(jobs), 10)
        result = validate([measured(job, 10+index) for index, job in enumerate(jobs)], plan(1), plan(1))
        self.assertTrue(result['same_allocation'])
        self.assertEqual(result['unique_plans_measured'], 1)
        self.assertEqual(result['greedy'], result['reference'])
        self.assertEqual(result['paired_greedy_minus_reference_ms'], [0]*10)
        self.assertEqual(result['jct_gap_percent'], 0)

    def test_independent_validation_can_reverse_without_reselecting(self):
        jobs = validation_jobs(plan(1), plan(2))
        self.assertEqual(len(jobs), 20)
        rows = [measured(job, 10 if job['id'].startswith('greedy-') else 12) for job in jobs]
        result = validate(rows, plan(1), plan(2))
        self.assertEqual(result['paired_mean_difference_ms'], -2)
        self.assertLess(result['jct_gap_percent'], 0)
        rows[0]['client_metrics']['failed_requests'] = 1
        failed = validate(rows, plan(1), plan(2))
        self.assertIsNone(failed['jct_gap_percent'])
        self.assertIsNone(failed['paired_mean_difference_ms'])
        self.assertEqual(failed['greedy']['failed_requests'], 1)

    def test_queue_seed_is_explicit_u64_and_does_not_replace_workload_seed(self):
        config = {'models': ['resnet50', 'vgg19'], 'nodes': [
            {'name': 'parent', 'template_name': 'full', 'kind': 'nvidia', 'gpus': 1,
             'share_targets': ['1g.10gb'], 'resource': {'ce': 7, 'memory': 80}}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'jobs.jsonl'
            path.write_text(json.dumps({'id': 'native', 'policy': 'greedy', 'runtime_seed': 0})+'\n')
            self.assertEqual(list(run_queue.jobs(path, config))[0]['runtime_seed'], 0)
            for invalid in [True, None, -1, 2**64, 1.5]:
                path.write_text(json.dumps({'id': 'native', 'policy': 'greedy', 'runtime_seed': invalid})+'\n')
                with self.assertRaisesRegex(ValueError, 'runtime seed'):
                    list(run_queue.jobs(path, config))
        args = argparse.Namespace(seconds=30, rate=75, seed=20260923, replay=Path('replay.json'),
                                  noise_cv=.1, ready_timeout=900, runner_memory_gib=4,
                                  model_rates=None, drain_seconds=None)
        command = run_queue.command(args, Path('config.json'), Path('out'), 24300, None, SEARCH_SEEDS[0])
        self.assertEqual(command[command.index('--seed')+1], '20260923')
        self.assertEqual(command[command.index('--random-seed')+1], str(SEARCH_SEEDS[0]))
        self.assertNotIn('--random-seed', run_queue.command(args, Path('config.json'), Path('out'), 24300, None))


if __name__ == '__main__':
    unittest.main()
