import json
from pathlib import Path
import tempfile
import unittest

from long_analysis import resource_audit, task_metrics


class LongAnalysisTest(unittest.TestCase):
    def test_failed_and_pending_requests_remain_in_slo_denominator(self):
        tasks = [
            {'status': 'completed', 'received_at': 1000, 'completed_at': 1100},
            {'status': 'completed', 'received_at': 1000, 'completed_at': 1400},
            {'status': 'failed', 'received_at': 1000, 'completed_at': 1100},
            {'status': 'processing', 'received_at': 1000, 'completed_at': None},
        ]
        result = task_metrics(tasks)
        self.assertEqual(result['slo_percent'], {'250': 25, '400': 50})
        self.assertEqual((result['completed'], result['failed'], result['pending']), (2, 1, 1))
        self.assertEqual(result['p99_ms'], 400)

    def audit(self, capacity=4, omit_release=False, create_elapsed=15, profile='2g.20gb'):
        events = [
            (1, {'event':'resource_reserved','reservation_id':'r','amount':'NvidiaMigResource { ce: 2, memory: 20 }'}),
            (1, {'event':'create_started','id':'w','slice':profile,'delay_ms':15000}),
            (1+create_elapsed, {'event':'create_finished','id':'w'}),
            (18, {'event':'destroy_started','id':'w','delay_ms':15000}),
            (33, {'event':'destroy_finished','id':'w'}),
        ]
        if not omit_release:
            events += [(33, {'event':'resource_released','reservation_id':'r','amount':'NvidiaMigResource { ce: 2, memory: 20 }'}),
                       (33, {'event':'worker_released','id':'w'})]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'node.log').write_text('\n'.join(json.dumps({'timestamp':f'1970-01-01T00:00:{t:02d}Z','fields':e}) for t,e in events))
            config={'nodes':[{'name':'node','resource':{'ce':capacity,'memory':40},'gpus':1}]}
            return resource_audit(root,config,17000,36000)

    def test_slice_is_counted_until_release_and_timing_is_checked(self):
        result=self.audit()
        self.assertEqual(result['violations'], [])
        self.assertEqual(result['timing_pairs_checked'], 2)
        self.assertEqual(result['initial_profiles'], {'2g.20gb':1})
        self.assertEqual(result['runtime_destroys'], {'2g.20gb':1})
        self.assertEqual(result['profile_points'], [[0,1],[16,0],[19,0]])

    def test_capacity_leak_and_shortened_wait_are_not_hidden(self):
        self.assertTrue(any('capacity violation' in s for s in self.audit(capacity=1)['violations']))
        self.assertTrue(any('unreleased capacity' in s for s in self.audit(omit_release=True)['violations']))
        self.assertTrue(any('shorter' in s for s in self.audit(create_elapsed=1)['violations']))

    def test_huawei_profile_names_do_not_require_nvidia_format(self):
        result = self.audit(profile='vir05_1c_16g')
        self.assertEqual(result['violations'], [])
        self.assertEqual(result['profiles'], ['vir05_1c_16g'])


if __name__ == '__main__':
    unittest.main()
