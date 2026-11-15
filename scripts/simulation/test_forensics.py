import json
from pathlib import Path
import tempfile
import unittest

from forensics import error_evidence, model_binding_audit, queue_timings


class ForensicTest(unittest.TestCase):
    def test_detects_stale_reported_model_after_physical_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            node, perf = Path(directory)/'node', Path(directory)/'perf'
            def row(second, fields):
                return json.dumps({'timestamp': f'1970-01-01T00:00:{second:02d}Z', 'fields': fields})
            node.write_text('\n'.join([
                row(0, {'event':'model_load_finished','worker_id':'w','model':'old'}),
                row(2, {'event':'model_unload_started','worker_id':'w'}),
                row(4, {'event':'model_load_finished','worker_id':'w','model':'new'})]))
            perf.write_text('\n'.join([
                row(1, {'worker_id':'w','model_name':'old'}),
                row(3, {'worker_id':'w','model_name':'old'}),
                row(5, {'worker_id':'w','model_name':'old'}),
                row(6, {'worker_id':'w','model_name':'new'}),
                row(7, {'worker_id':'unknown','model_name':'new'})]))
            result = model_binding_audit([node], [perf])
        self.assertEqual(result['counts'], {'records_checked':5,'matched':2,
            'not_loaded':1,'model_mismatch':1,'unknown_worker':1})

    def test_unload_clock_edge_is_reported_as_ambiguous(self):
        with tempfile.TemporaryDirectory() as directory:
            node, perf = Path(directory)/'node', Path(directory)/'perf'
            node.write_text('\n'.join(json.dumps({'timestamp': stamp, 'fields': fields})
                for stamp, fields in [
                    ('1970-01-01T00:00:00Z', {'event':'model_load_finished','worker_id':'w','model':'old'}),
                    ('1970-01-01T00:00:02Z', {'event':'model_unload_started','worker_id':'w'})]))
            perf.write_text(json.dumps({'timestamp':'1970-01-01T00:00:02.001Z',
                'fields': {'worker_id':'w','model_name':'old'}}))
            result = model_binding_audit([node], [perf])
        self.assertEqual(result['counts'], {'records_checked':1,'boundary_ambiguous':1})

    def test_undated_errors_do_not_inherit_startup_time_as_onset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'measurement.log'
            path.write_text("1970-01-01T00:00:00.000Z startup\nCan't assign requested address\n1970-01-01T00:01:00.000Z Can't assign requested address\n")
            result = error_evidence([path], 0)
        examples = result['first_examples']['cannot_assign_address']
        self.assertEqual(examples[0]['seconds_after_origin'], 60)
        self.assertIsNone(examples[1]['seconds_after_origin'])
        self.assertIsNone(examples[1]['timestamp'])

    def test_joins_out_of_order_logs_and_keeps_unfinished_requests_separate(self):
        rows = [('task_assigned', 'a', 10000), ('task_arrived', 'a', 0),
                ('compute_completed', 'a', 10020), ('task_arrived', 'b', 50)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trace.jsonl'
            path.write_text('\n'.join(json.dumps({'timestamp': '1970-01-01T00:00:00Z', 'fields': {'event': event, 'task_id': task, 'timestamp_ms': when}}) for event, task, when in rows))
            result = queue_timings([path])
        self.assertEqual(result['matched_completed'], 1)
        self.assertEqual(result['unmatched_active_at_log_end'], 1)
        self.assertEqual(result['queue_wait_p99_ms'], 10000)
        self.assertEqual(result['after_assignment_p99_ms'], 20)
        self.assertAlmostEqual(result['over_5s_queue_fraction'], 10000/10020)
        self.assertNotIn('slo_percent', result)

    def test_explicit_http_failure_is_not_reported_as_unknown_orphan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trace.jsonl'
            path.write_text('\n'.join(json.dumps({'timestamp':'1970-01-01T00:00:00Z',
                'fields':{'event':'task_arrived','task_id':task,'timestamp_ms':0}})
                for task in ['expired', 'still_pending', 'unknown']))
            result = queue_timings([path], {'expired':'failed', 'still_pending':'processing'})
        self.assertEqual(result['unmatched_active_at_log_end'], 3)
        self.assertEqual(result['unfinished_compute_http_statuses'],
            {'failed':1,'processing':1,'unknown':1})
