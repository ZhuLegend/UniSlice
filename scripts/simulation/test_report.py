import json
from pathlib import Path
import tempfile
import unittest

from report import resource_timeline, resource_timelines, slice_units


class ResourceTimelineTest(unittest.TestCase):
    def test_preserves_initial_allocation_and_delayed_release(self):
        rows = [
            (0, 'create_started', {'slice': '1g.10gb'}),
            (1, 'model_loaded', {'model': 'a'}),
            (3, 'model_unloaded', {}),
            (5, 'worker_released', {}),
        ]
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / 'node.log'
            log.write_text('\n'.join(json.dumps({'timestamp': f'1970-01-01T00:00:0{t}Z', 'fields': {'event': event, 'id': 'w', **fields}}) for t, event, fields in rows))
            node = {'name': 'a100', 'kind': 'nvidia', 'gpus': 1, 'resource': {'ce': 7, 'memory': 80}}
            result = resource_timeline(node, ['a'], log, 2000, 6000)
        self.assertEqual(result['points'], [[0, 1, 0], [1, 0, 1], [3, 0, 0], [4, 0, 0]])
        self.assertEqual(result['units_per_card'], 7)

    def test_unknown_profile_is_not_guessed(self):
        self.assertEqual(slice_units('nvidia', '4g.40gb'), 4)
        self.assertEqual(slice_units('huawei', 'vir10_3c_32g'), 2)
        with self.assertRaises(ValueError):
            slice_units('huawei', 'vir10_3c_16g_nm')

    def test_failed_creation_retry_does_not_publish_a_partial_pool(self):
        node = {'name': 'a100', 'kind': 'nvidia', 'gpus': 1, 'resource': {'ce': 7, 'memory': 80}}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            events = [('create_started', {'slice': '1g.10gb'}), ('create_started', {'slice': '1g.10gb'}), ('model_loaded', {'model': 'a'})]
            (root/'a100.log').write_text('\n'.join(json.dumps({'timestamp': f'1970-01-01T00:00:0{i}Z', 'fields': {'event': event, 'id': 'w', **fields}}) for i, (event, fields) in enumerate(events)))
            resources, warnings = resource_timelines({'nodes': [node]}, ['a'], root, 0, 5000)
        self.assertEqual(resources, [])
        self.assertIn('duplicate create', warnings[0])


if __name__ == '__main__':
    unittest.main()
