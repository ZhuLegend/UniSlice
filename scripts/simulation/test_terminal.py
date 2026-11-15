import csv
import json
from pathlib import Path
import tempfile
import unittest
import uuid

from terminal import FIELDS, audit, read_terminal_tasks


class TerminalTest(unittest.TestCase):
    def test_full_csv_exceeds_history_cap_and_keeps_failures(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'terminal.csv'
            with path.open('w', newline='') as source:
                writer = csv.writer(source)
                writer.writerow(FIELDS)
                for index in range(100_001):
                    writer.writerow(['m', 1000, 20, 3, str(uuid.UUID(int=index)),
                                     'failed' if index == 0 else 'completed', 'deadline' if index == 0 else ''])
            tasks = read_terminal_tasks(path)
            self.assertEqual(len(tasks), 100_001)
            self.assertEqual(tasks[0]['status'], 'failed')
            self.assertEqual(tasks[-1]['completed_at'], 1020)
            with path.open('a') as source:
                source.write(f'm,1000,20,3,{tasks[0]["task_id"]},completed,\n')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                read_terminal_tasks(path)

    def test_audit_preserves_explicit_failures_and_excludes_warmup(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'scheduler').mkdir(); (root/'measurement').mkdir()
            ids = [str(uuid.UUID(int=i)) for i in range(3)]
            with (root/'scheduler/task_terminal_log.csv').open('w', newline='') as source:
                writer = csv.writer(source); writer.writerow(FIELDS)
                for index, task_id in enumerate(ids):
                    writer.writerow(['m', 1000, 20, 3, task_id, 'failed' if index == 2 else 'completed', 'deadline' if index == 2 else ''])
            tasks = read_terminal_tasks(root/'scheduler/task_terminal_log.csv')[1:]
            def dump(path, value): (root/path).write_text(json.dumps(value))
            dump('tasks.json', tasks)
            dump('shutdown.json', {'errors': []})
            dump('error.json', {'error': 'measurement finished with explicit failed requests; complete records were preserved'})
            dump('summary.json', {'all_requests_terminal': True, 'all_offered_received': True,
                                 'all_offered_samples_received': True, 'terminal_source': 'server_csv_after_flush',
                                 'server_history_baseline': 1, 'received_tasks': 2, 'sender_exit_code': 0})
            dump('measurement/final_summary.json', {'sent_tasks': 2, 'completed_tasks': 1,
                                                  'failed_tasks': 1, 'sent_samples': 6, 'completed_samples': 3})
            clients = [{'request_id': task['task_id'], 'model': 'm', 'batch_size': 3,
                        'success': task['status'] == 'completed', 'planned_due_ms': 1000,
                        'completed_at_ms': 1025, 'end_to_end_ms': 25} for task in tasks]
            (root/'measurement/client-terminal.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in clients))
            self.assertEqual(audit(root, 1)[0], tasks)
            with self.assertRaises(ValueError): audit(root, 0)


if __name__ == '__main__':
    unittest.main()
