"""Read complete, flushed terminal records; independent of paper analysis tools."""
import csv
from collections import defaultdict
import json
import math
import uuid

FIELDS = ['model_name', 'received_at', 'processing_duration_ms', 'batch_size', 'task_id', 'status', 'error']


def read_terminal_tasks(path):
    tasks, seen = [], set()
    with path.open(newline='') as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != FIELDS:
            raise ValueError('unexpected server terminal CSV schema')
        for row in reader:
            task_id = row['task_id']
            received, duration, batch = (int(row[key]) for key in ('received_at', 'processing_duration_ms', 'batch_size'))
            if (None in row or any(value is None for value in row.values())
                    or str(uuid.UUID(task_id)) != task_id or task_id in seen
                    or received < 0 or duration < 0 or batch <= 0
                    or row['status'] not in ('completed', 'failed') or not row['model_name']):
                raise ValueError('invalid or duplicate server terminal record')
            seen.add(task_id)
            tasks.append({'task_id': task_id, 'model': row['model_name'], 'batch_size': batch,
                          'received_at': received, 'completed_at': received + duration,
                          'status': row['status'], 'error': row['error'] or None})
    return tasks


def audit(run, runner_exit_code, runner_log=None):
    """Check new runner output without recovering or rewriting any files."""
    def read(path):
        return json.loads((run/path).read_text())

    summary, final = read('summary.json'), read('measurement/final_summary.json')
    if (runner_exit_code not in (0, 1) or read('shutdown.json')['errors']
            or (run/'memory_guard.json').exists() or not summary['all_requests_terminal']
            or not summary['all_offered_received'] or not summary['all_offered_samples_received']
            or summary.get('terminal_source') != 'server_csv_after_flush'):
        raise ValueError('run is incomplete or lacks complete server CSV measurement')
    tasks = read_terminal_tasks(run/'scheduler/task_terminal_log.csv')[summary['server_history_baseline']:]
    if tasks != read('tasks.json') or len(tasks) != summary['received_tasks']:
        raise ValueError('server CSV and collected records disagree')
    server = {task['task_id']: task for task in tasks}
    clients = {}
    with (run/'measurement/client-terminal.jsonl').open() as source:
        for line in source:
            row = json.loads(line)
            task_id = row['request_id']
            if task_id in clients or task_id not in server or type(row['success']) is not bool:
                raise ValueError('duplicate or unmatched client outcome')
            task = server[task_id]
            if ((row['model'], row['batch_size'], 'completed' if row['success'] else 'failed')
                    != (task['model'], task['batch_size'], task['status'])):
                raise ValueError('client/server terminal outcome disagrees')
            times = [row[key] for key in ('planned_due_ms', 'completed_at_ms', 'end_to_end_ms')]
            if not all(math.isfinite(value) for value in times) or times[2] < 0:
                raise ValueError('invalid client timing')
            clients[task_id] = row
    successful = [row for row in clients.values() if row['success']]
    if (set(clients) != set(server) or len(clients) != final['sent_tasks']
            or len(successful) != final['completed_tasks']
            or len(clients) - len(successful) != final['failed_tasks']
            or sum(row['batch_size'] for row in clients.values()) != final['sent_samples']
            or sum(row['batch_size'] for row in successful) != final['completed_samples']):
        raise ValueError('client terminal counts or samples disagree')
    if (run/'replay.json').exists():
        expected, observed = defaultdict(list), defaultdict(list)
        replay = read('replay.json')
        origin = final['planned_origin_ms']
        for offset,model,batch in replay:
            expected[model,batch].append(origin + offset*1000)
        for row in clients.values():
            observed[row['model'],row['batch_size']].append(row['planned_due_ms'])
        if set(expected) != set(observed) or len(replay) != len(clients):
            raise ValueError('client requests differ from the frozen replay')
        for key,times in expected.items():
            actual = sorted(observed[key])
            if len(times) != len(actual) or any(abs(a-b) > max(math.ulp(a),math.ulp(b))
                                               for a,b in zip(sorted(times),actual)):
                raise ValueError('client planned arrivals differ from the frozen replay')
    if (bool(runner_exit_code) != bool(final['failed_tasks'])
            or summary['sender_exit_code'] not in (0, 1)
            or (not final['failed_tasks'] and summary['sender_exit_code'] != 0)):
        raise ValueError('unexplained runner or sender exit code')
    if runner_exit_code and read('error.json')['error'] != 'measurement finished with explicit failed requests; complete records were preserved':
        raise ValueError('runner exit was not an explicit failed-request outcome')
    return tasks, summary, {'source': summary['terminal_source'], 'records': len(tasks), 'audit': 'complete-client-server-join'}
