"""Passive POSIX process sampling for long simulation runs (no scheduler RPCs)."""
import json
import os
import subprocess
import threading
import time


def descendants(rows, roots):
    selected = set(roots) & rows.keys()
    while True:
        extra = {pid for pid, row in rows.items() if row['ppid'] in selected} - selected
        if not extra:
            return selected
        selected.update(extra)


class ProcessMonitor:
    def __init__(self, output, processes, interval=15, max_rss_kib=None):
        self.output, self.processes, self.interval = output, processes, interval
        self.max_rss_kib = max_rss_kib
        self.limit_exceeded = threading.Event()
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.run, name='simulation-observer', daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stopped.set()
        self.thread.join(timeout=10)

    def run(self):
        with self.output.open('w') as log:
            while True:
                sample = {'timestamp_ms': round(time.time() * 1000)}
                try:
                    output = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,rss=,pcpu='], text=True, timeout=5)
                    rows = {}
                    for line in output.splitlines():
                        pid, ppid, rss, cpu = line.split()
                        rows[int(pid)] = {'ppid': int(ppid), 'rss_kib': int(rss), 'ps_cpu_percent': float(cpu)}
                    roots = [p.pid for p in list(self.processes) if p.poll() is None]
                    selected = descendants(rows, roots)
                    # Include the runner's retained replay data, but not this ps helper.
                    if os.getpid() in rows:
                        selected.add(os.getpid())
                    sample.update(process_count=len(selected),
                                  rss_kib=sum(rows[p]['rss_kib'] for p in selected),
                                  ps_cpu_percent=sum(rows[p]['ps_cpu_percent'] for p in selected),
                                  processes={str(p): rows[p] for p in sorted(selected)})
                    if self.max_rss_kib is not None and sample['rss_kib'] > self.max_rss_kib and not self.limit_exceeded.is_set():
                        sample['host_memory_limit_kib'] = self.max_rss_kib
                        self.output.with_name('memory_guard.json').write_text(json.dumps(sample, indent=2))
                        self.limit_exceeded.set()
                        # Stop arrivals and let node parents perform their existing graceful cleanup.
                        for process in list(self.processes):
                            if process.poll() is None:
                                try: process.terminate()
                                except ProcessLookupError: pass
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    sample['error'] = str(exc)
                log.write(json.dumps(sample) + '\n')
                log.flush()
                if self.stopped.wait(self.interval):
                    return
