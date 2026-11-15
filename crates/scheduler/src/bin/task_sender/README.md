# Request sender

`task_sender` submits requests to `scheduler_daemon` and records offered, completed, failed and unfinished work. It supports Poisson arrivals, periodic rate changes, finite runs and exact replay.

From the repository root, after starting nodes and the daemon:

```sh
cargo run --locked -p scheduler --features task_sender --bin task_sender -- \
  --config scripts/exper_config/hardware/task_sender.toml \
  --daemon-url http://127.0.0.1:3000 \
  --duration-secs 30 --drain-timeout-secs 300
```

The [hardware guide](../../../../../docs/hardware.md) explains the one-model example and the historical A100/Huawei configurations. A TOML file must already exist; the sender does not generate one. For a managed CPU run, use `scripts/simulation/run.py`.

## Rate and batch fields

`models[].rate_range` is in **samples/s**. `batch_size_range` is the number of samples in a client request, not the worker's execution batch. Request inter-arrival times depend on both the selected sample rate and request size. The service may split or combine these requests internally.

Rate strategies are `sudden`, `smooth` and `random_walk`. See [`config.rs`](config.rs) for TOML fields and [`types.rs`](types.rs) for CLI arguments.

| Argument | Meaning |
|---|---|
| `--http-protocol http1` / `http2` | Transport; HTTP/1 is the sender default |
| `--http-connections` | Persistent data-client count; default 4 |
| `--max-in-flight` | Accepted client-future budget; overflow counts as failure |
| `--request-timeout-secs` | Request deadline including queues/reconfiguration; default 300 s |
| `--duration-secs` | Finite sending duration; otherwise runs continuously |
| `--drain-timeout-secs` | Wait for in-flight requests after sending; default 300 s |
| `--replay` | JSON `[offset_seconds, model_name, batch_size]` rows; requires finite duration |

Statistics use a separate control client. A finite run saves client totals before querying server statistics. Cancellation after a drain deadline is failure. Missing server responses are not counted as success and failed inference is not retried automatically.

Outputs under `sender.log_dir` include `progress.json`, `final_summary.json` and request/statistics files. Keep offered, completed, failed and unfinished counts together. See [simulation metrics](../../../../../docs/simulation.md#metrics-and-output-files) for the server/end-to-end distinction.
