# CSV-driven simulation

## Execution path

```text
task_sender -> scheduler_daemon -> pool routing and model queues
            -> split/merge and adaptive batch control
            -> mock_virtual -> independent mock_vgpu processes
```

Each internal batch waits for its CSV-derived execution time. Mock workers return placeholder tensors: these runs evaluate scheduling, communication, lifecycle and queue behavior, not prediction accuracy of a real model.

For the four experiments used in the revised paper, use the [paper experiment guide](../resource/simulation/experiments/README.md). Its index follows the current manuscript order. Use `restore` before building a paper runtime. Root-level preparation instead uses the development source.

## Latency profiles

[`logs/model_latency.csv`](../logs/model_latency.csv) contains 1,760 rows covering eight models and eleven measured configurations. The parser uses `model_name,split_param,batch_size,avg_latency(s)`; the final legacy microsecond column is not used for waiting.

- Measured points are preserved, including non-monotone points.
- Between measured batch sizes, latency is interpolated linearly.
- Out-of-range batches, duplicate keys, non-finite/non-positive costs and missing profiles fail validation rather than being extrapolated.
- Runtime execution uses mean-preserving lognormal variation, normally CV=0.1. CV is a standard-deviation/mean ratio, not a prediction-error bound.
- Internal batches use 1-32 samples. A client request can also contain multiple samples and is not the same as an execution batch.

The reported collection procedure warms up, times 100 calls, removes the fastest and slowest 10, and averages the remaining 80. This is the author's retained collection description; complete historical driver/runtime/model versions and collection dates were not separately archived. The CSV measures the collection boundary, which may include input preparation and transfer costs, rather than kernel-only time.

Historical Huawei names use explicit aliases in simulation JSON:

| Current profile | AI Core / AI CPU / GiB | Historical CSV name |
|---|---|---|
| `vir05_1c_16g` | 5 / 1 / 16 | `vir05_1c_8g` |
| `vir10_3c_32g` | 10 / 3 / 32 | `vir10_3c_16g` |

These names follow the retained measurement mapping, not an inference from the memory suffix. `_m` / `_nm` rows remain distinct. Real-device daemon startup needs matching CSV names; the [hardware guide](hardware.md#6-profiles-and-configuration-fields) explains preparation of a separate alias-adjusted copy.

## Configuration fields

Paths in JSON are resolved relative to that JSON file. [`heterogeneous.json`](../resource/simulation/heterogeneous.json) is the generic two-pool preset.

| Field | Meaning |
|---|---|
| `models` | Models served; all permitted profiles need covered curves |
| `latency_csv` | Execution-cost measurements |
| `batching_config` | TOML supplying the batch extractor; preparation snapshots it |
| `nodes[].template_name` | Shared model-queue/routing pool; same-name nodes require compatible resources/profiles |
| `nodes[].gpus` | Number of simulated parent devices in that node |
| `nodes[].resource` | Compute/memory budget per parent |
| `nodes[].share_targets` | Allowed profiles; presence in the CSV does not enable a profile |
| `create_delay_ms` / `destroy_delay_ms` | Wall-clock resource lifecycle delays, normally 15,000 ms each |
| `model_lifecycle` | Default load/warmup 1,000 ms and unload 100 ms; model/profile/device overrides allowed |
| `execution_noise` | CV and seed; direct JSON loading defaults to no noise, managed runs normally use CV=0.1 |
| `control_timeout_ms` | Control-RPC budget, exceeding lifecycle waits with startup margin |
| `inference_timeout_ms` | Queue-plus-inference deadline, distinct from an SLO |

Creation on one parent is sequential. One configuration application can therefore take longer than a single create/delete pair. The global loop waits its interval **after** the previous application ends. Request-rate estimates use a recent 5 s window; routing refreshes every second; local batch updates occur after two successful executions. Paper runtimes retain their own settings.

## Running

Use Rust with edition-2024/let-chain support, Python 3.11+, C/C++ build tools and the [portable Cargo configuration](../resource/simulation/cargo-config.toml). CPU simulation needs no CUDA, CANN, LibTorch or weights. Linux builds may require OpenSSL development packages and `pkg-config`.

```sh
mkdir -p .cargo
test -e .cargo/config.toml || cp resource/simulation/cargo-config.toml .cargo/config.toml
python3 scripts/simulation/run.py --seconds 30
```

Inspect any existing Cargo configuration before changing it. Initial allocation can take several minutes at normal lifecycle delays. For a short development check:

```sh
python3 scripts/simulation/run.py --seconds 12 --warmup-seconds 2 \
  --create-delay-ms 20 --destroy-delay-ms 20 --schedule-interval 10
```

Use `--offline` after dependencies are cached or `--skip-build` with already built matching binaries. These short delays are not the paper's settings. The runner starts nodes, waits for model readiness, sends requests, drains and shuts down only its own process group. Startup without progress/routing files is not itself a failure; inspect startup, processes and host observations together.

The runners snapshot configuration, workload, code and binary identities. New builds/runs obtain their own identities and cannot be relabeled as the author's historical binaries or measurements.

## Randomness and replay

Replay rows are `[offset_seconds, model, samples]`. A workload seed fixes this offered trace separately from the runtime seed. The sender chooses request size before sampling an exponential gap based on its sample rate. Therefore a multi-sample trace is not a fixed-request-rate Poisson trace.

Mock execution, model loading, routing, extraction and adaptive rounding use explicit random streams. Worker/model-load budgets are recorded and exhaustion fails visibly. Logical ordering is fixed in experiment mode. Common seeds do not imply identical random consumption when allocation/execution paths differ, and real clocks/async scheduling still affect outcomes.

## Metrics and output files

| Output | Meaning |
|---|---|
| `summary.json` | Startup/sending/drain/shutdown timing, completion/failure totals and SLO summaries |
| `measurement/` | Client progress, final counters and optional per-request terminal records |
| `scheduler/` | Lifecycle, resource-application and worker-execution observations |
| `routing_weights.jsonl` | Periodic routing/configuration observations when enabled |
| `host_metrics.jsonl` | Host samples used for environment review |
| Input/identity manifests | Copied settings, CSV/replay hashes, source and binary identities |

Server JCT counts each successfully completed original request once, from server arrival to completion. End-to-end latency runs from planned arrival to the complete client response. End-to-end SLO attainment uses **all offered original requests**, including failures. Completed, failed and pending counts must reconcile; server-only success latency cannot replace the all-offered denominator. Reports preserve both 250 ms and 400 ms SLO summaries where available.

The batching comparison uses jointly completed request IDs for paired mean/P99 and all offered IDs for SLO/unfinished counts. Validation P99 is the mean of per-run percentiles, not a pooled percentile. Successful throughput includes sending and drain time where specified by that experiment.

Execution timing includes the measured call boundary and additional RPC/host work; queue waiting is separate. Decision time and resource-application time are also separate. Assigned capacity is not runtime utilization. Non-atomic worker-count observations do not establish a causal capacity ablation.

```sh
python3 scripts/simulation/report.py simulation-runs/YOUR_RUN
```

## Online correction

Successful batches update a per-pool, per-model/profile multiplicative correction with EMA weight 0.3. Input CSV files remain unchanged. Unexecuted/unsuccessful configurations lack correction evidence. Processing observations include preparation/RPC within the call boundary but exclude queue/batch waiting. Router weights are refreshed from corrected costs and the last applied allocation.

Correction preserves the base batch-curve shape. It is not guaranteed to follow arbitrary rapidly changing costs or calibrate configurations without successful observations.

## Workload switching and communication

The retained paper switching input has Equal/VGG/Equal/ResNet/VGG/Equal phases, each 360 s. Use the adopted 1x/8x/20x settings in the paper guide, not an earlier exploratory preset. Input rates are samples/s; request counts depend on sampled request sizes. Lifecycle delays and the 15 s post-application interval remain separate.

Request handoff preserves original completion channels and moves only unclaimed work. Already executing batches finish at their original worker. Failed target acceptance restores ownership. If no pool can serve, work waits for availability or its deadline. No blanket zero-interruption claim follows from this mechanism.

The managed sender defaults to persistent HTTP/2 data connections. Keep enough streams/in-flight capacity for the expected backlog. Input overflow, unexplained exits, missing terminal records and required-observation gaps invalidate completeness; they are not silently retried. Model-binding and seed/terminal audits check recorded identities and accounting, not output-tensor accuracy.

## Timeline reports and boundaries

The reporting tools can display demand, assignment, completion and SLO over time. Resource groups retain their native units (for example A100 compute equivalents and Huawei AI Cores). Do not turn assigned resources into utilization. Old runs without required timing/lifecycle records cannot support reconstructed resource traces.

CPU simulation exercises the communication and scheduling path but not real kernels, vendor isolation or model output correctness. Use [hardware deployment](hardware.md) for real inference and the [paper index](../resource/simulation/experiments/README.md) for the exact scope of available reproduction material.
