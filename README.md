# UniSlice

UniSlice is a heterogeneous inference scheduling implementation. It includes NVIDIA MIG and Huawei virtual-slice resource models,
greedy allocation, slice reconfiguration, model-specific request queues,
dynamic batching and AIMD feedback control.

The repository also provides a **CPU-only, CSV-driven simulator**. It runs the
scheduler, communication protocols, queues and controllers, using measured
model execution times to simulate inference and returning placeholder tensors.
It supports scheduling and resource-management experiments; it does not check
model output correctness.

## Reproduce the paper experiments

The [paper experiment guide](resource/simulation/experiments/README.md)
describes the selected inputs, source versions, build commands, execution and
analysis procedures.

| Experiment in the revised paper | Command name | Conditions |
| --- | --- | --- |
| Workload switching and overload, Section IX-B | `e0` | 1×, 8× and 20×; six 360-second equal-demand/hotspot phases |
| Online prediction correction, Section IX-B | `e1` | Five initial biases: −40%, −20%, 0%, +20%, +40%; eight models, 30-second replay |
| Dynamic versus fixed batching, Section IX-C | `batching` | Low, medium, high and switching demand; retuned fixed batch sizes |
| Greedy versus optimal allocation, Section IX-C | `e2` | Six scenarios, 416 candidate allocations, independent search and validation seeds |

The guide's [paper-to-artifact index](resource/simulation/experiments/README.md#paper-to-artifact-index) follows the current manuscript's figures/tables, including the available hardware templates. Stable CLI names do not encode manuscript order. The index distinguishes complete controlled-experiment inputs from historical hardware material that is not a complete reproduction package.

The measured experiments use different frozen runtime versions. **Restore the
appropriate version before building**, rather than using the current development
code as a substitute. For example, from a Git checkout of this repository:

```sh
python3 scripts/simulation/experiments.py restore e1 --output simulation-runs/paper-e1-workspace
cd simulation-runs/paper-e1-workspace
cargo build --locked -p compute_node --features mock,vgpu,bin_help --bin mock_virtual --bin mock_vgpu
cargo build --locked -p scheduler --features scheduler_daemon,task_sender,mock,huawei,nvidia --bin scheduler_daemon --bin task_sender
python3 scripts/simulation/experiments.py prepare e1 --output simulation-runs/e1-new
python3 scripts/simulation/experiments.py run e1 --output simulation-runs/e1-new
python3 scripts/simulation/experiments.py analyze e1 --output simulation-runs/e1-new
```

Use `e0`, `e2` or `batching` in both the restore and experiment commands for the
other experiments, and choose fresh output paths. Preparation snapshots inputs
without starting an experiment. Each new run records its own source, binary
and input hashes; rebuilt binaries and new measurements are not relabeled as
the author's original run. Existing output directories are preserved. The
guide explains disk requirements, the fixed-batch search and result metrics.

The source archives contain only the runtime versions used by these experiments.
Unused exploratory conditions, manuscript and response-letter sources,
internal audits, historical result logs, model weights and compiled binaries
are excluded from the code release. Historical hardware experiments require
the corresponding devices, software and model setup; the CPU simulator does
not by itself reproduce every hardware result in the paper.

## Run on A100 or Huawei nodes

Follow the [hardware deployment guide](docs/hardware.md) for driver/runtime prerequisites, model preparation, backend builds, compute-node startup, scheduler configuration and requests. It explains the existing A100/Huawei presets and provides single-device service examples. Hardware deployments use real inference; they do not use mock binaries. The [paper index](resource/simulation/experiments/README.md) distinguishes these examples from the exact paper experiments.

## Requirements and a development quick start

Requirements are Rust with edition 2024 and let-chain support, C/C++ build
tools, and Python 3.11 or later. The CPU build was checked with Rust 1.98.1.
On Linux, HTTP/TLS dependencies generally also require `pkg-config` and OpenSSL
development packages. CPU simulation requires no CUDA, CANN, LibTorch or model
weights.

For a development run using the current system code, start at the repository root:

```sh
mkdir -p .cargo
test -e .cargo/config.toml || cp resource/simulation/cargo-config.toml .cargo/config.toml
python3 scripts/simulation/run.py --seconds 30
```

Preserve and inspect any existing `.cargo/config.toml` before changing it.
The script builds `mock_virtual`, `mock_vgpu`, `scheduler_daemon` and
`task_sender`, starts local nodes, waits for initial allocation, warms up,
measures, saves results and shuts down the processes it created. The generic
defaults use 15-second creation/deletion waits and 1/0.1-second model
loading/unloading waits. Initial setup can take several minutes because
creation operations on a device are sequential. These defaults are not a
replacement for each paper experiment's frozen settings.

For a short communication and control-path check, lifecycle delays can be
reduced explicitly:

```sh
python3 scripts/simulation/run.py --seconds 12 --warmup-seconds 2 \
  --create-delay-ms 20 --destroy-delay-ms 20 --schedule-interval 10
```

These short delays are for development checks. Add `--offline` after build
dependencies have been cached, or `--skip-build` to reuse existing binaries.

## Configuration and results

[`resource/simulation/heterogeneous.json`](resource/simulation/heterogeneous.json)
is the shared node/scheduler configuration for generic simulation. The default
contains one A100 resource pool and one Huawei B3 resource pool serving the
eight models in the CSV profile. The configuration specifies their capacities,
allowed partitions, model lifecycle and batch-size ranges.

[`logs/model_latency.csv`](logs/model_latency.csv) contains 1,760 measurements
covering eight models and eleven measured configurations. The current execution
path uses model-level CSV latency profiles. See the
[implementation guide](docs/architecture.md#development-allocation-semantics) for the
current allocation scoring and its relationship to the paper's pseudocode.

Runs are saved in separate `simulation-runs/` directories, including input
snapshots, CSV checksums, source/build identities, request results, lifecycle
logs and summaries. `summary.json` records completed, failed and unfinished
requests. The [experiment guide](resource/simulation/experiments/README.md#outputs-and-metrics)
defines the paper experiments' JCT, SLO, throughput and paired-comparison metrics.

## Documentation

- [Paper experiment reproduction](resource/simulation/experiments/README.md)
- [A100 and Huawei deployment](docs/hardware.md)
- [Simulation configuration and timing](docs/simulation.md)
- [Architecture and implementation scope](docs/architecture.md)
- [Contribution and test instructions](CONTRIBUTING.md)
- Component documentation: [compute node](crates/compute_node/README.md),
  [worker](crates/worker/README.md), [scheduler](crates/scheduler/README.md)

## Hardware and network scope

This is a research implementation and simulation toolkit. Real NVIDIA A100 or
Huawei execution needs the corresponding hardware, drivers, inference runtime,
model files and deployment configuration. The hardware entry points and
`scripts/exper_config` retain historical experiment settings; review addresses,
paths and device assumptions before use.

HTTP and internal RPC interfaces assume a trusted network and currently have
no authentication layer. A standalone `scheduler_daemon` binds to `0.0.0.0`
by default; use `--host 127.0.0.1` for local operation, as the simulation scripts
do. Use a controlled network for multi-machine experiments.

The [CSV documentation](docs/simulation.md#latency-profiles) records known measurement
scope and remaining provenance information. Confirm code, contributor and
measurement-data permissions before public release.

## License and citation

The project is dual-licensed under [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE). Third-party dependencies retain their own licenses;
model weights are not included. Formal paper citation metadata will be added
after the publication details are confirmed.
