# Paper simulation experiments

The four controlled experiments below are used in the revised paper. `restore` selects
the corresponding measured runtime source version before building. Each new
run records its own source, binary and input hashes; rebuilt binaries and
host-dependent timing are not relabeled as the author's original measurements.
The release includes no historical run logs or unused experimental conditions.

| Paper experiment | CLI name | Protocol and inputs |
|---|---|---|
| Switching and overload (IX-B) | `e0` | Eight models, 1×/8×/20× arrival rates, six 360 s phases per rate; exact retained replays, online allocation and batching |
| Initial prediction error (IX-B) | `e1` | Eight models at 100 samples/s each, 30 s replay, five initial prediction multipliers: 1.0, 0.6, 0.8, 1.2, 1.4; initial greedy allocation held fixed; online prediction calibration and dynamic batching |
| Dynamic versus fixed batching (IX-C) | `batching` | Two models, fixed allocation, low/medium/high and low–high–low traces; bounded fixed-target search and five independent paired validation seeds |
| Greedy versus optimal allocation (IX-C) | `e2` | Six scenarios, 416 feasible static placements; five search seeds and ten independent validation seeds; dynamic batching in both arms |

## Paper-to-artifact index

The index follows the revised manuscript's display order. CLI identifiers remain stable and do not indicate the order in which experiments were performed. Each configuration directory is relative to this guide. Use the matching `runtime/` source archive through `restore` for a controlled paper experiment.

| Manuscript evidence | Execution/configuration entry | Available material |
|---|---|---|
| IX-B, Figure 5: balanced A100 loads | [A100 hardware setup](../../../docs/hardware.md), historical `scripts/exper_config/4nv+4nv/` | Real-device code and historical templates; exact per-load baseline recipes/results are not bundled |
| IX-B, Figure 6: imbalanced A100 load | Same hardware templates | Sender example has doubled VGG demand; not a frozen reproduction of every plotted result |
| IX-B, Figure 8 / Table V: switching/overload | `e0/`, CLI `e0` | Selected 1x/8x/20x replays, settings, runtime and analysis |
| IX-B, Figure 9: online correction | `e1/`, CLI `e1` | Five bias conditions, execution/prediction CSVs, replay and runtime |
| IX-C, Figure 7: hardware component ablation | A100 hardware code/templates | Exact variant configurations and original aggregate results are not bundled |
| IX-C, Table VI: batching ablation | `batching/`, CLI `batching` | Four traces, fixed placements, bounded tuning, paired validation and analysis |
| IX-C, Figure 10: hardware JCT breakdown | A100 hardware code/templates | Exact historical result data and plotting input are not bundled |
| IX-C, Table VII: greedy/optimal | `e2/`, CLI `e2` | All 416 feasible placements in six domains, search/validation seeds and runtime |
| IX-D, Figure 11: 24-NPU distribution | [Huawei setup](../../../docs/hardware.md), historical `scripts/exper_config/8hw/` | Node/worker code and example templates; folder name does not instantiate a 24-NPU deployment |
| IX-D, Figures 12-13: overhead/scaling | `crates/scheduler/benches/`, hardware code | Benchmark source, but no complete historical 32-NPU/baseline/scaling result package |

The [hardware guide](../../../docs/hardware.md) supplies commands for starting the current system and explains both historical presets. Its new one-device configs under `scripts/exper_config/hardware/` are service examples, not additional paper experiments. The measured model-level costs are in `logs/model_latency.csv`; these are execution inputs, not the original JCT/throughput results for the hardware figures. An operator-level profiling package is not included. New executions generate new result files; the release contains no historical run logs.

The four sets use CSV execution profiles and CV=0.1 execution-time variation.
CV describes runtime variation, not a bound on prediction error. Internal batch
sizes range from 1 to 32. E0 uses resource creation/destruction costs of
12.545 s and model loading/unloading costs of 0.836/0.084 s; these are
per-operation costs, not a fixed duration for a complete reconfiguration.
E1 retains 15 s creation/destruction waits; E2/batching set resource lifecycle
costs to zero and hold placement fixed. Exact model rates, seeds, counts and timeouts are in each
`settings.json` and copied into the prepared `protocol.json`.

## Restore the paper runtime and build

Use Python 3.11+ and the CPU build setup in the [root README](../../../README.md).
From the repository root, select one of `e0`, `e1`, `e2`, `batching`:

```sh
python3 scripts/simulation/experiments.py restore e1 --output simulation-runs/paper-e1-workspace
cd simulation-runs/paper-e1-workspace
cargo build --locked -p compute_node --features mock,vgpu,bin_help --bin mock_virtual --bin mock_vgpu
cargo build --locked -p scheduler --features scheduler_daemon,task_sender,mock,huawei,nvidia --bin scheduler_daemon --bin task_sender
```

Add `--offline` when the dependencies are already cached. The experiment runners
use these four binaries in the restored workspace's `target/debug/`. Keep
that default location for the older paper runners. `PAPER_RUNTIME.json`
records the original full-tree identity, archive/member checksums and the
public subset's scope. Rust/Python execution files and parsed Cargo settings are
preserved; archive documentation is omitted and non-executable comments/license branding are updated in English, as recorded in the runtime manifest. Portable orchestration and selected inputs are added on restore,
so each new run records its new complete source fingerprint.

The archives contain source only. E0 uses the selected 1×/8×/20× inputs;
E1 uses the adopted five bias cases; E2 and batching restore their own
measurement source versions. Restore inside a Git checkout, as above: the
older E1/E2/batching runners read that checkout's Git metadata for provenance.
The runtime archives intentionally contain no Git history. Existing output directories are preserved.
The ordinary `prepare` command in the root checkout remains available for
new development runs, but does not select the paper's runtime version.

## Prepare, run, analyze

Inside the restored workspace, for example:

```sh
python3 scripts/simulation/experiments.py prepare e1 --output simulation-runs/e1-new
python3 scripts/simulation/experiments.py run e1 --output simulation-runs/e1-new
python3 scripts/simulation/experiments.py analyze e1 --output simulation-runs/e1-new
```

Replace `e1` with `e0`, `e2` or `batching`, choosing a fresh output directory.
Preparation only checks and snapshots inputs; it starts no processes. `run`
verifies the prepared bytes and current source, records binary identities,
executes the suite, audits it, and writes the report. Rebuild after Rust changes
and prepare again after execution-source changes. `analyze` reads existing run
records without launching experiments.

Suites run serially by default. E2 requires at least 8 GiB of free disk before
starting a queue. Completed search chunks are archived losslessly after hash
verification. E2 alone supports `run ... --resume`: verified complete chunks are
reused; incomplete chunks stop for inspection instead of being silently retried.
For parallel E2 preparation, `--concurrency N --calibration reviewed.json` needs
a reviewed host-interference report with `gate_passed: true`. Its file is saved
with the suite. The batching experiment requires a 2 GiB reserve and runs its own two-sided concurrency
screen before tuning; validation remains serial. Do not bypass a failed screen
or rerun measurements to obtain favorable results.

E2's 416 candidates include all feasible placements for the capacities and
slice type in its input configurations, including unused capacity. Selection
uses the five-seed mean among failure-free candidates; validation never
reselects the winner. Identical greedy/selected placements share measurements.
The selected optimum is an empirical optimum within this enumerated domain.

The batching experiment searches target pairs in `{1,2,4,8,16,32}²`, then each coordinate's immediate
integer neighbors around the coarse winner, using two disjoint local-search
seeds. There are at most 216 tuning, 40 validation and 11 calibration trials.
Both arms retain the same allocation and offered requests. The dynamic arm uses
the existing controller; the fixed arm sets one target per model with
`SplitAndSatisfy` extraction. The restored batching version passes its
fixed target through the original mock-worker environment interface; the
current development runner uses `--experiment-fixed-batches`. The pipeline
clears inherited targets and records which interface is selected.

## Outputs and metrics

`report.json`, `summary.csv` and `report.md` are the main summaries. E1 also
writes per-configuration feedback and prediction-error summaries. E2 retains
candidate coverage, search selection, independent validation, seed and terminal
audits. The batching experiment retains search-stage common request IDs, sample accounting, actual
executed batch histograms, `paired-validation-metrics.json` and
`paired-phase-metrics.csv` for the switching trace.

Server mean/P99 use one vote per successful original request. End-to-end
400 ms SLO uses **all offered original requests**, including failures. The batching experiment's
paired mean/P99 use the same jointly completed request IDs in both arms; its
SLO and unfinished counts retain all offered requests. Reported validation P99
is the mean of per-run percentiles, not a pooled percentile. Observation-end
censoring in the batching experiment is retained and audited; it is not discarded as an invalid
candidate. Missing terminal records, unexplained process exits and host
sampling gaps stop the suite.

Inputs use relative profile paths; preparation resolves only the new output
copies. [`input-manifest.json`](input-manifest.json) records per-file hashes.
Only conditions used in the current paper are release inputs. Earlier E1
switching protocols, opposed resource-group biases and other exploratory
load levels are retained locally, outside the released experiment inputs.
