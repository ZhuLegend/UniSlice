# Implementation scope

## Components

| Capability | Implementation |
|---|---|
| Heterogeneous slice allocation | Buddy-cell/constrained resource models and greedy scheduling |
| Resource cost | `RankableShareTarget` / `Bid`, using the maximum relative resource component |
| Reconfiguration and reuse | Current/target state comparison and migration commands |
| Per-model queues and batching | `RequestScheduler`, sample accounting, split/timeout extractors |
| Adaptive batch control | Worker handlers; the current control loop resides on the scheduler side |
| Capacity routing | Profile-derived weights for successfully applied allocations |
| Execution-cost inputs | Shared model-level CSV parsing/interpolation and online correction |
| CPU simulation | Resource quotas, lifecycle waits, real queues/RPC and independent mock workers |
| Real inference | NVIDIA/LibTorch and Huawei/ONNX Runtime adapters |
| Operator-level profiling | Not bundled in this repository |

`model` defines tensor/inference interfaces; `worker` implements backends; `compute_node` owns devices and workers; `comm` provides binary RPC; `scheduler` owns global allocation, routing and HTTP service. `frontend` retains direct hardware clients.

## Queues and lifecycle

Node-side `BaseVirtualComputeNode` owns processes and physical-resource credentials. Scheduler-side `VirtualComputeNode` owns per-model queues and worker registration. `HeterogeneousSystem` connects pools, applies targets and transfers unclaimed requests when a pool loses its serving workers.

Retirement closes new extraction and waits for claimed batches before model/process cleanup. Unclaimed fragments retain their original response channel when transferred. Control timeouts preserve ownership for inspection; they do not prove that cleanup finished. Node commands preserve per-device ordering and stop dependent operations after an earlier failure. Resource starvation or cyclic migration dependencies can still create a temporary service gap.

Workers bind a listener before reporting their address through stdout. Logs use stderr. The parent validates the report and checks the worker; model loading must complete before inference.

## Development allocation semantics

The development allocator first reserves minimum feasible coverage for every model. That reservation consumes resources but does not subtract model demand. Remaining sample demand is initialized from the original arrival rates and the model with the largest signed residual is selected at each step.

For positive residual demand, candidate scoring uses `L(b) + (b-1)/(2*lambda*pi)`, where `lambda` is the remaining sample rate and `pi` is the largest relative resource component. After an incremental assignment, the residual is reduced using the profile's maximum saturated capacity `max_b b/L(b)`. Once demand is covered, spare resources are allocated by capacity gained per resource cost, avoiding a zero/negative rate in the waiting term.

The planning batch, capacity-maximizing batch and actual adaptive batch may differ. This development rule differs from pseudocode that subtracts capacity at the selected planning batch. The four paper experiments restore their respective measured source versions; do not replace those versions with the development allocator or infer a new proof from these implementation details.

## Online feedback

Each logical pool maintains a correction factor per model/resource profile. Successful batch processing updates it by an exponential moving average with weight 0.3 on the new observation. Only executed configurations receive feedback. Routing is refreshed from the successfully applied allocation and corrected profiles; queue waiting is not treated as execution capacity. A common multiplier does not reconstruct an arbitrary latency-versus-batch curve.

## Running the system

Follow [hardware deployment](hardware.md) for A100/Huawei prerequisites, model files, builds, nodes and daemon startup. Follow [simulation](simulation.md) for CPU execution. MPS has a low-level adapter but is not the MIG daemon startup path documented here. Hardware execution and every historical paper result have not been revalidated by the CPU checks.
