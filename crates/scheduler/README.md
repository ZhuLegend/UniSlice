# Scheduler

The scheduler owns resource allocation, cross-pool routing, arrival-rate estimates and HTTP service. Use [`scheduler_daemon`](src/bin/scheduler_daemon/main.rs) with the [request sender](src/bin/task_sender/README.md). Real-device deployment is described in the [hardware guide](../../docs/hardware.md).

## Decisions and execution

[`Scheduler`](src/lib.rs) computes a target from `SystemContext`; `SchedulableSystem` compares that target with current state and applies changes. [`HeterogeneousSystem`](src/schedule_system/virtual_compute_node/system/heterogeneous/system.rs) combines resource pools, allocators, model profiles and routing weights. [`GreedyScheduler`](src/schedule_system/schedulers/greedy.rs) assigns resources under device constraints. Invalid/infeasible demand can return an error without replacing the current allocation.

Reconfiguration and worker management are implemented in `compute_node`. The heterogeneous system orders changes across pools and transfers unclaimed requests when a pool loses its serving workers. Claimed batches finish before worker cleanup. The tree also contains single-device, buddy-cell, resource-enumeration and migration-matching components; their roles depend on the caller.

The removed `example_inss` feature and obsolete examples are not current startup paths. The [implementation guide](../../docs/architecture.md) describes development-versus-measurement scope.

## Service interfaces

- HTTP/1 and HTTP/2 task submission, health and statistics endpoints.
- `/health` means the HTTP listener is reachable. Inspect `/scheduler/stats` and node logs for allocation/model readiness.
- `--max-pending-requests` bounds accepted requests; `--request-memory-mib` bounds encoded inputs.
- `--interval 15` waits 15 seconds after a resource application ends before starting the next global round.
- Sender rates are samples/s, not requests/s.

The standalone daemon binds `0.0.0.0` by default and has no authentication layer. Choose a trusted interface explicitly. Local simulation binds loopback.

## CPU check

```sh
python3 scripts/simulation/run.py --seconds 12 --warmup-seconds 2 \
  --create-delay-ms 20 --destroy-delay-ms 20 --schedule-interval 10
```

This is a short development check. Use each paper experiment's restored runtime and settings for paper comparisons. See [simulation](../../docs/simulation.md) and [contributing](../../CONTRIBUTING.md).
