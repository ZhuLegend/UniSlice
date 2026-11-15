# Compute nodes

Compute nodes own device resources and worker processes. Start with the [A100 and Huawei deployment guide](../../docs/hardware.md) for real hardware, or the [root README](../../README.md) for CPU simulation.

| Interface | Role |
|---|---|
| [`ComputeDevice`](src/compute_device/mod.rs) | Device requests, logical indices and resource release |
| [`ShareableComputeDevice`](src/compute_device/vgpu/shareable.rs) | Reserve resources and create slices |
| [`BaseVirtualComputeNode`](src/compute_node/virtual/base.rs) | Create, load, unload and destroy physical workers |
| [`VirtualComputeNode`](src/compute_node/virtual/virtual_compute_node/mod.rs) | Scheduler-side model queues, worker registration and batching |
| [`WorkerProxy`](src/util/worker_proxy/mod.rs) | Start/connect workers and send inference/control RPCs |
| [`RunnableOn`](src/runnable_on.rs) | Compile-time compatibility between workers and devices |

Requests enter per-model queues and may be split or merged into execution batches. Retiring a worker stops new extraction, waits for already claimed batches, and then releases resources. Unclaimed work can move to another serving pool while retaining its original result channel. A cleanup timeout retains the handle for inspection/retry rather than claiming successful release.

A child worker binds its listener and reports its address through stdout; logs use stderr. The parent validates that report and checks the RPC service. Model loading must finish before the worker executes requests.

## Backends

- `mock`: configuration-driven resources and CSV execution waits; placeholder outputs, no model weights.
- `nvidia` with `torch`: NVIDIA MIG nodes and LibTorch inference.
- `huawei` with `onnx,cann`: Huawei virtual devices and ONNX Runtime CANN inference.
- `mock_virtual` requires `--simulation-config` and `--node`.

Hardware build flags, runtime libraries, model files, Docker requirements and node startup commands are documented in [hardware deployment](../../docs/hardware.md). CPU checks do not validate hardware execution or model output correctness.
