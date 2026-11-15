# Hardware clients

Historical direct clients for the compute-node binary protocol. Normal service use goes through [`scheduler_daemon` and `task_sender`](../../docs/hardware.md); CPU simulation does not require these clients.

[`src/lib.rs`](src/lib.rs) provides `connect_gpulet()`, model loading/unloading and inference helpers. Here, a gpulet means a worker endpoint, not the Gpulet baseline scheduling algorithm. `SimpleComputeNodeReq` / `SimpleComputeNodeRes` are concrete binary-protocol request/response aliases, not the global resource-allocation interface.

| Binary | Features |
|---|---|
| `nv_torch` | `nv_torch` |
| `hw_onnx` | `hw_onnx` |
| `nv_standalone` | `uuid,nv_torch,vgpu` |
| `hw_standalone` | `uuid,hw_onnx,vgpu` |

Arguments and interactive commands follow the implementations in [`src/bin`](src/bin). Prepare the corresponding device, models and inference runtime first. These historical clients are not part of the CPU validation path. Use them only on a trusted network.
