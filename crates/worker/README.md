# Inference workers

Model loading, unloading and execution backends. Device allocation, child processes, queues and batch control are managed by `compute_node` and `scheduler`.

[`Worker`](src/lib.rs) defines associated `Model` and `ModelRef` types. `model()` returns a loaded reference, `load_model()` loads a model, `unload_model()` waits for active references to drain, and `execute()` converts/checks a `DynTensor` before inference. The interface is composed with concrete types and generics, not a direct `Box<dyn Worker>`.

| Feature | Implementation | Requirements |
|---|---|---|
| `mock` | [`MockWorker`](src/mock.rs) | CSV profiles, lifecycle waits and execution noise; no weights |
| `torch` | [`TorchWorker`](src/torch/worker.rs) | TorchScript models, `tch` and LibTorch |
| `onnx` | [`OnnxWorker`](src/onnx/worker.rs) | ONNX models and an appropriate ONNX Runtime execution provider |

See [hardware deployment](../../docs/hardware.md) for backend builds and model naming. A CPU/mock check does not verify real inference outputs.

```sh
cargo test --locked -p worker --features mock --lib
```
