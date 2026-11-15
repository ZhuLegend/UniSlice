# Model and tensor interfaces

Shared tensor representations and inference interfaces for communication, workers and scheduling.

- [`Tensor<T>`](src/tensor/mod.rs) stores an element type, dimensions and `ndarray::ArrayD<T>` data.
- [`DynTensor`](src/tensor/dyn_tensor/mod.rs) handles dynamically typed tensors, batch splitting/concatenation, conversion and serialization.
- [`InferenceModel`](src/model.rs) defines associated input, output, load-configuration and error types, with `load()`, `check_input()` and `forward()` operations.

Workers accept `DynTensor` and convert it into backend inputs. Torch and ONNX model wrappers are implemented in [`worker`](../worker/README.md).

`serde` / `bincode` enable serialization, `half` adds half precision, `rand` generates test/workload data, and `torch` / `onnx` enable conversions requiring the relevant runtime.

```sh
cargo test --locked -p model --features bincode,serde,half --lib --test tensor_boundaries
```

These tensor checks need no GPU. Simulation uses placeholder outputs and does not check real-model accuracy.
