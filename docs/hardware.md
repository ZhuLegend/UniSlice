# A100 and Huawei deployment

This guide starts the current UniSlice system on real devices: compute nodes, slice workers, the scheduling daemon and a request sender. The single-device configurations below are service examples, not replacements for the paper's measured configurations. The [paper index](../resource/simulation/experiments/README.md) identifies the available paper inputs and the remaining historical hardware-reproduction material.

## 1. Deployment layout

```text
request sender -> scheduler_daemon (HTTP)
                  -> nv_virtual / hw_virtual (node RPC)
                     -> nv_vgpu / hw_vgpu (one child per slice)
```

Start nodes first, the daemon second and the sender last. Each node advertises the devices selected by its repeated `--device-id` arguments. A `remote_nodes` entry names a node process, not necessarily one physical card. Entries sharing `template_name` form one logical pool and use the same resource model.

Use dedicated devices and a trusted network. Nodes create/delete device slices, so existing workloads on those devices can be affected. The services do not supply an authentication layer. Use explicit addresses and your site's normal network controls.

## 2. Host prerequisites

All commands run from a checkout root on Linux. Prepare Rust with edition-2024/let-chain support, a C/C++ toolchain, Python 3.11+, `pkg-config` and OpenSSL development libraries. Preserve an existing `.cargo/config.toml`; a fresh checkout can use:

```sh
mkdir -p .cargo
test -e .cargo/config.toml || cp resource/simulation/cargo-config.toml .cargo/config.toml
```

The copied file selects build/profile flags; it does not turn hardware builds into mocks.

### NVIDIA A100

Install a compatible NVIDIA driver, CUDA toolkit and CUDA-enabled LibTorch **2.6.0**, as required by locked `tch=0.19.0`. Set paths for your installation:

```sh
export CUDA_HOME=/usr/local/cuda
export LIBTORCH=/opt/libtorch
export LD_LIBRARY_PATH="$LIBTORCH/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
nvidia-smi -L
```

Choose a LibTorch CUDA build matching the host, using the [official version instructions](https://docs.pytorch.org/get-started/previous-versions/). On designated A100 devices, enable MIG through your administrator's normal procedure before starting the node. Confirm device indices with `nvidia-smi`; do not register the same physical parent through overlapping node processes. The MIG daemon path does not require the separate MPS helper scripts.

### Huawei Ascend

Install the Ascend driver/firmware, a compatible CANN environment, DCMI, Docker and Ascend Docker Runtime. Run `npu-smi info` on the host first. The DCMI wrapper reads `HW_DCMI_PATH` when its headers/libraries are not in the default `/usr/local/dcmi` layout. Source the installed CANN environment and add its driver/DCMI libraries to the native search path as required by that installation.

`Cargo.lock` selects ONNX Runtime bindings with C API version 22. Supply an ABI-compatible ONNX Runtime 1.22.x library **with the CANN execution provider**, matching the host's CANN version and CPU architecture. An ordinary CPU-only ONNX Runtime library is insufficient. For dynamic loading, set `ORT_DYLIB_PATH` to its absolute `libonnxruntime.so` path and include its dependent libraries in `LD_LIBRARY_PATH`.

The slice proxy launches workers with Docker, setting `ASCEND_RUNTIME_OPTIONS=VIRTUAL` and the assigned virtual-device identifier. It does not add `--runtime`, so configure Ascend Docker Runtime as Docker's default runtime on the worker host. See the vendor's [runtime status instructions](https://www.hiascend.com/document/detail/en/mindcluster/2610/clustersched/schedulingug/docs/en/scheduling/installation_guide/03_confirming_status.md) and [virtual-device runtime options](https://www.hiascend.com/doc_center/source/en/mindcluster/730/clustersched/schedulingug/dlruntime_ug_006.html).

## 3. Models and model configuration

Models are not included. A node reads one JSON file per model from `--model-config`, and loads a matching file from `--model-dir`:

| Backend | Model file | JSON requirements |
|---|---|---|
| LibTorch | `MODEL.pt` (TorchScript) | `input_type`, `output_type`, `input_dim`, `torch_model_config` |
| ONNX Runtime | `MODEL.onnx` | Same tensor fields and `onnx_model_config` |

For example, `resnet50.pt` or `resnet50.onnx` pairs with [`resnet50.json`](../resource/config/model_config/resnet50.json). The JSON uses `Float` for float32; `input_dim: [3,224,224]` excludes the batch dimension. The worker supplies the leading batch dimension. Torch JSON controls warmup; ONNX JSON controls optimization level and thread settings. Use the per-model directory, not the historical aggregate `resource/config/compute_node/model_config.json` file.

For a one-model service, copy only that model's JSON into a separate directory so the node does not advertise other models:

```sh
export MODEL_DIR=/srv/unislice/models
export MODEL_CONFIG_DIR=/srv/unislice/model-config
mkdir -p "$MODEL_DIR" "$MODEL_CONFIG_DIR"
cp resource/config/model_config/resnet50.json "$MODEL_CONFIG_DIR/resnet50.json"
```

An example export can be produced in a Python environment with compatible PyTorch/torchvision packages (`torch==2.6.0`, `torchvision==0.21.0`) and `onnx`. This downloads torchvision's pretrained weights; use your retained model files instead when reproducing a particular measurement:

```sh
python3 - <<'PY'
import os
from pathlib import Path
import torch
from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.DEFAULT).eval()
directory = Path(os.environ['MODEL_DIR'])
torch.jit.script(model).save(str(directory/'resnet50.pt'))
torch.onnx.export(model, torch.zeros(1,3,224,224), str(directory/'resnet50.onnx'),
                  input_names=['input'], output_names=['output'], opset_version=17,
                  dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
                  dynamo=False)
PY
```

Check the exported model's supported operators and dynamic batches against your chosen backend. Record weight versions, preprocessing and export settings. The eight historical model names alone do not identify a unique checkpoint/export.

## 4. Build and start an A100 node

On the A100 host:

```sh
cargo build --locked -p compute_node --features nvidia,torch,vgpu,bin_help \
  --bin nv_virtual --bin nv_vgpu
RUST_LOG=info target/debug/nv_virtual \
  --device-id 0 --port 9000 \
  --model-dir "$MODEL_DIR" --model-config "$MODEL_CONFIG_DIR" \
  --bin-worker-path "$(pwd)/target/debug/nv_vgpu" \
  --worker-type torch --batch-size-only
```

Leave this process running. For multiple independent parents, repeat `--device-id`, for example `--device-id 0 --device-id 1`. The node starts slice workers as allocations are applied; do not manually start `nv_vgpu` for this flow. The parent sets each worker's `CUDA_VISIBLE_DEVICES` to the assigned MIG identifier.

`--batch-size-only` is intended for synthetic service/load checks: the node reconstructs image-shaped inputs instead of receiving full image payloads. It is not an accuracy test. An application sending actual tensors should omit this option and use the matching request/tensor interface.

## 5. Build and start a Huawei node

On the Huawei host, in the matching CANN/DCMI build environment:

```sh
cargo build --locked -p compute_node --features huawei,onnx,cann,vgpu,bin_help \
  --bin hw_virtual --bin hw_vgpu
ldd target/debug/hw_vgpu
```

Resolve every required shared library. Build workers in an environment compatible with the worker container's architecture, libc and native libraries. The container must contain compatible CANN, ONNX Runtime/CANN and worker dependencies. A small packaging recipe is provided at [`scripts/containers/ascend-worker.Dockerfile`](../scripts/containers/ascend-worker.Dockerfile):

```sh
# ASCEND_BASE_IMAGE is an existing, validated image in your environment.
# onnx/ contains libonnxruntime.so and its required provider libraries.
docker build -f scripts/containers/ascend-worker.Dockerfile \
  --build-arg BASE_IMAGE="$ASCEND_BASE_IMAGE" \
  -t unislice-ascend:local .
export HW_VGPU_DOCKER_IMAGE=unislice-ascend:local
export HW_VGPU_STARTUP_TIMEOUT_SECS=60
export HW_VGPU_KERNEL_CACHE_PATH=/srv/unislice/atc-cache
mkdir -p "$HW_VGPU_KERNEL_CACHE_PATH"
docker info --format '{{.DefaultRuntime}}'
docker run --rm -e ASCEND_VISIBLE_DEVICES=0 "$HW_VGPU_DOCKER_IMAGE" npu-smi info
```

The default runtime check should report `ascend`. The base image and `onnx/` library payload are environment-specific dependencies, not files supplied by this repository. The recipe does not install drivers or manufacture a compatible CANN runtime.

Start the parent process with absolute paths:

```sh
RUST_LOG=info target/debug/hw_virtual \
  --device-id 0 --port 9000 \
  --model-dir "$MODEL_DIR" --model-config "$MODEL_CONFIG_DIR" \
  --bin-worker-path "$(pwd)/target/debug/hw_vgpu" \
  --worker-type onnx --batch-size-only
```

The proxy mounts the worker binary, models and model-config directories at their host absolute paths, mounts the kernel cache and publishes each child's RPC port on host loopback. Docker access is therefore needed by the parent user. The node manages the assigned virtual-device environment and worker cleanup. Use the same input-mode distinction as for A100.

## 6. Profiles and configuration fields

The daemon loads model-level CSV costs via `--latency-csv`. A permitted model/profile/batch must have a measured or interpolable curve; resources absent from the profile data should not be enabled. The shipped CSV is a starting dataset, not a calibration of every new hardware/software installation.

The Huawei template names and historical CSV names differ. To use the retained name mapping, create a separate copy on the scheduler host; the original measured file stays unchanged:

```sh
mkdir -p deployment
python3 - <<'PY'
import csv
from pathlib import Path
source = Path('logs/model_latency.csv')
target = Path('deployment/latency-hardware.csv')
aliases = {'vir05_1c_8g': 'vir05_1c_16g', 'vir10_3c_16g': 'vir10_3c_32g'}
with source.open(newline='') as inp, target.open('w', newline='') as out:
    reader = csv.DictReader(inp)
    writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
    writer.writeheader()
    for row in reader:
        row['split_param'] = aliases.get(row['split_param'], row['split_param'])
        writer.writerow(row)
PY
```

For an A100-only service, the original `logs/model_latency.csv` can be used directly. The alias transformation changes profile labels only, following the retained measurement mapping; it does not convert hardware performance between devices.

| Field | Interpretation |
|---|---|
| `device_combinations.NAME.type` | `Nvidia` or `Huawei` real-device adapter |
| `config.resource_template` | Budget per parent device, not total cluster capacity |
| `config.share_targets` | Permitted slice profiles |
| `remote_nodes.NAME.ip/port` | Address of a started `nv_virtual` / `hw_virtual` process |
| `remote_nodes.NAME.template_name` | Selects a compatible device template and shared pool |
| `init_arrival_rate` | Initial model demand in samples/s |
| `batch_size_range` | Worker execution batch range |
| `batch_extractor_strategy` | Queue extraction/timeout policy |
| sender `models[].rate_range` | Offered samples/s, with request size in `batch_size_range` |

The historical A100 preset is [`4nv+4nv`](../scripts/exper_config/4nv+4nv/scheduler.toml). It defines budgets `ce=7,memory=80` and `ce=3,memory=40`; its active example has one full-budget and two half-budget node entries. The Huawei [`8hw`](../scripts/exper_config/8hw/scheduler.toml) preset defines `ai_core=20,ai_cpu=7,memory=64` and two virtual-device profiles; its active example has one node entry. Folder names are historical labels, not an automatically instantiated card count. Their sender files use random-walk rates; the A100 sender gives VGG16/19 approximately twice the demand of the other models.

These templates describe allocator budgets. They do not create a physical full/half-card cluster by themselves, and they are not a recipe for nested MIG. Use real, disjoint devices/supported instances and verify the node-advertised resources. Reproducing the paper's eight-A100 or 24/32-NPU measurements additionally requires the retained physical-device mapping, workloads, baselines and software environment listed in the paper index.

## 7. Start the daemon and send requests

On the scheduler host, no CUDA/CANN worker runtime is needed for the following scheduler build:

```sh
cargo build --locked -p scheduler \
  --features scheduler_daemon,task_sender,nvidia,huawei \
  --bin scheduler_daemon --bin task_sender
```

For the one-model example, use `scripts/exper_config/hardware/a100-single.toml` or `huawei-single.toml`. Replace the node address when nodes run on a different machine. Keep model names synchronized across JSON files, CSV curves, initial demand and sender configuration.

```sh
target/debug/scheduler_daemon \
  --config scripts/exper_config/hardware/a100-single.toml \
  --latency-csv logs/model_latency.csv \
  --host 127.0.0.1 --port 3000 --interval 15 \
  --log-dir deployment/scheduler
```

For Huawei, select `huawei-single.toml` and `deployment/latency-hardware.csv`. For both vendors together, combine their device templates and node entries, assigning unique names and consistent model/profile coverage. Expose the HTTP listener on a trusted reachable interface if the sender is remote; loopback is suitable only on the same host.

In another shell, check service readiness and start a finite synthetic load:

```sh
curl --fail http://127.0.0.1:3000/health
curl --fail http://127.0.0.1:3000/scheduler/stats
target/debug/task_sender \
  --config scripts/exper_config/hardware/task_sender.toml \
  --daemon-url http://127.0.0.1:3000 \
  --duration-secs 30 --drain-timeout-secs 300
```

Check node/model-loading logs as well as HTTP availability; `/health` alone does not establish that models are serving. Inspect sender progress/final totals and daemon lifecycle/worker logs. Stop the sender first, allow outstanding work to finish, then stop the daemon and node parents with their normal interrupt handling. Inspect owned slice workers/containers if cleanup reports a failure.

## 8. Verification scope

Commands, feature names, paths and configuration fields were checked against the released source. The single-device examples were statically validated; they have not been exercised on A100/Ascend hardware in this documentation update. New measurements must record their own source, model, device and runtime identities. Historical hardware result data, exact baseline recipes and the separate operator-level profiler are not supplied by these service examples.
