//! A universal binary worker proxy implementation for different compute devices.

use std::{fmt::Debug, marker::PhantomData, path::Path, pin::Pin, process::Stdio, time::Duration};

use bincode::{Decode, Encode};
use comm::rpc_pool::RpcPool;
use serde::{Deserialize, Serialize};
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, BufReader},
    process::{Child, Command},
    time::timeout,
};
use uuid::Uuid;
#[cfg(feature = "mock")]
use worker::mock::MockWorker;
#[cfg(feature = "onnx")]
use worker::onnx::worker::OnnxWorker;
#[cfg(feature = "torch")]
use worker::torch::worker::TorchWorker;
use worker::{Worker, model::tensor::DynTensor};

use crate::{
    compute_device::ComputeDevice,
    util::{
        simple_compute_node::{
            BinSimpleComputeNodeRequest, BinSimpleComputeNodeResponse, SimpleComputeNodeCommand,
        },
        worker_proxy::WorkerProxy,
    },
};

#[derive(Serialize, Deserialize)]
struct WorkerAddress {
    id: Uuid,
    port: u16,
}

/// One startup message on stdout; worker logs belong on stderr.
pub fn announce_worker_address(id: Uuid, port: u16) -> std::io::Result<()> {
    use std::io::Write;
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(&mut stdout, &WorkerAddress { id, port })?;
    writeln!(stdout)?;
    stdout.flush()
}

pub(crate) async fn bound_worker_port(process: &mut Child, id: Uuid) -> Result<u16, String> {
    let stdout = process
        .stdout
        .take()
        .ok_or("worker startup pipe is missing")?;
    let mut startup = BufReader::new(stdout.take(4096));
    let mut bytes = Vec::new();
    timeout(Duration::from_secs(10), async {
        tokio::select! {
            read = startup.read_until(b'\n', &mut bytes) => {
                read.map_err(|error| error.to_string())?;
                if bytes.last() != Some(&b'\n') {
                    return Err("worker startup pipe closed or message exceeded 4096 bytes".into());
                }
                let address: WorkerAddress = serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
                if address.id != id || address.port == 0 {
                    return Err("invalid worker startup address".into());
                }
                Ok(address.port)
            }
            status = process.wait() => Err(format!("worker exited before registration: {}", status.map_err(|error| error.to_string())?)),
        }
    }).await.map_err(|_| "worker registration timed out".to_owned())?
}

/// Cleanup function type
pub type CleanupFn =
    Box<dyn FnOnce() -> Pin<Box<dyn Future<Output = Result<(), String>> + Send>> + Send + Sync>;

/// Convenience [BinSimpleWorkerProxyImpl] Impl
pub struct BinSimpleWorkerProxyImpl<T> {
    /// Worker 工作进程监听端口
    port: u16,
    connections: RpcPool<BinSimpleComputeNodeRequest, BinSimpleComputeNodeResponse>,
    /// 虚拟GPU
    _device: T,
    /// 加载的模型
    loaded_model: Option<String>,
    /// 工作进程
    _process: Child,
    on_clean_up: CleanupFn,
    inference_timeout: Duration,
    control_timeout: Duration,
}

impl<T: Debug> Debug for BinSimpleWorkerProxyImpl<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerProxyImpl")
            .field("port", &self.port)
            .field("_device", &self._device)
            .field("loaded_model", &self.loaded_model)
            .field("_process", &self._process)
            .finish()
    }
}

/// A [Worker] that can be used with a [BinSimpleWorkerProxyImpl] in a specific [ComputeDevice].
pub trait WorkerProxyCapable<CD: ComputeDevice>: Worker + 'static {}

impl<CD: ComputeDevice + Send + Sync + 'static> BinSimpleWorkerProxyImpl<CD> {
    /// Exclusive request/response exchange on a reusable connection.
    async fn send(
        &self,
        request: BinSimpleComputeNodeRequest,
    ) -> Result<BinSimpleComputeNodeResponse, String> {
        let budget = if matches!(request, BinSimpleComputeNodeRequest::Compute(_)) {
            self.inference_timeout
        } else {
            self.control_timeout
        };
        self.connections
            .request(
                &request,
                Duration::from_secs(10),
                Duration::from_secs(60),
                budget,
            )
            .await
    }

    /// Wait for a protocol response, including the child's OS startup time.
    pub(crate) async fn wait_ready(&mut self) -> Result<(), String> {
        self.wait_ready_with_timeout(Duration::from_secs(10)).await
    }

    pub(crate) async fn wait_ready_with_timeout(&mut self, budget: Duration) -> Result<(), String> {
        timeout(budget, async {
            loop {
                if let Some(status) = self._process.try_wait().map_err(|e| e.to_string())? {
                    return Err(format!("worker exited before readiness: {status}"));
                }
                if matches!(
                    self.send(BinSimpleComputeNodeRequest::Status).await,
                    Ok(BinSimpleComputeNodeResponse::Status(_))
                ) {
                    return Ok(());
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .map_err(|_| "worker readiness deadline exceeded".to_string())?
    }

    pub(crate) async fn finish_startup(mut self, id: Uuid) -> Result<Self, String> {
        let result = async {
            self.port = bound_worker_port(&mut self._process, id).await?;
            self.connections = RpcPool::new(format!("127.0.0.1:{}", self.port));
            self.wait_ready().await
        }
        .await;
        if let Err(error) = result {
            let cleanup = self.destroy().await;
            return Err(format!("{error}; startup cleanup: {cleanup:?}"));
        }
        Ok(self)
    }

    pub(crate) fn with_inference_timeout(mut self, timeout: Duration) -> Self {
        self.inference_timeout = timeout;
        self
    }

    pub(crate) fn with_control_timeout(mut self, timeout: Duration) -> Self {
        self.control_timeout = timeout;
        self
    }

    pub(crate) fn new(
        port: u16,
        _device: CD,
        loaded_model: Option<String>,
        _process: Child,
        on_clean_up: CleanupFn,
    ) -> Self {
        Self {
            port,
            connections: RpcPool::new(format!("127.0.0.1:{port}")),
            _device,
            loaded_model,
            _process,
            on_clean_up,
            inference_timeout: Duration::from_secs(60),
            control_timeout: Duration::from_secs(60),
        }
    }

    pub(crate) fn get_worker_type<W: WorkerProxyCapable<CD>>() -> Option<&'static str> {
        match std::any::TypeId::of::<W>() {
            #[cfg(feature = "onnx")]
            type_id if type_id == std::any::TypeId::of::<OnnxWorker>() => Some("onnx"),
            #[cfg(feature = "torch")]
            type_id if type_id == std::any::TypeId::of::<TorchWorker>() => Some("torch"),
            #[cfg(feature = "mock")]
            type_id if type_id == std::any::TypeId::of::<MockWorker>() => Some("mock"),
            _ => None,
        }
    }

    pub(crate) fn get_process_args(
        bin_worker_path_str: &str,
        id: Uuid,
        device_id: u32,
        model_dir_str: &str,
        model_config_dir_str: &str,
        worker_type: &str,
    ) -> (Vec<String>, u16) {
        let port = 8080; // Private container namespace; Docker owns the host port.

        let args = vec![
            bin_worker_path_str.to_string(),
            "--id".to_string(),
            id.to_string(),
            "--device-id".to_string(),
            device_id.to_string(),
            "--model-dir".to_string(),
            model_dir_str.to_string(),
            "--model-config".to_string(),
            model_config_dir_str.to_string(),
            "--port".to_string(),
            port.to_string(),
            "--worker-type".to_string(),
            worker_type.to_string(),
        ];
        (args, port)
    }

    fn get_process(
        bin_worker_path: impl AsRef<Path> + Send,
        id: Uuid,
        device_id: u32,
        model_dir: impl AsRef<Path> + Send,
        model_config_dir: impl AsRef<Path> + Send,
        worker_type: &str,
    ) -> Command {
        let mut command = Command::new(bin_worker_path.as_ref().to_str().unwrap());
        command
            .args([
                "--id",
                id.to_string().as_str(),
                "--device-id",
                device_id.to_string().as_str(),
                "--model-dir",
                model_dir.as_ref().to_str().unwrap(),
                "--model-config",
                model_config_dir.as_ref().to_str().unwrap(),
                "--port",
                "0",
                "--worker-type",
                worker_type,
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);
        command
    }

    #[cfg(all(feature = "vgpu", feature = "onnx"))]
    pub(crate) async fn new_inner<W: WorkerProxyCapable<CD>>(
        id: Uuid,
        device: CD,
        model_dir: impl AsRef<Path> + Send,
        model_config_dir: impl AsRef<Path> + Send,
        bin_worker_path: impl AsRef<Path> + Send,
        on_clean_up: CleanupFn,
    ) -> Result<BinSimpleWorkerProxyImpl<CD>, String> {
        let worker_type =
            Self::get_worker_type::<W>().ok_or("Unsupported worker type".to_string())?;
        let mut _process = Self::get_process(
            bin_worker_path,
            id,
            device.logic_index().unwrap(),
            model_dir,
            model_config_dir,
            worker_type,
        );
        let _process = _process
            .spawn()
            .map_err(|e| format!("Failed to spawn worker process: {e}"))?;
        Self::new(0, device, None, _process, on_clean_up)
            .finish_startup(id)
            .await
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn new_with_process_handler<
        W: WorkerProxyCapable<CD>,
        F: FnOnce(Command) -> Fut,
        Fut: Future<Output = Result<Child, String>>,
    >(
        id: Uuid,
        device: CD,
        model_dir: impl AsRef<Path> + Send,
        model_config_dir: impl AsRef<Path> + Send,
        bin_worker_path: impl AsRef<Path> + Send,
        _worker: PhantomData<W>,
        process_handler: F,
        on_clean_up: CleanupFn,
    ) -> Result<BinSimpleWorkerProxyImpl<CD>, String> {
        let worker_type =
            Self::get_worker_type::<W>().ok_or("Unsupported worker type".to_string())?;
        let _process = Self::get_process(
            bin_worker_path,
            id,
            device.logic_index().unwrap(),
            model_dir,
            model_config_dir,
            worker_type,
        );
        let _process = process_handler(_process)
            .await
            .map_err(|e| format!("Failed to spawn worker process: {e}"))?;
        Self::new(0, device, None, _process, on_clean_up)
            .finish_startup(id)
            .await
    }
}

impl<CD: ComputeDevice + Send + Sync + 'static> WorkerProxy for BinSimpleWorkerProxyImpl<CD> {
    type Request = BinSimpleComputeNodeRequest;
    type Response = BinSimpleComputeNodeResponse;

    fn loaded_model(&self) -> Option<&String> {
        self.loaded_model.as_ref()
    }

    async fn load_model(&mut self, model_name: String) -> Result<(), String> {
        let response = self
            .send(BinSimpleComputeNodeRequest::Command(
                SimpleComputeNodeCommand::Load(model_name.clone()),
            ))
            .await?;

        match response {
            BinSimpleComputeNodeResponse::State(Some(curr_model_name)) => {
                if model_name == curr_model_name {
                    self.loaded_model.replace(model_name);
                    Ok(())
                } else {
                    Err("Model name mismatch".to_string())
                }
            }
            BinSimpleComputeNodeResponse::Status(_) => {
                Err("Unexpected status response".to_string())
            }
            BinSimpleComputeNodeResponse::Error(e) => Err(e),
            BinSimpleComputeNodeResponse::Compute(_) => {
                Err("Unexpected compute response".to_string())
            }
            BinSimpleComputeNodeResponse::State(None) => Err("Model did not load".to_string()),
        }
    }

    async fn unload_model(&mut self) -> Result<(), String> {
        let response = self
            .send(BinSimpleComputeNodeRequest::Command(
                SimpleComputeNodeCommand::UnLoad,
            ))
            .await?;

        match response {
            BinSimpleComputeNodeResponse::State(None) => {
                self.loaded_model.take();
                Ok(())
            }
            BinSimpleComputeNodeResponse::Status(_) => {
                Err("Unexpected status response".to_string())
            }
            BinSimpleComputeNodeResponse::Error(e) => Err(e),
            BinSimpleComputeNodeResponse::Compute(_) => {
                Err("Unexpected compute response".to_string())
            }
            BinSimpleComputeNodeResponse::State(Some(_)) => Err("Unload model failed".to_string()),
        }
    }

    async fn compute(&self, input: DynTensor) -> Result<DynTensor, String> {
        let response = self
            .send(BinSimpleComputeNodeRequest::Compute(input.into()))
            .await?;

        match response {
            BinSimpleComputeNodeResponse::Compute(output) => Ok(output.into()),
            BinSimpleComputeNodeResponse::State(_) => Err("Unexpected state response".to_string()),
            BinSimpleComputeNodeResponse::Status(_) => {
                Err("Unexpected status response".to_string())
            }
            BinSimpleComputeNodeResponse::Error(e) => Err(e),
        }
    }

    #[tracing::instrument(level = "trace")]
    async fn destroy(mut self) -> Result<(), String> {
        let result: Result<(), String> = async {
            if self
                ._process
                .try_wait()
                .map_err(|e| e.to_string())?
                .is_none()
            {
                self._process
                    .kill()
                    .await
                    .map_err(|e| format!("Failed to kill worker process: {e}"))?;
            }
            self._process
                .wait()
                .await
                .map_err(|e| format!("Failed to reap worker process: {e}"))?;
            (self.on_clean_up)()
                .await
                .map_err(|e| format!("Failed to clean up worker process: {e}"))?;
            Ok(())
        }
        .await;
        if let Err(error) = result {
            // Consuming destroy cannot hand ownership back to its caller. Retain
            // the lease on failure; a node restart/operator recovery is required.
            std::mem::forget(self._device);
            return Err(error);
        }
        // Hardware device Drop may call a blocking vendor CLI. Keep ownership
        // until it finishes without blocking a Tokio executor thread.
        tokio::task::spawn_blocking(move || self._device.release())
            .await
            .map_err(|e| format!("device release task failed: {e}"))??;
        Ok(())
    }
}

/// A convenience trait for initializing a [BinSimpleWorkerProxyImpl] implementation.
pub trait BinSimpleWorkerProxyImplInit<CD: ComputeDevice>: Sized {
    /// The worker type of the [BinSimpleWorkerProxyImpl] implementation on.
    const PROXY_TYPE: WorkerProxyType;

    /// Creates a new [BinSimpleWorkerProxyImpl] instance.
    fn new<W: WorkerProxyCapable<CD>>(
        id: Uuid,
        device: CD,
        model_dir: impl AsRef<Path> + Send,
        model_config_dir: impl AsRef<Path> + Send,
        bin_worker_path: impl AsRef<Path> + Send,
    ) -> impl Future<Output = Result<Self, String>> + Send;
}

#[derive(Debug, Serialize, Deserialize, Encode, Decode)]
#[non_exhaustive]
/// [BinSimpleWorkerProxyImpl] Type
pub enum WorkerProxyType {
    /// [BinSimpleWorkerProxyImpl] using Nvidia GPU and Nvidia MPS
    NvMps,
    /// [BinSimpleWorkerProxyImpl] using Nvidia GPU and Nvidia MIG
    NvMig,
    /// [BinSimpleWorkerProxyImpl] using Huawei NPU alone
    Hw,
    /// [BinSimpleWorkerProxyImpl] using Huawei NPU and MIG like resource partition
    HwMig,
    /// [BinSimpleWorkerProxyImpl] using Mock device with Nvidia MIG like resource sharing
    Mock,
}

#[cfg(all(test, feature = "mock", feature = "vgpu", unix))]
mod cleanup_tests {
    use super::*;
    use crate::compute_device::{
        mock::{MockComputeDevice, nv::MockNvidiaMigShareableGpu},
        vgpu::{
            ShareableComputeDevice,
            resource_spec::nv::mig::{A100Template, NvidiaMigResource, VDeviceScale},
        },
    };

    #[tokio::test]
    async fn startup_pipe_validates_identity_port_and_message_length() {
        let id = Uuid::new_v4();
        for (message, expected) in [
            (
                serde_json::to_string(&WorkerAddress { id, port: 12345 }).unwrap(),
                Some(12345),
            ),
            (
                serde_json::to_string(&WorkerAddress {
                    id: Uuid::new_v4(),
                    port: 12345,
                })
                .unwrap(),
                None,
            ),
            (
                serde_json::to_string(&WorkerAddress { id, port: 0 }).unwrap(),
                None,
            ),
            ("x".repeat(4096), None),
        ] {
            let mut child = Command::new("sh")
                .args([
                    "-c",
                    "printf '%s\\n' \"$1\"; exec sleep 30",
                    "startup-test",
                    &message,
                ])
                .stdout(Stdio::piped())
                .kill_on_drop(true)
                .spawn()
                .unwrap();
            assert_eq!(bound_worker_port(&mut child, id).await.ok(), expected);
            child.kill().await.unwrap();
            child.wait().await.unwrap();
        }
    }

    #[tokio::test]
    async fn explicit_device_release_retains_capacity_on_failure() {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };

        use crate::compute_device::vgpu::{ShareableResource, SharedResource};
        #[derive(Debug)]
        struct Device {
            lease: Option<SharedResource<NvidiaMigResource>>,
            attempts: Arc<AtomicUsize>,
            fail: bool,
        }
        impl ComputeDevice for Device {
            type Request = ();
            type Response = ();
            async fn request(&self, _: ()) {}
            fn logic_index(&self) -> Option<u32> {
                None
            }
            fn release(mut self) -> Result<(), String> {
                if self.lease.is_some() {
                    self.attempts.fetch_add(1, Ordering::Relaxed);
                    if self.fail {
                        // The consuming release owns failure quarantine; Drop
                        // must not refund an unconfirmed physical destruction.
                        std::mem::forget(self.lease.take().unwrap());
                        return Err("physical destruction failed".into());
                    }
                    self.lease.take();
                }
                Ok(())
            }
        }
        impl Drop for Device {
            fn drop(&mut self) {
                assert!(
                    self.lease.is_none(),
                    "unconfirmed destruction must not drop the lease"
                );
            }
        }
        for fail in [false, true] {
            let capacity = NvidiaMigResource { ce: 1, memory: 10 };
            let resource = ShareableResource::from(capacity);
            let attempts = Arc::new(AtomicUsize::new(0));
            let device = Device {
                lease: resource.share(capacity),
                attempts: attempts.clone(),
                fail,
            };
            let process = Command::new("sleep")
                .arg("30")
                .kill_on_drop(true)
                .spawn()
                .unwrap();
            let proxy = BinSimpleWorkerProxyImpl::new(
                0,
                device,
                None,
                process,
                Box::new(|| Box::pin(async { Ok(()) })),
            );
            assert_eq!(proxy.destroy().await.is_err(), fail);
            assert_eq!(attempts.load(Ordering::Relaxed), 1);
            assert_eq!(resource.current_resource().ce, if fail { 0 } else { 1 });
        }
    }

    #[tokio::test]
    async fn failed_cleanup_does_not_refund_capacity() {
        let device = MockNvidiaMigShareableGpu::with_capacity(
            MockComputeDevice::new("cleanup-failure-test".into()),
            NvidiaMigResource { ce: 1, memory: 10 },
        );
        let target = VDeviceScale::A100(A100Template::Tiny);
        let slice = device.share(target).unwrap().unwrap();
        let process = Command::new("sleep")
            .arg("30")
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let cleanup: CleanupFn =
            Box::new(|| Box::pin(async { Err("intentional cleanup failure".into()) }));
        let proxy = BinSimpleWorkerProxyImpl::new(0, slice, None, process, cleanup);
        let error = proxy.destroy().await.unwrap_err();
        assert!(error.contains("intentional cleanup failure"));
        assert_eq!(device.remain_resource().ce, 0);
        assert!(device.share(target).is_none());
    }
}
