//! Compute Node implementation for Huawei Ascend Device

#[cfg(all(feature = "onnx", feature = "vgpu"))]
use std::{env, process::Stdio, time::Duration};

use clap::ValueEnum;
#[cfg(all(feature = "onnx", feature = "vgpu"))]
use tokio::process::Command;
#[cfg(all(feature = "vgpu", feature = "onnx"))]
use worker::onnx::worker::OnnxWorker;

#[cfg(all(feature = "vgpu", feature = "onnx"))]
use crate::compute_device::hw::HuaweiGpu;
#[cfg(all(feature = "vgpu", feature = "onnx"))]
use crate::compute_device::{hw::HuaweiVirtualGpu, vgpu::FullDeviceShareWrapper};
#[cfg(all(feature = "vgpu", feature = "onnx"))]
use crate::util::worker_proxy::bin_simple_worker_proxy::BinSimpleWorkerProxyImpl;
#[cfg(all(feature = "vgpu", feature = "onnx"))]
use crate::util::worker_proxy::bin_simple_worker_proxy::BinSimpleWorkerProxyImplInit;
#[cfg(all(feature = "vgpu", feature = "onnx"))]
use crate::util::worker_proxy::bin_simple_worker_proxy::WorkerProxyCapable;
#[cfg(all(feature = "vgpu", feature = "onnx"))]
use crate::util::worker_proxy::bin_simple_worker_proxy::WorkerProxyType;

/// [HuaweiGpu]支持的Worker类型
#[derive(Clone, ValueEnum)]
pub enum WorkerType {
    #[cfg(feature = "onnx")]
    /// 使用 Onnx Backend 执行模型推理
    Onnx,
}

#[cfg(all(feature = "vgpu", feature = "onnx"))]
impl WorkerProxyCapable<FullDeviceShareWrapper<HuaweiGpu>> for OnnxWorker {}

/// Huawei Ascend Device Worker Proxy
#[cfg(all(feature = "vgpu", feature = "onnx"))]
pub type HwGpuWorkerProxy = BinSimpleWorkerProxyImpl<FullDeviceShareWrapper<HuaweiGpu>>;

#[cfg(all(feature = "vgpu", feature = "onnx"))]
impl BinSimpleWorkerProxyImplInit<HuaweiGpu> for HwGpuWorkerProxy {
    const PROXY_TYPE: WorkerProxyType = WorkerProxyType::Hw;

    async fn new<W: WorkerProxyCapable<HuaweiGpu>>(
        id: uuid::Uuid,
        device: HuaweiGpu,
        model_dir: impl AsRef<std::path::Path> + Send,
        model_config_dir: impl AsRef<std::path::Path> + Send,
        bin_worker_path: impl AsRef<std::path::Path> + Send,
    ) -> Result<Self, String> {
        let device = FullDeviceShareWrapper::from(device);
        let on_clean_up = Box::new(|| {
            Box::into_pin(Box::new(async move { Ok(()) })
                as Box<dyn Future<Output = Result<(), String>> + Send>)
        });
        Self::new_inner::<OnnxWorker>(
            id,
            device,
            model_dir,
            model_config_dir,
            bin_worker_path,
            on_clean_up,
        )
        .await
    }
}

#[cfg(all(feature = "onnx", feature = "vgpu"))]
impl WorkerProxyCapable<HuaweiVirtualGpu> for OnnxWorker {}

/// Huawei VGpu 工作代理
///
/// By default, the worker will be initialized with the following environment variables:
/// - `ASCEND_RUNTIME_OPTIONS=VIRTUAL`
/// - `ASCEND_VISIBLE_DEVICES={vchip_id}`
///
/// When creating docker, worker will read docker image and timeout docker init time from env,
/// you can use:
/// - `HW_VGPU_DOCKER_IMAGE` to specify the docker image name, default is "hsgs-cann-env"
/// - `HW_VGPU_STARTUP_TIMEOUT_SECS` to specify the timeout in seconds, default is 10
#[cfg(all(feature = "onnx", feature = "vgpu"))]
pub type HwVirtualGpuWorkerProxy = BinSimpleWorkerProxyImpl<HuaweiVirtualGpu>;

#[cfg(all(feature = "onnx", feature = "vgpu"))]
impl BinSimpleWorkerProxyImplInit<HuaweiVirtualGpu> for HwVirtualGpuWorkerProxy {
    const PROXY_TYPE: WorkerProxyType = WorkerProxyType::HwMig;

    async fn new<W: WorkerProxyCapable<HuaweiVirtualGpu>>(
        id: uuid::Uuid,
        device: HuaweiVirtualGpu,
        model_dir: impl AsRef<std::path::Path> + Send,
        model_config_dir: impl AsRef<std::path::Path> + Send,
        bin_worker_path: impl AsRef<std::path::Path> + Send,
    ) -> Result<Self, String> {
        // init via docker, require docker image named "hsgs-cann-env"
        //
        // As mentioned in [Huawei documentation](https://www.hiascend.com/document/detail/zh/mindx-dl/600/clusterscheduling/clusterscheduling/cpaug/cpaug_0016.html),
        // we need use "ASCEND_RUNTIME_OPTIONS=VIRTUAL" and "ASCEND_VISIBLE_DEVICES={vchip_id}" to initialize the device.

        let bin_worker_path_str = bin_worker_path.as_ref().to_str().unwrap().to_string();
        let model_dir_str = model_dir.as_ref().to_str().unwrap().to_string();
        let model_config_dir_str = model_config_dir.as_ref().to_str().unwrap().to_string();

        let (process_args, port) = Self::get_process_args(
            &bin_worker_path_str,
            id,
            0,
            &model_dir_str,
            &model_config_dir_str,
            Self::get_worker_type::<W>().unwrap(),
        );

        let container_name = format!("hw_vgpu_{}", id);

        let args = vec![
            "run".into(),
            "--rm".into(),
            "-e".into(),
            format!("ASCEND_VISIBLE_DEVICES={}", device.vchip_id),
            "-e".into(),
            "ASCEND_RUNTIME_OPTIONS=VIRTUAL".to_string(),
            "-e".into(),
            "RUST_LOG=trace".into(),
            "-e".into(),
            "ASCEND_SLOG_PRINT_TO_STDOUT=1".into(),
            "-p".into(),
            format!("127.0.0.1::{port}"),
            "-v".into(),
            format!("{}:{}", bin_worker_path_str, bin_worker_path_str),
            "-w".into(),
            model_dir_str.clone(),
            "-v".into(),
            format!("{}:{}", model_dir_str, model_dir_str),
            "-v".into(),
            format!("{}:{}", model_config_dir_str, model_config_dir_str),
            "-v".into(),
            format!(
                "{}:/root/atc_data",
                env::var("HW_VGPU_KERNEL_CACHE_PATH").unwrap_or("/root/atc_data".to_string())
            ),
            "--name".into(),
            container_name.clone(),
        ];

        let docker_image = env::var("HW_VGPU_DOCKER_IMAGE").unwrap_or("hsgs-cann-env".to_string());

        let mut _process = Command::new("docker")
            .args(&args)
            .args(&[docker_image])
            .args(&process_args)
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| format!("Failed to start Docker: {}", e))?;

        // 等待容器启动
        let startup_timeout_secs: u64 = env::var("HW_VGPU_STARTUP_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(20);
        let startup_started = tokio::time::Instant::now();
        // Docker atomically allocates and owns the published host port.
        let published_port =
            tokio::time::timeout(Duration::from_secs(startup_timeout_secs), async {
                loop {
                    if let Some(status) = _process.try_wait().map_err(|e| e.to_string())? {
                        return Err(format!(
                            "Docker worker exited before publishing its port: {status}"
                        ));
                    }
                    if let Ok(Ok(output)) = tokio::time::timeout(
                        Duration::from_secs(1),
                        Command::new("docker")
                            .args(["port", &container_name, &format!("{port}/tcp")])
                            .kill_on_drop(true)
                            .output(),
                    )
                    .await
                        && output.status.success()
                        && let Some(port) = String::from_utf8_lossy(&output.stdout)
                            .trim()
                            .rsplit(':')
                            .next()
                            .and_then(|port| port.parse::<u16>().ok())
                    {
                        return Ok(port);
                    }
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            })
            .await
            .map_err(|_| "Docker port publication timed out".to_owned())
            .and_then(|result| result);

        let on_clean_up = Box::new(|| {
            Box::into_pin(Box::new(async move {
                let output = Command::new("docker")
                    .arg("stop")
                    .arg(&container_name)
                    .output()
                    .await
                    .map_err(|e| format!("Failed to stop Docker container: {}", e))?;
                if output.status.success() {
                    Ok(())
                } else {
                    Err(format!(
                        "Failed to stop Docker container, status: {}, stderr: {}",
                        output.status,
                        String::from_utf8_lossy(&output.stderr)
                    ))
                }
            })
                as Box<dyn Future<Output = Result<(), String>> + Send>)
        });

        let port = match published_port {
            Ok(port) => port,
            Err(error) => {
                let proxy = Self::new(0, device, None, _process, on_clean_up);
                let cleanup = crate::util::worker_proxy::WorkerProxy::destroy(proxy).await;
                return Err(format!("{error}; startup cleanup: {cleanup:?}"));
            }
        };
        let mut proxy = BinSimpleWorkerProxyImpl::new(port, device, None, _process, on_clean_up);
        if let Err(error) = proxy
            .wait_ready_with_timeout(
                Duration::from_secs(startup_timeout_secs).saturating_sub(startup_started.elapsed()),
            )
            .await
        {
            let cleanup = crate::util::worker_proxy::WorkerProxy::destroy(proxy).await;
            return Err(format!("{error}; startup cleanup: {cleanup:?}"));
        }
        Ok(proxy)
    }
}
