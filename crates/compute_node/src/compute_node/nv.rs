//! Compute Node implementation for Nvidia GPU

#[cfg(all(any(feature = "onnx", feature = "torch"), feature = "vgpu"))]
use std::marker::PhantomData;
use std::path::Path;

use clap::ValueEnum;
use uuid::Uuid;
#[cfg(all(feature = "onnx", feature = "vgpu"))]
use worker::onnx::worker::OnnxWorker;
#[cfg(all(feature = "torch", feature = "vgpu"))]
use worker::torch::worker::TorchWorker;

#[cfg(all(any(feature = "onnx", feature = "torch"), feature = "vgpu"))]
use crate::compute_device::nv::{NvidiaMpsVirtualGpu, vgpu::NvidiaMigVirtualGpu};
#[cfg(all(any(feature = "onnx", feature = "torch"), feature = "vgpu"))]
use crate::util::worker_proxy::bin_simple_worker_proxy::BinSimpleWorkerProxyImpl;
#[cfg(all(any(feature = "onnx", feature = "torch"), feature = "vgpu"))]
use crate::util::worker_proxy::bin_simple_worker_proxy::BinSimpleWorkerProxyImplInit;
#[cfg(all(any(feature = "onnx", feature = "torch"), feature = "vgpu"))]
use crate::util::worker_proxy::bin_simple_worker_proxy::WorkerProxyCapable;
#[cfg(all(any(feature = "onnx", feature = "torch"), feature = "vgpu"))]
use crate::util::worker_proxy::bin_simple_worker_proxy::WorkerProxyType;

/// [NvidiaGpu](crate::compute_device::nv::gpu::NvidiaGpu)支持的Worker类型
#[derive(Clone, ValueEnum)]
pub enum WorkerType {
    /// 使用 Pytorch C++ API 执行模型推理
    #[cfg(feature = "torch")]
    Torch,
    /// 使用 Onnx Backend 执行模型推理
    #[cfg(feature = "onnx")]
    Onnx,
}

#[cfg(all(feature = "onnx", feature = "vgpu"))]
impl WorkerProxyCapable<NvidiaMpsVirtualGpu> for OnnxWorker {}
#[cfg(all(feature = "torch", feature = "vgpu"))]
impl WorkerProxyCapable<NvidiaMpsVirtualGpu> for TorchWorker {}

/// Nvidia Mps Virtual GPU Worker Proxy
#[cfg(all(any(feature = "onnx", feature = "torch"), feature = "vgpu"))]
pub type NvMpsVirtualGpuWorkerProxy = BinSimpleWorkerProxyImpl<NvidiaMpsVirtualGpu>;

#[cfg(all(any(feature = "onnx", feature = "torch"), feature = "vgpu"))]
impl BinSimpleWorkerProxyImplInit<NvidiaMpsVirtualGpu> for NvMpsVirtualGpuWorkerProxy {
    const PROXY_TYPE: WorkerProxyType = WorkerProxyType::NvMps;

    async fn new<W: WorkerProxyCapable<NvidiaMpsVirtualGpu>>(
        id: Uuid,
        device: NvidiaMpsVirtualGpu,
        model_dir: impl AsRef<Path> + Send,
        model_config_dir: impl AsRef<Path> + Send,
        bin_worker_path: impl AsRef<Path> + Send,
    ) -> Result<Self, String> {
        let percentage = device.percentage.to_string();
        let on_clean_up = Box::new(|| {
            Box::into_pin(Box::new(async move { Ok(()) })
                as Box<dyn Future<Output = Result<(), String>> + Send>)
        });
        Self::new_with_process_handler(
            id,
            device,
            model_dir,
            model_config_dir,
            bin_worker_path,
            PhantomData::<W>,
            |mut process| async move {
                let process = process
                    .env("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", percentage)
                    .spawn()
                    .map_err(|e| format!("Failed to spawn worker process: {e}"))?;

                Ok(process)
            },
            on_clean_up,
        )
        .await
    }
}

#[cfg(all(feature = "onnx", feature = "vgpu"))]
impl WorkerProxyCapable<NvidiaMigVirtualGpu> for OnnxWorker {}
#[cfg(all(feature = "torch", feature = "vgpu"))]
impl WorkerProxyCapable<NvidiaMigVirtualGpu> for TorchWorker {}

/// Nvidia Mig Virtual GPU Worker Proxy
#[cfg(all(any(feature = "onnx", feature = "torch"), feature = "vgpu"))]
pub type NvMigVirtualGpuWorkerProxy = BinSimpleWorkerProxyImpl<NvidiaMigVirtualGpu>;

#[cfg(all(any(feature = "onnx", feature = "torch"), feature = "vgpu"))]
impl BinSimpleWorkerProxyImplInit<NvidiaMigVirtualGpu> for NvMigVirtualGpuWorkerProxy {
    const PROXY_TYPE: WorkerProxyType = WorkerProxyType::NvMig;

    async fn new<W: WorkerProxyCapable<NvidiaMigVirtualGpu>>(
        id: Uuid,
        device: NvidiaMigVirtualGpu,
        model_dir: impl AsRef<Path> + Send,
        model_config_dir: impl AsRef<Path> + Send,
        bin_worker_path: impl AsRef<Path> + Send,
    ) -> Result<Self, String> {
        let uuid = device.uuid.clone();
        let on_clean_up = Box::new(|| {
            Box::into_pin(
                Box::new(async { Ok(()) }) as Box<dyn Future<Output = Result<(), String>> + Send>
            )
        });
        Self::new_with_process_handler(
            id,
            device,
            model_dir,
            model_config_dir,
            bin_worker_path,
            PhantomData::<W>,
            |mut process| async move {
                let process = process
                    .env("CUDA_VISIBLE_DEVICES", uuid)
                    .spawn()
                    .map_err(|e| format!("Failed to spawn worker process: {e}"))?;

                Ok(process)
            },
            on_clean_up,
        )
        .await
    }
}
