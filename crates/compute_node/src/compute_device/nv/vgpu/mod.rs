//! Nvidia VGpu 模块

use std::{ops::Deref, sync::Arc};

#[cfg(feature = "onnx")]
use worker::onnx::worker::OnnxWorker;
#[cfg(feature = "torch")]
use worker::torch::worker::TorchWorker;

#[cfg(feature = "onnx")]
use crate::compute_device::config::OnnxWorkerModelLoadConfig;
#[cfg(feature = "torch")]
use crate::compute_device::config::TorchWorkerModelLoadConfig;
use crate::compute_device::{
    ComputeDevice, VirtualGpu,
    nv::{gpu::NvidiaGpu, mig::destroy_mig_vdevice},
    vgpu::{
        SharedResource,
        resource_spec::nv::{mig::NvidiaMigResource, mps::NvidiaMpsResource},
    },
};
#[cfg(any(feature = "onnx", feature = "torch"))]
use crate::runnable_on::{IndependentModelLoadConfig, RunnableOn};

pub mod mig;
pub mod mps;

/// Nvidia VGpu
#[derive(Debug)]
pub struct NvidiaMpsVirtualGpu {
    pub(crate) host_device: Arc<NvidiaGpu>,
    pub(crate) percentage: u32,
    _resource: SharedResource<NvidiaMpsResource>,
}

impl VirtualGpu for NvidiaMpsVirtualGpu {
    type HostDevice = NvidiaGpu;
    type State = u32;
    type Error = String;

    fn host_device(&self) -> impl Deref<Target = Self::HostDevice> + Send + '_ {
        self.host_device.clone()
    }

    async fn state(&self) -> Self::State {
        self.percentage
    }

    fn logic_index(&self) -> Option<u32> {
        self.host_device.logic_index()
    }
}

#[cfg(feature = "torch")]
impl RunnableOn<NvidiaMpsVirtualGpu> for TorchWorker {
    type DeviceIndependentConfig = TorchWorkerModelLoadConfig;

    fn device_config_getter(
        device: &NvidiaMpsVirtualGpu,
    ) -> impl FnOnce() -> <Self::DeviceIndependentConfig as IndependentModelLoadConfig>::RequireDevice
    {
        || {
            worker::model::tensor::Device::Gpu(
                device
                    .host_device
                    .device
                    .index()
                    .expect("Failed to get device index") as usize,
            )
        }
    }
}

#[cfg(feature = "onnx")]
impl RunnableOn<NvidiaMpsVirtualGpu> for OnnxWorker {
    type DeviceIndependentConfig = OnnxWorkerModelLoadConfig;

    fn device_config_getter(
        device: &NvidiaMpsVirtualGpu,
    ) -> impl FnOnce() -> <Self::DeviceIndependentConfig as IndependentModelLoadConfig>::RequireDevice
    {
        || {
            vec![
                ort::execution_providers::CUDAExecutionProvider::default()
                    .with_device_id(
                        device
                            .host_device
                            .device
                            .index()
                            .expect("Failed to get device index") as i32,
                    )
                    .build()
                    .error_on_failure(),
            ]
        }
    }
}

/// Nvidia Mig Virtual GPU
#[derive(Debug)]
pub struct NvidiaMigVirtualGpu {
    pub(crate) host_device: Arc<NvidiaGpu>,
    pub(crate) scale: &'static str,
    pub(crate) gpu_instance_id: u32,
    pub(crate) compute_instance_id: u32,
    pub(crate) uuid: String,
    pub(crate) _resource: Option<SharedResource<NvidiaMigResource>>,
}

impl VirtualGpu for NvidiaMigVirtualGpu {
    type HostDevice = NvidiaGpu;
    type State = &'static str;
    type Error = String;

    fn host_device(&self) -> impl Deref<Target = Self::HostDevice> + Send + '_ {
        self.host_device.clone()
    }

    async fn state(&self) -> Self::State {
        self.scale
    }

    fn release(mut self) -> Result<(), String> {
        self.release_inner()
    }

    fn logic_index(&self) -> Option<u32> {
        Some(0)
    }
}

impl NvidiaMigVirtualGpu {
    fn release_inner(&mut self) -> Result<(), String> {
        let result = (|| {
            if self._resource.is_some() {
                let gpu_id = self
                    .host_device
                    .logic_index()
                    .ok_or("Missing GPU index")?
                    .to_string();
                destroy_mig_vdevice(
                    gpu_id,
                    self.gpu_instance_id.to_string(),
                    self.compute_instance_id.to_string(),
                )?;
                self._resource.take();
            }
            Ok(())
        })();
        if result.is_err() {
            if let Some(resource) = self._resource.take() {
                std::mem::forget(resource);
            }
        }
        result
    }
}

impl Drop for NvidiaMigVirtualGpu {
    #[tracing::instrument(level = "trace")]
    fn drop(&mut self) {
        if let Err(e) = self.release_inner() {
            tracing::error!("Failed to destroy MIG vGPU; lease retained: {e}");
        }
    }
}

#[cfg(feature = "torch")]
impl RunnableOn<NvidiaMigVirtualGpu> for TorchWorker {
    type DeviceIndependentConfig = TorchWorkerModelLoadConfig;

    fn device_config_getter(
        _device: &NvidiaMigVirtualGpu,
    ) -> impl FnOnce() -> <Self::DeviceIndependentConfig as IndependentModelLoadConfig>::RequireDevice
    {
        || worker::model::tensor::Device::Gpu(0)
    }
}

#[cfg(feature = "onnx")]
impl RunnableOn<NvidiaMigVirtualGpu> for OnnxWorker {
    type DeviceIndependentConfig = OnnxWorkerModelLoadConfig;

    fn device_config_getter(
        _device: &NvidiaMigVirtualGpu,
    ) -> impl FnOnce() -> <Self::DeviceIndependentConfig as IndependentModelLoadConfig>::RequireDevice
    {
        || {
            vec![
                ort::execution_providers::CUDAExecutionProvider::default()
                    .with_device_id(0)
                    .build()
                    .error_on_failure(),
            ]
        }
    }
}
