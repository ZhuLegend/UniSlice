//! Nvidia GPU 支持

#[cfg(feature = "vgpu")]
use std::ops::Deref;

use nvml_wrapper::struct_wrappers::device::MemoryInfo;
#[cfg(feature = "onnx")]
use worker::onnx::worker::OnnxWorker;
#[cfg(feature = "torch")]
use worker::torch::worker::TorchWorker;

#[cfg(feature = "onnx")]
use crate::compute_device::config::OnnxWorkerModelLoadConfig;
#[cfg(feature = "torch")]
use crate::compute_device::config::TorchWorkerModelLoadConfig;
#[cfg(feature = "vgpu")]
use crate::compute_device::nv::nvml;
use crate::compute_device::{
    ComputeDevice,
    resource_spec::nv::{NvGpuRequest, NvGpuResponse},
};
#[cfg(feature = "vgpu")]
use crate::compute_node::r#virtual::base::DeviceManageHelper;
#[cfg(any(feature = "onnx", feature = "torch"))]
use crate::runnable_on::{IndependentModelLoadConfig, RunnableOn};

/// Nvidia GPU 支持
#[derive(Debug)]
pub struct NvidiaGpu {
    logic_index: Option<u32>,
    /// GPU
    pub(crate) device: nvml_wrapper::Device<'static>,
}

impl NvidiaGpu {
    /// 创建一个 Nvidia GPU
    pub fn new(device: nvml_wrapper::Device<'static>) -> Self {
        let logic_index = device.index().ok();
        NvidiaGpu {
            logic_index,
            device,
        }
    }
}

impl NvidiaGpu {
    pub(crate) fn memory_info(&self) -> NvGpuResponse {
        match self.device.memory_info() {
            Ok(MemoryInfo {
                total, free, used, ..
            }) => NvGpuResponse::MemoryInfo { total, free, used },
            Err(e) => NvGpuResponse::NvmlError(e.to_string()),
        }
    }
}

impl ComputeDevice for NvidiaGpu {
    type Request = NvGpuRequest;
    type Response = NvGpuResponse;

    async fn request(&self, request: Self::Request) -> Self::Response {
        match request {
            NvGpuRequest::QueryMemoryInfo => self.memory_info(),
        }
    }

    fn logic_index(&self) -> Option<u32> {
        self.logic_index
    }
}

#[cfg(feature = "vgpu")]
impl DeviceManageHelper for NvidiaGpu {
    fn get_device(id: usize) -> (uuid::Uuid, Self) {
        let nvml = nvml();
        let device = nvml
            .device_by_index(id as u32)
            .unwrap_or_else(|_| panic!("Could not find device {id}"));
        (uuid::Uuid::new_v4(), NvidiaGpu::new(device))
    }

    fn get_device_state(device: impl Deref<Target = Self>) -> <Self as ComputeDevice>::Response {
        device.memory_info()
    }
}

#[cfg(feature = "torch")]
impl RunnableOn<NvidiaGpu> for TorchWorker {
    type DeviceIndependentConfig = TorchWorkerModelLoadConfig;

    fn device_config_getter(
        device: &NvidiaGpu,
    ) -> impl FnOnce() -> <Self::DeviceIndependentConfig as IndependentModelLoadConfig>::RequireDevice
    {
        || {
            worker::model::tensor::Device::Gpu(
                device.device.index().expect("Failed to get device index") as usize,
            )
        }
    }
}

#[cfg(feature = "onnx")]
impl RunnableOn<NvidiaGpu> for OnnxWorker {
    type DeviceIndependentConfig = OnnxWorkerModelLoadConfig;

    fn device_config_getter(
        device: &NvidiaGpu,
    ) -> impl FnOnce() -> <Self::DeviceIndependentConfig as IndependentModelLoadConfig>::RequireDevice
    {
        || {
            vec![
                ort::execution_providers::CUDAExecutionProvider::default()
                    .with_device_id(
                        device.device.index().expect("Failed to get device index") as i32
                    )
                    .build()
                    .error_on_failure(),
            ]
        }
    }
}
