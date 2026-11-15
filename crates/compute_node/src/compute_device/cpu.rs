//! CPU Compute Device

use clap::ValueEnum;
#[cfg(feature = "onnx")]
use worker::onnx::worker::OnnxWorker;
#[cfg(feature = "torch")]
use worker::torch::worker::TorchWorker;

use crate::compute_device::ComputeDevice;
#[cfg(feature = "onnx")]
use crate::compute_device::config::OnnxWorkerModelLoadConfig;
#[cfg(feature = "torch")]
use crate::compute_device::config::TorchWorkerModelLoadConfig;
#[cfg(any(feature = "onnx", feature = "torch"))]
use crate::runnable_on::{IndependentModelLoadConfig, RunnableOn};

/// CPU Compute Device
#[derive(Debug)]
pub struct Cpu {}

impl ComputeDevice for Cpu {
    type Request = ();
    type Response = ();

    async fn request(&self, _request: Self::Request) -> Self::Response {}

    fn logic_index(&self) -> Option<u32> {
        None
    }
}

/// [Worker](worker::Worker)支持的Worker类型
#[derive(Clone, ValueEnum)]
pub enum WorkerType {
    /// 使用 Pytorch C++ API 执行模型推理
    #[cfg(feature = "torch")]
    Torch,
    /// 使用 Onnx Backend 执行模型推理
    #[cfg(feature = "onnx")]
    Onnx,
}

#[cfg(feature = "onnx")]
impl RunnableOn<Cpu> for OnnxWorker {
    type DeviceIndependentConfig = OnnxWorkerModelLoadConfig;

    fn device_config_getter(
        _device: &Cpu,
    ) -> impl FnOnce() -> <Self::DeviceIndependentConfig as IndependentModelLoadConfig>::RequireDevice
    {
        || vec![ort::execution_providers::CPUExecutionProvider::default().build()]
    }
}

#[cfg(feature = "torch")]
impl RunnableOn<Cpu> for TorchWorker {
    type DeviceIndependentConfig = TorchWorkerModelLoadConfig;

    fn device_config_getter(
        _device: &Cpu,
    ) -> impl FnOnce() -> <Self::DeviceIndependentConfig as IndependentModelLoadConfig>::RequireDevice
    {
        || worker::model::tensor::Device::Cpu
    }
}
