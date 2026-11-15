//! Huawei GPU 支持

#[cfg(feature = "vgpu")]
use std::ops::Deref;

use hw_dcmi_wrapper::{
    device::{card::Card, chip::Chip},
    error::DCMIResult,
};
#[cfg(feature = "vgpu")]
use uuid::Uuid;
#[cfg(feature = "onnx")]
use worker::onnx::worker::OnnxWorker;

#[cfg(feature = "vgpu")]
use crate::compute_device::hw::dcmi;
use crate::compute_device::{
    ComputeDevice,
    resource_spec::hw::{HwGpuRequest, HwGpuResponse},
};
#[cfg(feature = "vgpu")]
use crate::compute_node::r#virtual::base::DeviceManageHelper;
#[cfg(feature = "onnx")]
use crate::{
    compute_device::config::OnnxWorkerModelLoadConfig,
    runnable_on::{IndependentModelLoadConfig, RunnableOn},
};

/// Huawei GPU 支持
#[derive(Debug)]
pub struct HuaweiGpu {
    /// GPU
    device: Card<'static>,
}

impl HuaweiGpu {
    /// 创建一个 Huawei GPU
    pub fn new(device: Card<'static>) -> Self {
        HuaweiGpu { device }
    }

    /// 获取GPU Device
    pub fn device(&self) -> &Card<'static> {
        &self.device
    }
}

fn map_chip<T, U>(
    device: &Card,
    get_info: impl Fn(&Chip) -> DCMIResult<T>,
) -> DCMIResult<Vec<(u32, Result<U, String>)>>
where
    U: From<T>,
{
    let (chips, _mcu_chip, _cpu_chip) = device.get_chips()?;
    Ok(chips
        .iter()
        .map(|chip| {
            (
                chip.id(),
                match get_info(chip) {
                    Ok(v) => Ok(v.into()),
                    Err(e) => Err(e.to_string()),
                },
            )
        })
        .collect())
}

impl HuaweiGpu {
    pub(crate) fn memory_info(&self) -> HwGpuResponse {
        let memory_infos = map_chip(self.device(), |chip| chip.get_memory_info());
        match memory_infos {
            Ok(v) => HwGpuResponse::MemoryInfos(v),
            Err(e) => HwGpuResponse::DCMIError(e.to_string()),
        }
    }
    pub(crate) fn hbm_info(&self) -> HwGpuResponse {
        let hbm_infos = map_chip(self.device(), |chip| chip.get_hbm_info());
        match hbm_infos {
            Ok(v) => HwGpuResponse::HBMInfos(v),
            Err(e) => HwGpuResponse::DCMIError(e.to_string()),
        }
    }
}

impl ComputeDevice for HuaweiGpu {
    type Request = HwGpuRequest;
    type Response = HwGpuResponse;

    async fn request(&self, request: Self::Request) -> Self::Response {
        match request {
            HwGpuRequest::QueryMemoryInfo => self.memory_info(),
            HwGpuRequest::QueryHBMInfo => self.hbm_info(),
        }
    }

    fn logic_index(&self) -> Option<u32> {
        Some(self.device.id())
    }
}

#[cfg(feature = "vgpu")]
impl DeviceManageHelper for HuaweiGpu {
    fn get_device(id: usize) -> (Uuid, Self) {
        let dcmi = dcmi();
        let device = Card::new_unchecked(dcmi, id as u32);
        let device = HuaweiGpu::new(device);
        (Uuid::new_v4(), device)
    }

    fn get_device_state(device: impl Deref<Target = Self>) -> <Self as ComputeDevice>::Response {
        device.hbm_info()
    }
}

#[cfg(feature = "onnx")]
impl RunnableOn<HuaweiGpu> for OnnxWorker {
    type DeviceIndependentConfig = OnnxWorkerModelLoadConfig;

    fn device_config_getter(
        device: &HuaweiGpu,
    ) -> impl FnOnce() -> <Self::DeviceIndependentConfig as IndependentModelLoadConfig>::RequireDevice
    {
        || {
            vec![
                ort::execution_providers::CANNExecutionProvider::default()
                    .with_device_id(device.device.id() as i32)
                    .build()
                    .error_on_failure(),
            ]
        }
    }
}
