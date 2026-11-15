//! Huawei VGpu 模块

use std::{ops::Deref, sync::Arc};

use hw_dcmi_wrapper::device::{chip::Chip, vchip::VChip};
#[cfg(feature = "onnx")]
use worker::onnx::worker::OnnxWorker;

#[cfg(feature = "vgpu")]
use crate::compute_device::vgpu::resource_spec::hw::HuaweiMigResource;
#[cfg(feature = "vgpu")]
use crate::compute_device::vgpu::resource_spec::hw::VChipScale;
#[cfg(feature = "vgpu")]
use crate::compute_device::{VirtualGpu, hw::gpu::HuaweiGpu, vgpu::SharedResource};
#[cfg(feature = "onnx")]
use crate::{
    compute_device::config::OnnxWorkerModelLoadConfig,
    runnable_on::{IndependentModelLoadConfig, RunnableOn},
};

pub mod mig;

/// Huawei VGpu
#[derive(Debug)]
pub struct HuaweiVirtualGpu {
    host_device: Arc<HuaweiGpu>,
    chip_id: u32,
    pub(crate) vchip_id: u32,
    vfg_id: u32,
    pub(crate) scale: VChipScale,
    _resource: Option<SharedResource<HuaweiMigResource>>,
}

impl VirtualGpu for HuaweiVirtualGpu {
    type HostDevice = HuaweiGpu;
    type State = VChipScale;
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
}

impl HuaweiVirtualGpu {
    fn release_inner(&mut self) -> Result<(), String> {
        let result = (|| {
            if self._resource.is_some() {
                let chip = Chip::new_unchecked(self.host_device.device(), self.chip_id);
                let vchip = VChip::new_unchecked(&chip, self.vchip_id, self.vfg_id);
                vchip
                    .destroy()
                    .map_err(|e| format!("Failed to destroy vchip: {e}"))?;
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

impl Drop for HuaweiVirtualGpu {
    fn drop(&mut self) {
        if let Err(e) = self.release_inner() {
            tracing::error!("Failed to destroy vchip; lease retained: {e}");
        }
    }
}

#[cfg(feature = "onnx")]
impl RunnableOn<HuaweiVirtualGpu> for OnnxWorker {
    type DeviceIndependentConfig = OnnxWorkerModelLoadConfig;

    fn device_config_getter(
        device: &HuaweiVirtualGpu,
    ) -> impl FnOnce() -> <Self::DeviceIndependentConfig as IndependentModelLoadConfig>::RequireDevice
    {
        || {
            vec![
                ort::execution_providers::CANNExecutionProvider::default()
                    .with_device_id(device.vchip_id as i32)
                    .with_cann_graph(false)
                    .with_dump_graphs(true)
                    .build()
                    .error_on_failure(),
            ]
        }
    }
}
