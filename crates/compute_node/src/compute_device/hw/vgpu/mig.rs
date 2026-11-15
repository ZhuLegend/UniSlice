//! MIG-like NPU resource sharing for Huawei GPUs

use std::sync::Arc;

use hw_dcmi_wrapper::device::chip::Chip;

use crate::compute_device::{
    hw::{HuaweiGpu, HuaweiVirtualGpu, mig::create_vchip},
    vgpu::{
        ShareableComputeDevice, ShareableResource, SharedResource,
        resource_spec::hw::{HuaweiMigResource, VChipScale},
    },
};

/// Huawei MIG-like [ShareableComputeDevice]
pub struct HuaweiMigShareableGpu {
    raw_device: Arc<HuaweiGpu>,
    chip_id: u32,
    total_resource: HuaweiMigResource,
    resource: ShareableResource<HuaweiMigResource>,
}

impl From<HuaweiGpu> for HuaweiMigShareableGpu {
    fn from(raw_device: HuaweiGpu) -> Self {
        // TODO: for now, use hardcode number
        let resource = HuaweiMigResource {
            ai_core: 20,
            ai_cpu: 7,
            memory: 64,
        };
        let chip_id = raw_device
            .device()
            .get_chips()
            .map(|chips| chips.0.first().unwrap().id())
            .unwrap_or(0);
        Self {
            raw_device: Arc::new(raw_device),
            chip_id,
            total_resource: resource,
            resource: resource.into(),
        }
    }
}

impl ShareableComputeDevice for HuaweiMigShareableGpu {
    type Resource = HuaweiMigResource;
    type RawComputeDevice = HuaweiGpu;
    type ShareParam = VChipScale;
    type SharedComputeDevice = HuaweiVirtualGpu;

    fn raw_device(&self) -> &Self::RawComputeDevice {
        &self.raw_device
    }

    fn shareable_resource(&self) -> &ShareableResource<Self::Resource> {
        &self.resource
    }

    fn total_resources(&self) -> Self::Resource {
        self.total_resource
    }

    fn remain_resource(&self) -> Self::Resource {
        self.resource.current_resource()
    }
}

impl
    TryFrom<(
        &'_ HuaweiMigShareableGpu,
        VChipScale,
        SharedResource<HuaweiMigResource>,
    )> for HuaweiVirtualGpu
{
    type Error = String;

    fn try_from(
        (device, vchip_scale, _resource): (
            &HuaweiMigShareableGpu,
            VChipScale,
            SharedResource<HuaweiMigResource>,
        ),
    ) -> Result<Self, Self::Error> {
        let chip = Chip::new_unchecked(device.raw_device.device(), device.chip_id);
        let vchip = create_vchip(&chip, vchip_scale)
            .map_err(|e| format!("Failed to create vchip: {e}"))?
            .1;
        Ok(HuaweiVirtualGpu {
            host_device: device.raw_device.clone(),
            chip_id: device.chip_id,
            vchip_id: vchip.id(),
            vfg_id: vchip.vfg_id(),
            scale: vchip_scale,
            _resource: Some(_resource),
        })
    }
}
