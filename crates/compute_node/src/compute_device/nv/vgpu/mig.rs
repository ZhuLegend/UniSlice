//! MIG GPU resource sharing for Nvidia GPUs

use std::sync::Arc;

use crate::compute_device::{
    nv::{NvidiaGpu, mig::create_mig_vdevice, vgpu::NvidiaMigVirtualGpu},
    vgpu::{
        ShareableComputeDevice, ShareableResource, SharedResource,
        resource_spec::nv::mig::{NvidiaMigResource, VDeviceScale},
    },
};

/// Nvidia MIG [ShareableComputeDevice]
pub struct NvidiaMigShareableGpu {
    raw_device: Arc<NvidiaGpu>,
    #[allow(dead_code)]
    uuid: String,
    total_resource: NvidiaMigResource,
    resource: ShareableResource<NvidiaMigResource>,
}

impl From<NvidiaGpu> for NvidiaMigShareableGpu {
    fn from(raw_device: NvidiaGpu) -> Self {
        // TODO: for now, use hardcode number
        let resource = NvidiaMigResource { ce: 7, memory: 80 };
        let uuid = raw_device.device.uuid().unwrap();
        Self {
            raw_device: Arc::new(raw_device),
            uuid,
            total_resource: resource,
            resource: resource.into(),
        }
    }
}

impl ShareableComputeDevice for NvidiaMigShareableGpu {
    type Resource = NvidiaMigResource;
    type RawComputeDevice = NvidiaGpu;
    type ShareParam = VDeviceScale;
    type SharedComputeDevice = NvidiaMigVirtualGpu;

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
        &'_ NvidiaMigShareableGpu,
        VDeviceScale,
        SharedResource<NvidiaMigResource>,
    )> for NvidiaMigVirtualGpu
{
    type Error = String;

    fn try_from(
        (device, scale, _resource): (
            &'_ NvidiaMigShareableGpu,
            VDeviceScale,
            SharedResource<NvidiaMigResource>,
        ),
    ) -> Result<Self, Self::Error> {
        let host_device = device.raw_device.clone();
        let v_device_output = create_mig_vdevice(host_device.clone(), scale)
            .map_err(|e| format!("Failed to create MIG device: {e}"))?;
        Ok(NvidiaMigVirtualGpu {
            host_device,
            scale: scale.name(),
            gpu_instance_id: v_device_output.gpu_instance_id,
            compute_instance_id: v_device_output.compute_instance_id,
            uuid: v_device_output.uuid,
            _resource: Some(_resource),
        })
    }
}
