//! Mps Resource Sharing for Nvidia GPU

use std::sync::Arc;

use crate::compute_device::{
    nv::{NvidiaGpu, NvidiaMpsVirtualGpu},
    vgpu::{
        ShareableComputeDevice, ShareableResource, SharedResource,
        resource_spec::nv::mps::{NvidiaMpsResource, NvidiaMpsShareTarget},
    },
};

/// Nvidia Mps Shareable GPU
///
/// allows sharing of GPU resources using Nvidia's Multi-Process Service (MPS).
pub struct NvidiaMpsShareableGpu {
    raw_device: Arc<NvidiaGpu>,
    total_resource: NvidiaMpsResource,
    resource: ShareableResource<NvidiaMpsResource>,
}

impl From<NvidiaGpu> for NvidiaMpsShareableGpu {
    fn from(raw_device: NvidiaGpu) -> Self {
        NvidiaMpsShareableGpu {
            raw_device: Arc::new(raw_device),
            total_resource: NvidiaMpsResource(100),
            resource: ShareableResource::new(NvidiaMpsResource(100)),
        }
    }
}

impl
    TryFrom<(
        &'_ NvidiaMpsShareableGpu,
        NvidiaMpsShareTarget,
        SharedResource<NvidiaMpsResource>,
    )> for NvidiaMpsVirtualGpu
{
    type Error = ();

    fn try_from(
        (device, target, _resource): (
            &'_ NvidiaMpsShareableGpu,
            NvidiaMpsShareTarget,
            SharedResource<NvidiaMpsResource>,
        ),
    ) -> Result<Self, Self::Error> {
        Ok(NvidiaMpsVirtualGpu {
            host_device: device.raw_device.clone(),
            percentage: target.0,
            _resource,
        })
    }
}

impl ShareableComputeDevice for NvidiaMpsShareableGpu {
    type Resource = NvidiaMpsResource;
    type RawComputeDevice = NvidiaGpu;
    type ShareParam = NvidiaMpsShareTarget;
    type SharedComputeDevice = NvidiaMpsVirtualGpu;

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
