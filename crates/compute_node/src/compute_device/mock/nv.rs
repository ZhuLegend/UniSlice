//! Nvidia virtual shareable GPU mock implementation.

use std::{fmt::Debug, sync::Arc};

use crate::compute_device::{
    mock::{MockComputeDevice, MockVirtualComputeDevice},
    vgpu::{
        ShareableComputeDevice, ShareableResource, SharedResource,
        resource_spec::nv::mig::{NvidiaMigResource, VDeviceScale},
    },
};

/// A mock implementation for Nvidia MIG like shareable GPU for testing purposes.
#[derive(Debug)]
pub struct MockNvidiaMigShareableGpu {
    raw_device: Arc<MockComputeDevice>,
    total_resource: NvidiaMigResource,
    resource: ShareableResource<NvidiaMigResource>,
}

impl From<MockComputeDevice> for MockNvidiaMigShareableGpu {
    fn from(raw_device: MockComputeDevice) -> Self {
        let resource = NvidiaMigResource { ce: 7, memory: 80 };
        Self::with_capacity(raw_device, resource)
    }
}

impl MockNvidiaMigShareableGpu {
    /// Create a mock resource pool with an explicit capacity.
    pub fn with_capacity(raw_device: MockComputeDevice, resource: NvidiaMigResource) -> Self {
        Self {
            raw_device: Arc::new(raw_device),
            total_resource: resource,
            resource: resource.into(),
        }
    }
}

impl ShareableComputeDevice for MockNvidiaMigShareableGpu {
    type Resource = NvidiaMigResource;
    type RawComputeDevice = MockComputeDevice;
    type ShareParam = VDeviceScale;
    type SharedComputeDevice = MockVirtualComputeDevice;

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
        &'_ MockNvidiaMigShareableGpu,
        VDeviceScale,
        SharedResource<NvidiaMigResource>,
    )> for MockVirtualComputeDevice
{
    type Error = &'static str;

    fn try_from(
        (device, scale, resource): (
            &MockNvidiaMigShareableGpu,
            VDeviceScale,
            SharedResource<NvidiaMigResource>,
        ),
    ) -> Result<Self, Self::Error> {
        if device
            .raw_device
            .simulation
            .as_ref()
            .is_some_and(|s| !s.share_targets.iter().any(|name| name == scale.name()))
        {
            return Err("slice is not in simulation whitelist");
        }
        Ok(MockVirtualComputeDevice {
            host_device: (*device.raw_device).clone(),
            config: scale.name().to_string(),
            _resource: Arc::new(super::MockResourceLease::new(resource, (&scale).into())),
        })
    }
}
