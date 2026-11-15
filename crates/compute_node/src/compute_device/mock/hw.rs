//! A Huawei Virtual Compute Device mock for testing purposes.

use std::sync::Arc;

use crate::compute_device::{
    mock::{MockComputeDevice, MockVirtualComputeDevice},
    vgpu::{
        ShareableComputeDevice, ShareableResource, SharedResource,
        resource_spec::hw::{HuaweiMigResource, VChipScale},
    },
};

/// Mock Huawei MIG-like [ShareableComputeDevice] for testing purposes.
#[derive(Debug)]
pub struct MockHuaweiMigShareableGpu {
    raw_device: Arc<MockComputeDevice>,
    total_resource: HuaweiMigResource,
    resource: ShareableResource<HuaweiMigResource>,
}

impl From<MockComputeDevice> for MockHuaweiMigShareableGpu {
    fn from(raw_device: MockComputeDevice) -> Self {
        let resource = HuaweiMigResource {
            ai_core: 20,
            ai_cpu: 7,
            memory: 64,
        };
        Self::with_capacity(raw_device, resource)
    }
}

impl MockHuaweiMigShareableGpu {
    /// Create a mock resource pool with an explicit capacity.
    pub fn with_capacity(raw_device: MockComputeDevice, resource: HuaweiMigResource) -> Self {
        Self {
            raw_device: Arc::new(raw_device),
            total_resource: resource,
            resource: resource.into(),
        }
    }
}

impl ShareableComputeDevice for MockHuaweiMigShareableGpu {
    type Resource = HuaweiMigResource;
    type RawComputeDevice = MockComputeDevice;
    type ShareParam = VChipScale;
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
        &'_ MockHuaweiMigShareableGpu,
        VChipScale,
        SharedResource<HuaweiMigResource>,
    )> for MockVirtualComputeDevice
{
    type Error = &'static str;

    fn try_from(
        (device, scale, resource): (
            &MockHuaweiMigShareableGpu,
            VChipScale,
            SharedResource<HuaweiMigResource>,
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
