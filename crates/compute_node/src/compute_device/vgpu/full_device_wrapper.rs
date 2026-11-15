use std::{
    ops::Deref,
    sync::{Arc, LazyLock},
};

use crate::compute_device::{
    ComputeDevice,
    vgpu::{
        ShareableComputeDevice, ShareableComputeDeviceResource, ShareableResource, SharedResource,
        share_target::ShareTarget,
    },
};

/// Full Device Share Wrapper
///
/// This wrapper allows sharing the entire compute device.
///
/// Share strategy is decided by the underlying `ComputeDevice`'s physic device.
#[derive(Debug)]
pub struct FullDeviceShareWrapper<T: ComputeDevice> {
    inner: Arc<T>,
    _resource: FullDeviceShareResource,
}

impl<T: ComputeDevice> Clone for FullDeviceShareWrapper<T> {
    fn clone(&self) -> Self {
        FullDeviceShareWrapper {
            inner: self.inner.clone(),
            _resource: FullDeviceShareResource,
        }
    }
}

impl<T: ComputeDevice> From<T> for FullDeviceShareWrapper<T> {
    fn from(device: T) -> Self {
        FullDeviceShareWrapper {
            inner: Arc::new(device),
            _resource: FullDeviceShareResource,
        }
    }
}

impl<T: ComputeDevice> Deref for FullDeviceShareWrapper<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T: ComputeDevice + Sync> ComputeDevice for FullDeviceShareWrapper<T> {
    type Request = T::Request;
    type Response = T::Response;

    async fn request(&self, request: Self::Request) -> Self::Response {
        self.inner.request(request).await
    }

    fn logic_index(&self) -> Option<u32> {
        self.inner.logic_index()
    }
}

/// Full Device Share Resource
///
/// Do nothing but indicating that the entire device can be shared.
/// share strategy is decided by physic `ComputeDevice`.
#[derive(Debug, Copy, Clone)]
pub struct FullDeviceShareResource;

impl ShareableComputeDeviceResource for FullDeviceShareResource {
    #[inline]
    fn can_share(&self, _other: &Self) -> bool {
        true
    }

    #[inline]
    fn borrow(&mut self, _target: &Self) -> Option<Self> {
        Some(FullDeviceShareResource)
    }

    #[inline]
    fn refund(&mut self, _target: Self) {}
}

impl From<&'_ FullDeviceShareTarget> for FullDeviceShareResource {
    fn from(_: &'_ FullDeviceShareTarget) -> Self {
        FullDeviceShareResource
    }
}

impl<T: ComputeDevice>
    From<(
        &FullDeviceShareWrapper<T>,
        FullDeviceShareTarget,
        SharedResource<FullDeviceShareResource>,
    )> for FullDeviceShareWrapper<T>
{
    fn from(
        (device, _param, _shared_resource): (
            &FullDeviceShareWrapper<T>,
            FullDeviceShareTarget,
            SharedResource<FullDeviceShareResource>,
        ),
    ) -> Self {
        (*device).clone()
    }
}

/// Full Device Share Target
#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct FullDeviceShareTarget;

impl ShareTarget for FullDeviceShareTarget {
    fn name(&self) -> &'static str {
        "full"
    }

    fn from_name(name: &str) -> Option<Self> {
        if name == "full" {
            Some(FullDeviceShareTarget)
        } else {
            None
        }
    }
}

impl<T: ComputeDevice + Send + Sync> ShareableComputeDevice for FullDeviceShareWrapper<T> {
    type Resource = FullDeviceShareResource;
    type RawComputeDevice = T;
    type ShareParam = FullDeviceShareTarget;
    type SharedComputeDevice = FullDeviceShareWrapper<T>;

    fn raw_device(&self) -> &Self::RawComputeDevice {
        self
    }

    fn shareable_resource(&self) -> &ShareableResource<Self::Resource> {
        static FULL_DEVICE_SHARE_RESOURCE: LazyLock<ShareableResource<FullDeviceShareResource>> =
            LazyLock::new(|| ShareableResource::new(FullDeviceShareResource));
        &FULL_DEVICE_SHARE_RESOURCE
    }

    fn total_resources(&self) -> Self::Resource {
        FullDeviceShareResource
    }

    fn remain_resource(&self) -> Self::Resource {
        FullDeviceShareResource
    }
}
