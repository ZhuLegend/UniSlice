use std::{
    fmt::Debug,
    sync::{Arc, Mutex},
};

use super::ComputeDevice;
use crate::compute_device::vgpu::share_target::ShareTarget;

/// 可共享的GPU设备
pub trait ShareableComputeDevice: From<Self::RawComputeDevice> {
    /// 可共享的资源
    type Resource: ShareableComputeDeviceResource + for<'a> From<&'a Self::ShareParam>;

    /// 原始[ComputeDevice]
    type RawComputeDevice: ComputeDevice;

    /// 创建[Self::SharedComputeDevice]的参数
    type ShareParam: ShareTarget;

    /// 从原始[ComputeDevice]上共享资源的[ComputeDevice]
    ///
    /// # Warning
    /// SharedComputeDevice应当持有[SharedResource<Self::Resource>]，当[SharedResource<Self::Resource>]释放时，资源立刻释放
    type SharedComputeDevice: ComputeDevice
        + for<'a> TryFrom<(&'a Self, Self::ShareParam, SharedResource<Self::Resource>), Error: Debug>;

    /// 获取原始[ComputeDevice]的引用
    fn raw_device(&self) -> &Self::RawComputeDevice;

    /// [ShareableResource<Self::Resource>]的引用
    fn shareable_resource(&self) -> &ShareableResource<Self::Resource>;

    /// 总资源
    fn total_resources(&self) -> Self::Resource;

    /// 剩余资源
    fn remain_resource(&self) -> Self::Resource;

    /// 是否可以实现共享
    fn can_share(&self, param: &Self::ShareParam) -> bool {
        self.remain_resource().can_share(&(param.into()))
    }

    /// 进行资源分享
    ///
    /// 当资源不足时，返回[None]
    #[allow(clippy::type_complexity)]
    fn share(
        &self,
        param: Self::ShareParam,
    ) -> Option<
        Result<
            Self::SharedComputeDevice,
            <Self::SharedComputeDevice as TryFrom<(
                &Self,
                Self::ShareParam,
                SharedResource<Self::Resource>,
            )>>::Error,
        >,
    > {
        let shared_resource = self.shareable_resource().share((&param).into())?;
        Some((self, param, shared_resource).try_into())
    }
}

/// 可共享的计算设备资源
pub trait ShareableComputeDeviceResource: Sized + Copy + Send + Sync + 'static {
    /// 是否可以提供指定目标的资源进行共享
    fn can_share(&self, other: &Self) -> bool;

    /// 从资源中切分共享资源，如果资源不足，返回[None]
    fn borrow(&mut self, target: &Self) -> Option<Self>;

    /// 归还借用的资源
    fn refund(&mut self, target: Self);
}

/// [ShareableComputeDevice]可共享资源
#[derive(Debug)]
pub struct ShareableResource<T: ShareableComputeDeviceResource> {
    resource: Arc<Mutex<T>>,
}

impl<T: ShareableComputeDeviceResource> From<T> for ShareableResource<T> {
    fn from(resource: T) -> Self {
        Self::new(resource)
    }
}

impl<T: ShareableComputeDeviceResource> ShareableResource<T> {
    pub(crate) fn new(resource: T) -> Self {
        Self {
            resource: Arc::new(Mutex::new(resource)),
        }
    }

    /// 借用资源，当资源不足时，返回None
    pub(crate) fn share(&self, resource: T) -> Option<SharedResource<T>> {
        let borrowed_resource = self.resource.lock().unwrap().borrow(&resource)?;
        Some(SharedResource {
            parent: self.resource.clone(),
            resource: borrowed_resource,
        })
    }

    /// 当前可用资源
    pub(crate) fn current_resource(&self) -> T {
        *self.resource.lock().unwrap()
    }
}

/// 借用的资源，当Drop时自动归还资源
#[derive(Debug)]
pub struct SharedResource<T: ShareableComputeDeviceResource> {
    parent: Arc<Mutex<T>>,
    resource: T,
}

impl<T: ShareableComputeDeviceResource> Drop for SharedResource<T> {
    fn drop(&mut self) {
        self.parent.lock().unwrap().refund(self.resource);
    }
}

#[allow(missing_docs)]
#[cfg(any(test, feature = "test_utils"))]
pub mod test {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct TestResource {
        pub g: u8,
        pub gb: u8,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum TestShareTarget {
        G7Gb80,
        G4Gb40,
        G3Gb40,
        G2Gb20,
        G1Gb10,
    }

    impl ShareableComputeDeviceResource for TestResource {
        fn can_share(&self, other: &Self) -> bool {
            self.g >= other.g && self.gb >= other.gb
        }

        fn borrow(&mut self, target: &Self) -> Option<Self> {
            if self.can_share(target) {
                self.g -= target.g;
                self.gb -= target.gb;
                Some(*target)
            } else {
                None
            }
        }

        fn refund(&mut self, target: Self) {
            self.g += target.g;
            self.gb += target.gb;
        }
    }

    impl TestResource {
        pub const MAX_G: u8 = 7;
        pub const MAX_GB: u8 = 80;

        pub const MAX_SELF: Self = Self {
            g: Self::MAX_G,
            gb: Self::MAX_GB,
        };

        pub fn new(g: u8, gb: u8) -> Self {
            assert!(g <= Self::MAX_G && gb <= Self::MAX_GB);
            Self { g, gb }
        }

        pub fn as_tuple(&self) -> (u8, u8) {
            (self.g, self.gb)
        }

        pub fn from_tuple(g: u8, gb: u8) -> Option<Self> {
            if g <= Self::MAX_G && gb <= Self::MAX_GB {
                Some(Self { g, gb })
            } else {
                None
            }
        }
    }

    impl From<TestShareTarget> for TestResource {
        fn from(target: TestShareTarget) -> Self {
            match target {
                TestShareTarget::G7Gb80 => TestResource { g: 7, gb: 80 },
                TestShareTarget::G4Gb40 => TestResource { g: 4, gb: 40 },
                TestShareTarget::G3Gb40 => TestResource { g: 3, gb: 40 },
                TestShareTarget::G2Gb20 => TestResource { g: 2, gb: 20 },
                TestShareTarget::G1Gb10 => TestResource { g: 1, gb: 10 },
            }
        }
    }

    impl TestShareTarget {
        pub const ALL_CASES: [TestShareTarget; 5] = [
            TestShareTarget::G7Gb80,
            TestShareTarget::G4Gb40,
            TestShareTarget::G3Gb40,
            TestShareTarget::G2Gb20,
            TestShareTarget::G1Gb10,
        ];

        pub const MIN: [TestShareTarget; 1] = [TestShareTarget::G1Gb10];

        pub fn as_tuple(&self) -> (u8, u8) {
            match self {
                TestShareTarget::G7Gb80 => (7, 80),
                TestShareTarget::G4Gb40 => (4, 40),
                TestShareTarget::G3Gb40 => (3, 40),
                TestShareTarget::G2Gb20 => (2, 20),
                TestShareTarget::G1Gb10 => (1, 10),
            }
        }
    }

    #[cfg(test)]
    #[test]
    fn test_split_resource() {
        let mut res = TestResource { g: 7, gb: 80 };
        let target = TestResource { g: 4, gb: 40 };
        let borrowed = res.borrow(&target);
        assert_eq!(borrowed, Some(TestResource { g: 4, gb: 40 }));
        assert_eq!(res, TestResource { g: 3, gb: 40 });

        let mut res = TestResource { g: 3, gb: 40 };
        let target = TestResource { g: 1, gb: 10 };
        let borrowed = res.borrow(&target);
        assert_eq!(borrowed, Some(TestResource { g: 1, gb: 10 }));
        assert_eq!(res, TestResource { g: 2, gb: 30 });
    }

    #[cfg(test)]
    #[test]
    fn test_compose_resource() {
        let mut res1 = TestResource { g: 3, gb: 40 };
        let res2 = TestResource { g: 4, gb: 40 };
        res1.refund(res2);
        assert_eq!(res1, TestResource { g: 7, gb: 80 });
    }
}
