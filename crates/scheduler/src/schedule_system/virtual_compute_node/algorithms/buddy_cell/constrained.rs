//! 受约束的资源包装器，用于限制资源合并不超过配置的最大上限

use std::{
    collections::HashMap,
    fmt::Debug,
    hash::{Hash, Hasher},
};

use compute_node::compute_device::vgpu::{
    ShareableComputeDeviceResource, share_target::ShareTarget,
};

use crate::schedule_system::virtual_compute_node::algorithms::buddy_cell::{
    BuddyCellResource, CanExceedLimit,
};

/// 受约束的资源包装器
///
/// 此包装器允许为任何实现了 BuddyCellResource 的资源类型添加最大上限约束。
/// 在资源合并时，会检查合并后的资源是否超过配置的最大上限，如果超过则拒绝合并。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstrainedResource<T> {
    /// 当前资源
    pub resource: T,
    /// 资源的最大上限
    pub max_limit: T,
}

impl<T> ConstrainedResource<T> {
    /// 创建一个新的受约束资源
    ///
    /// # 参数
    /// * `resource` - 当前资源
    /// * `max_limit` - 资源的最大上限
    ///
    /// # 返回
    /// 新的受约束资源实例
    pub fn new(resource: T, max_limit: T) -> Self {
        Self {
            resource,
            max_limit,
        }
    }

    /// 获取当前资源的引用
    pub fn resource(&self) -> &T {
        &self.resource
    }

    /// 获取最大上限的引用
    pub fn max_limit(&self) -> &T {
        &self.max_limit
    }

    /// 获取当前资源的可变引用
    pub fn resource_mut(&mut self) -> &mut T {
        &mut self.resource
    }

    /// 消费包装器，返回内部资源
    pub fn into_resource(self) -> T {
        self.resource
    }
}

impl<T: Ord> Ord for ConstrainedResource<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.resource.cmp(&other.resource)
    }
}

impl<T: Ord> PartialOrd for ConstrainedResource<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Hash for ConstrainedResource<T>
where
    T: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.resource.hash(state);
    }
}

impl<T> From<T> for ConstrainedResource<T>
where
    T: Clone,
{
    fn from(resource: T) -> Self {
        // 自身作为约束
        ConstrainedResource::new(resource.clone(), resource)
    }
}

impl<T> AsRef<T> for ConstrainedResource<T> {
    fn as_ref(&self) -> &T {
        &self.resource
    }
}

impl<T> AsMut<T> for ConstrainedResource<T> {
    fn as_mut(&mut self) -> &mut T {
        &mut self.resource
    }
}

/// A helper trait for [ShareableComputeDeviceResource] implementing BuddyCellResource using [ConstrainedResource].
pub trait BuddyCellResourceImplHelper:
    ShareableComputeDeviceResource + Eq + Ord + From<Self::ShareParam> + CanExceedLimit + Debug
{
    /// The device type associated with this resource.
    type ShareParam: ShareTarget;
    /// Find the optimal combination of shareable resources that fits within the constraints.
    fn find_optimal_combination(
        resource: &ConstrainedResource<Self>,
        _candidates: &HashMap<Self::ShareParam, usize>,
    ) -> HashMap<Self::ShareParam, usize>;
}

impl<T: BuddyCellResourceImplHelper> BuddyCellResource for ConstrainedResource<T> {
    type ChoosableShareTarget = T::ShareParam;
    type InnerResource = T;

    fn new_with_limit(current: Self::InnerResource, limit: Self::InnerResource) -> Self {
        Self::new(current, limit)
    }

    fn limit(&self) -> &Self::InnerResource {
        &self.max_limit
    }

    fn find_optimal_combination(
        &self,
        _candidates: &HashMap<Self::ChoosableShareTarget, usize>,
    ) -> HashMap<Self::ChoosableShareTarget, usize> {
        BuddyCellResourceImplHelper::find_optimal_combination(self, _candidates)
    }

    fn into_inner(self) -> Self::InnerResource {
        self.resource
    }
}
