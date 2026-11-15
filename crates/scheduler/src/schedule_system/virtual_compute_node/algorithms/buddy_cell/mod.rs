//! Buddy Cell Allocator
//! Use buddy cell algorithm to check whether a target sharable device can be shared from a cluster of devices.
//!

use std::{cmp::Ordering, collections::HashMap, fmt::Debug, hash::Hash};

use compute_node::compute_device::vgpu::ShareableComputeDeviceResource;
#[cfg(feature = "tracing")]
use tracing::warn;

#[cfg(test)]
mod test;

pub mod constrained;

/// A `ShareableComputeDeviceResource` that can be used in a buddy cell allocator.
///
/// When implementing this trait, ensure that the comparison logic of `Eq` and `Ord`
/// reflects the actual physical size of the resource,
/// i.e., a larger resource A and a smaller resource B must always satisfy `A >= B`.
pub trait BuddyCellResource:
    Sized + Eq + Ord + AsRef<Self::InnerResource> + AsMut<Self::InnerResource> + Debug
{
    /// The type of target that can be shared with this resource.
    type ChoosableShareTarget: Into<Self::InnerResource> + Eq + Hash + Clone + Debug;

    /// The type of resource that can be used in a buddy cell allocator.
    type InnerResource: ShareableComputeDeviceResource + Ord + CanExceedLimit + Sized;

    /// Create a new Resource with limit
    fn new_with_limit(current: Self::InnerResource, limit: Self::InnerResource) -> Self;

    /// Check if this resource can share with the target.
    fn can_share(&self, target: &Self::InnerResource) -> bool {
        self.as_ref().can_share(target)
    }

    /// Share resource with the target. if the resource can be shared, return the shared resource, else return `None`.
    fn share(&mut self, target: &Self::InnerResource) -> Option<Self::InnerResource> {
        self.as_mut().borrow(target)
    }

    /// Share all remaining resource.
    fn share_all(&mut self) -> Self::InnerResource {
        let res = *self.as_ref();
        self.share(&res).expect("should be able to borrow all")
    }

    /// The upper bound of the resource.
    fn limit(&self) -> &Self::InnerResource;

    /// Check if this resource can share with the other resource.
    ///
    /// if a resource A can compose with resource B, then resource B must be able to compose with resource A.
    fn can_compose(&self, target: &Self::InnerResource) -> bool {
        let mut res = *self.as_ref();
        res.refund(*target);
        !res.exceeds_limit(self.limit())
    }

    /// Compose two resources into a larger resource.
    /// if they can, change `self` to the composed resource and return `None`,
    /// else return the other resource back.
    fn compose(&mut self, target: Self::InnerResource) -> Option<Self::InnerResource> {
        if self.can_compose(&target) {
            self.as_mut().refund(target);
            None
        } else {
            Some(target)
        }
    }

    /// Find an optimal combination of targets that can fully utilize this resource.
    /// Accepts a HashMap of candidates and their available counts.
    /// Returns a HashMap of targets and the number of each to use.
    ///
    /// An optimal combination is one that:
    /// 1. When combined with this resource, utilizes the resource as fully as possible
    /// 2. Uses the minimum number of elements from candidates
    ///
    /// The default implementation returns an empty map, meaning no optimization.
    fn find_optimal_combination(
        &self,
        _candidates: &HashMap<Self::ChoosableShareTarget, usize>,
    ) -> HashMap<Self::ChoosableShareTarget, usize> {
        HashMap::new()
    }

    /// Check if the resource is at maximum capacity and fully utilized.
    /// The default implementation returns false.
    fn is_fully_utilized(&self) -> bool {
        self.as_ref() == self.limit()
    }

    /// Consume the resource and return the inner resource.
    fn into_inner(self) -> Self::InnerResource;
}

/// 检查资源是否超过限制的trait
///
/// 需要为具体的资源类型实现此trait，以定义如何检查资源是否超过限制
pub trait CanExceedLimit {
    /// 检查当前资源是否超过给定的限制
    ///
    /// # 参数
    /// * `limit` - 资源限制
    ///
    /// # 返回
    /// 如果当前资源超过限制返回 true，否则返回 false
    fn exceeds_limit(&self, limit: &Self) -> bool;

    /// 检查当前资源有多接近限制
    ///
    /// 返回一个0.0到1.0之间的浮点数，表示资源使用率
    /// 0.0表示资源未使用，1.0表示资源已达到限制
    fn limit_used_rate(&self, limit: &Self) -> f32;
}

/// Buddy Cell Allocator uses the buddy cell algorithm to manage resources.
#[derive(Debug)]
pub struct BuddyCellAllocator<Resource: BuddyCellResource<InnerResource: Debug + Sized>> {
    remain_resource: RemainResource<Resource>,
    choosable_targets: Vec<Resource::ChoosableShareTarget>,
    min_targets: Vec<Resource::ChoosableShareTarget>,
    limit: Resource::InnerResource,
}

impl<Resource: BuddyCellResource<InnerResource: Debug>> BuddyCellAllocator<Resource> {
    /// Create a new BuddyCellAllocator with the given resources.
    pub fn new(
        resources: Vec<Resource>,
        choosable_targets: Vec<Resource::ChoosableShareTarget>,
        min_targets: Vec<Resource::ChoosableShareTarget>,
        limit: Resource::InnerResource,
    ) -> Self
    where
        Resource::InnerResource: Sized,
    {
        Self {
            remain_resource: RemainResource::new(resources),
            choosable_targets,
            min_targets,
            limit,
        }
    }

    /// Whether a target share type can be share from current [BuddyCellAllocator].
    pub fn can_share(&self, target: Resource::ChoosableShareTarget) -> bool {
        self.remain_resource.can_share(target)
    }

    /// Try share a resource with the target.
    pub fn share(
        &mut self,
        target: Resource::ChoosableShareTarget,
    ) -> Option<Resource::InnerResource> {
        self.remain_resource.share(target)
    }

    /// Refund a resource back to the allocator.
    pub fn refund(&mut self, target: Resource::ChoosableShareTarget) -> bool {
        self.remain_resource.refund(target)
    }

    /// Returns the minimum targets that can be shared from this allocator.
    pub fn min_targets(&self) -> &[Resource::ChoosableShareTarget] {
        &self.min_targets
    }

    /// Returns an iterator over the remaining resources.
    pub fn choosable_share_targets(&self) -> Vec<Resource::ChoosableShareTarget>
    where
        Resource::ChoosableShareTarget: Clone,
    {
        self.remain_resource
            .choosable_share_targets(self.choosable_targets.clone())
    }

    /// The upper bound of the resource.
    pub fn limit(&self) -> &Resource::InnerResource {
        &self.limit
    }
}

#[derive(Debug)]
struct TrackedResource<R: BuddyCellResource> {
    resource: R,
    borrowed_chunks: Vec<R::InnerResource>,
}

impl<R: BuddyCellResource> PartialEq for TrackedResource<R> {
    fn eq(&self, other: &Self) -> bool {
        self.resource == other.resource
    }
}

impl<R: BuddyCellResource> Eq for TrackedResource<R> {}

impl<R: BuddyCellResource> PartialOrd for TrackedResource<R> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<R: BuddyCellResource> Ord for TrackedResource<R> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.resource.cmp(&other.resource)
    }
}

impl<R: BuddyCellResource> AsRef<R::InnerResource> for TrackedResource<R> {
    fn as_ref(&self) -> &R::InnerResource {
        self.resource.as_ref()
    }
}

impl<R: BuddyCellResource> AsMut<R::InnerResource> for TrackedResource<R> {
    fn as_mut(&mut self) -> &mut R::InnerResource {
        self.resource.as_mut()
    }
}

impl<R: BuddyCellResource> TrackedResource<R> {
    fn new(resource: R) -> Self {
        Self {
            resource,
            borrowed_chunks: Vec::new(),
        }
    }
}

impl<R: BuddyCellResource> BuddyCellResource for TrackedResource<R>
where
    R::InnerResource: Debug,
{
    type ChoosableShareTarget = R::ChoosableShareTarget;
    type InnerResource = R::InnerResource;

    fn new_with_limit(current: Self::InnerResource, limit: Self::InnerResource) -> Self {
        Self {
            resource: R::new_with_limit(current, limit),
            borrowed_chunks: Vec::new(),
        }
    }

    fn share(&mut self, target: &Self::InnerResource) -> Option<Self::InnerResource> {
        let res = self.resource.share(target)?;
        self.borrowed_chunks.push(res);
        Some(res)
    }

    fn limit(&self) -> &Self::InnerResource {
        self.resource.limit()
    }

    fn can_compose(&self, target: &Self::InnerResource) -> bool {
        self.resource.can_compose(target) && self.borrowed_chunks.iter().any(|r| r == target)
    }

    fn compose(&mut self, target: Self::InnerResource) -> Option<Self::InnerResource> {
        if let Some(pos) = self.borrowed_chunks.iter().position(|r| r == &target) {
            self.borrowed_chunks.remove(pos);
            self.resource.compose(target)
        } else {
            Some(target)
        }
    }

    fn find_optimal_combination(
        &self,
        candidates: &HashMap<Self::ChoosableShareTarget, usize>,
    ) -> HashMap<Self::ChoosableShareTarget, usize> {
        candidates
            .iter()
            .fold(HashMap::new(), |mut acc, (target, count)| {
                let num = self
                    .borrowed_chunks
                    .iter()
                    .filter(|r| *r == &target.clone().into())
                    .count();
                acc.insert(target.clone(), *count.min(&num));
                acc
            })
    }

    fn into_inner(self) -> Self::InnerResource {
        self.resource.into_inner()
    }
}

/// RemainResource will always keep resources in a sorted order
#[derive(Debug)]
struct RemainResource<Resource: BuddyCellResource<InnerResource: Debug>>(
    Vec<TrackedResource<Resource>>,
);

impl<Resource: BuddyCellResource<InnerResource: Debug>> RemainResource<Resource> {
    fn new(mut resources: Vec<Resource>) -> Self {
        resources.sort();
        Self(resources.into_iter().map(TrackedResource::new).collect())
    }

    fn can_share<Target: Into<Resource::InnerResource>>(&self, target: Target) -> bool {
        let target = target.into();
        self.0.iter().any(|r| r.can_share(&target))
    }

    fn share<Target: Into<Resource::InnerResource>>(
        &mut self,
        target: Target,
    ) -> Option<Resource::InnerResource> {
        let target = target.into();
        if let Some(pos) = self.0.iter().position(|r| r.can_share(&target)) {
            let min_available_resource = self.0.get_mut(pos).unwrap();
            let resource = min_available_resource.share(&target)?;
            self.0.sort();
            Some(resource)
        } else {
            None
        }
    }

    /// Refund a resource back to the allocator.
    ///
    /// Return `Some(())` if refund success, `None` if the resource list is empty after refund.
    fn refund<Target: Into<Resource::InnerResource>>(&mut self, target: Target) -> bool {
        let target = target.into();
        // use rposition here to find the largest resource that can compose with the target,
        // which will help to keep more resources `full` rather than leave many small fragments.
        if let Some(pos) = self.0.iter().rposition(|r| r.can_compose(&target)) {
            assert!(
                self.0[pos].compose(target).is_none(),
                "compose must be able to execute"
            );
        } else {
            #[cfg(feature = "tracing")]
            warn!("can't refund the resource as it's full or no suitable resource to compose with");
            return false;
        }
        self.0.sort();

        true
    }
}

impl<Resource: BuddyCellResource<InnerResource: Debug>> RemainResource<Resource>
where
    Resource::ChoosableShareTarget: Clone,
{
    /// Returns a list of all cases that can be shared from the remain resources.
    fn choosable_share_targets(
        &self,
        all_targets: impl IntoIterator<Item = Resource::ChoosableShareTarget>,
    ) -> Vec<Resource::ChoosableShareTarget> {
        all_targets
            .into_iter()
            .filter(|target| self.can_share(target.clone()))
            .collect()
    }
}

impl<'a, Resource: BuddyCellResource<InnerResource: Debug>> IntoIterator
    for &'a RemainResource<Resource>
{
    type Item = &'a Resource;
    type IntoIter = std::vec::IntoIter<&'a Resource>;

    fn into_iter(self) -> Self::IntoIter {
        self.0
            .iter()
            .map(|r| &r.resource)
            .collect::<Vec<_>>()
            .into_iter()
    }
}

struct TargetCountMap<Target: Eq + Hash>(HashMap<Target, usize>);

impl<Target: Eq + Hash> TargetCountMap<Target> {
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn get_max_key<B: Ord, F>(&self, key: F) -> Option<&Target>
    where
        Self: Sized,
        F: FnMut(&&Target) -> B,
    {
        self.0.keys().max_by_key(key)
    }

    fn get_max_satisfied_key<B: Ord, F1, F2>(&self, key: F1, filter: F2) -> Option<&Target>
    where
        Self: Sized,
        F1: FnMut(&&Target) -> B,
        F2: FnMut(&&Target) -> bool,
    {
        self.0.keys().filter(filter).max_by_key(key)
    }

    fn remove(&mut self, target: &Target, count: usize) -> bool {
        if let Some(current_count) = self.0.get_mut(target) {
            if *current_count < count {
                return false;
            }
            *current_count -= count;
            if *current_count == 0 {
                self.0.remove(target);
            }
            true
        } else {
            false
        }
    }
}

impl<Target: Eq + Hash, I: IntoIterator<Item = Target>> From<I> for TargetCountMap<Target> {
    fn from(iter: I) -> Self {
        let mut map = HashMap::new();
        for target in iter {
            *map.entry(target).or_insert(0) += 1;
        }
        Self(map)
    }
}

// 取出最大的块作为新组
fn pop_largest<Resource>(
    candidate_map: &mut TargetCountMap<Resource::ChoosableShareTarget>,
) -> Option<(Resource::InnerResource, Resource::ChoosableShareTarget)>
where
    Resource: BuddyCellResource + Clone,
{
    // 从 map 的 key 中找到最大的一个
    let largest_key = candidate_map
        .get_max_key(|item| (*item).clone().into())
        .cloned()?;

    // 更新 map
    if !candidate_map.remove(&largest_key, 1) {
        panic!("Failed to remove the largest key from candidate_map, it should always be present.");
    }

    let resource = largest_key.clone().into();
    Some((resource, largest_key))
}

// 尝试最优组合，使用预先构建的候选映射
fn try_optimal_combination<Resource>(
    current_resource: &mut Resource,
    group: &mut Vec<Resource::ChoosableShareTarget>,
    candidate_map: &mut TargetCountMap<Resource::ChoosableShareTarget>,
) where
    Resource: BuddyCellResource + Clone,
{
    let optimal_map = current_resource.find_optimal_combination(&candidate_map.0);
    // 按 optimal_map 从 unused_targets 移除对应数量的 target
    for (target, count) in optimal_map {
        if !candidate_map.remove(&target, count) {
            panic!(
                "Failed to remove the optimal combination of target and count from candidate_map."
            );
        } else {
            group.extend(vec![target; count]);
        }
    }
}

// 贪心组合，同时更新候选映射
fn greedy_combine<Resource>(
    current_resource: &mut Resource,
    group: &mut Vec<Resource::ChoosableShareTarget>,
    candidate_map: &mut TargetCountMap<Resource::ChoosableShareTarget>,
) where
    Resource: BuddyCellResource + Clone,
{
    while let Some(target) = candidate_map.get_max_satisfied_key(
        |&item| item.clone().into(),
        |&item| current_resource.can_compose(&item.clone().into()),
    ) {
        let target = target.clone();

        current_resource.compose(target.clone().into());
        group.push(target.clone());
        if !candidate_map.remove(&target, 1) {
            panic!("Candidate map should contain the target being composed.");
        }
    }
}

/// Regroup buddy cell resources from a list of targets.
pub fn regroup_buddy_cell<Resource>(
    targets: Vec<Resource::ChoosableShareTarget>,
    limit: Resource::InnerResource,
) -> Vec<Vec<Resource::ChoosableShareTarget>>
where
    Resource: BuddyCellResource + Clone,
{
    let mut groups: Vec<Vec<Resource::ChoosableShareTarget>> = Vec::new();

    // 预先构建候选映射
    let mut candidate_map: TargetCountMap<Resource::ChoosableShareTarget> = targets.into();

    while !candidate_map.is_empty() {
        // 从 map 中弹出最大的元素
        let (current_resource, largest) =
            pop_largest::<Resource>(&mut candidate_map).expect("candidate_map should not be empty");

        let mut resource = Resource::new_with_limit(current_resource, limit);

        let mut group = vec![largest];

        // 优先尝试最优组合，否则退回贪心策略
        let before_len = group.len();
        try_optimal_combination::<Resource>(&mut resource, &mut group, &mut candidate_map);
        if group.len() == before_len {
            greedy_combine::<Resource>(&mut resource, &mut group, &mut candidate_map);
        }

        groups.push(group);
    }

    groups
}

#[allow(missing_docs)]
#[cfg(any(feature = "test_utils", test))]
pub mod test_utils {
    use std::collections::HashMap;

    use compute_node::compute_device::vgpu::shareable::test::{TestResource, TestShareTarget};

    use crate::schedule_system::virtual_compute_node::algorithms::{
        buddy_cell::{BuddyCellResource, CanExceedLimit, constrained::ConstrainedResource},
        optimal_combination::GpuResourceCombinationSearcher,
    };

    impl CanExceedLimit for TestResource {
        fn exceeds_limit(&self, limit: &Self) -> bool {
            self.g > limit.g || self.gb > limit.gb
        }

        fn limit_used_rate(&self, limit: &Self) -> f32 {
            let g_rate = self.g as f32 / limit.g as f32;
            let gb_rate = self.gb as f32 / limit.gb as f32;
            g_rate.max(gb_rate)
        }
    }

    impl BuddyCellResource for ConstrainedResource<TestResource> {
        type ChoosableShareTarget = TestShareTarget;
        type InnerResource = TestResource;

        fn new_with_limit(current: Self::InnerResource, limit: Self::InnerResource) -> Self {
            Self {
                resource: current,
                max_limit: limit,
            }
        }

        fn limit(&self) -> &Self::InnerResource {
            &self.max_limit
        }

        fn find_optimal_combination(
            &self,
            candidates: &HashMap<Self::ChoosableShareTarget, usize>,
        ) -> HashMap<Self::ChoosableShareTarget, usize> {
            // 将候选者转换为搜索算法需要的格式
            let search_candidates: HashMap<TestShareTarget, (u32, u32, usize)> = candidates
                .iter()
                .map(|(&scale, &count): (&TestShareTarget, &usize)| {
                    let resource: TestResource = scale.into();
                    (scale, (resource.g as u32, resource.gb as u32, count))
                })
                .collect();

            // 使用二维搜索算法找到最优组合
            GpuResourceCombinationSearcher::search_2d_combination(
                self.max_limit.g.saturating_sub(self.resource.g) as u32,
                self.max_limit.gb.saturating_sub(self.resource.gb) as u32,
                &search_candidates,
                10, // 最大搜索深度
            )
        }

        fn into_inner(self) -> Self::InnerResource {
            self.resource
        }
    }
}
