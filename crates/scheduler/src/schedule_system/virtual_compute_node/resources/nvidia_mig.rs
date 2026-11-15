//! 英伟达 MIG 资源的约束实现

use std::collections::HashMap;

use compute_node::compute_device::vgpu::resource_spec::nv::mig::{NvidiaMigResource, VDeviceScale};

use crate::schedule_system::virtual_compute_node::algorithms::{
    buddy_cell::{
        CanExceedLimit,
        constrained::{BuddyCellResourceImplHelper, ConstrainedResource},
    },
    optimal_combination::GpuResourceCombinationSearcher,
};

/// 英伟达 MIG 资源的约束实现
impl CanExceedLimit for NvidiaMigResource {
    fn exceeds_limit(&self, limit: &Self) -> bool {
        self.ce > limit.ce || self.memory > limit.memory
    }

    fn limit_used_rate(&self, limit: &Self) -> f32 {
        let ce_rate = self.ce as f32 / limit.ce as f32;
        let memory_rate = self.memory as f32 / limit.memory as f32;
        ce_rate.max(memory_rate)
    }
}

/// 英伟达 MIG 约束资源的类型别名
pub type ConstrainedNvidiaMigResource = ConstrainedResource<NvidiaMigResource>;

impl ConstrainedNvidiaMigResource {
    /// 根据设备最大容量创建约束资源
    ///
    /// # 参数
    /// * `current` - 当前资源
    /// * `max_ce` - 最大计算引擎数
    /// * `max_memory` - 最大内存（GB）
    pub fn new_with_limits(current: NvidiaMigResource, max_ce: u8, max_memory: u16) -> Self {
        let max_limit = NvidiaMigResource {
            ce: max_ce,
            memory: max_memory,
        };
        Self::new(current, max_limit)
    }

    /// 使用设备的完整容量作为限制
    ///
    /// # 参数
    /// * `current` - 当前资源
    /// * `device_full_capacity` - 设备的完整容量
    pub fn new_with_device_capacity(
        current: NvidiaMigResource,
        device_full_capacity: NvidiaMigResource,
    ) -> Self {
        Self::new(current, device_full_capacity)
    }
}

// 为英伟达MIG约束资源提供特定的BuddyCellResource实现
impl BuddyCellResourceImplHelper for NvidiaMigResource {
    type ShareParam = VDeviceScale;

    fn find_optimal_combination(
        resource: &ConstrainedResource<Self>,
        candidates: &HashMap<Self::ShareParam, usize>,
    ) -> HashMap<Self::ShareParam, usize> {
        // 将候选者转换为搜索算法需要的格式
        let search_candidates: HashMap<VDeviceScale, (u32, u32, usize)> = candidates
            .iter()
            .map(|(&scale, &count): (&VDeviceScale, &usize)| {
                let resource: NvidiaMigResource = (&scale).into();
                (scale, (resource.ce as u32, resource.memory as u32, count))
            })
            .collect();

        // 使用二维搜索算法找到最优组合
        GpuResourceCombinationSearcher::search_2d_combination(
            resource.max_limit.ce.saturating_sub(resource.resource.ce) as u32,
            resource
                .max_limit
                .memory
                .saturating_sub(resource.resource.memory) as u32,
            &search_candidates,
            10, // 最大搜索深度
        )
    }
}
