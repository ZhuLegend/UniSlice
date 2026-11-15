//! 华为 MIG 资源的约束实现

use std::collections::HashMap;

use compute_node::compute_device::vgpu::resource_spec::hw::{HuaweiMigResource, VChipScale};

use crate::schedule_system::virtual_compute_node::algorithms::{
    buddy_cell::{
        CanExceedLimit,
        constrained::{BuddyCellResourceImplHelper, ConstrainedResource},
    },
    optimal_combination::GpuResourceCombinationSearcher,
};

/// 华为 MIG 资源的约束实现
impl CanExceedLimit for HuaweiMigResource {
    fn exceeds_limit(&self, limit: &Self) -> bool {
        self.ai_core > limit.ai_core || self.ai_cpu > limit.ai_cpu || self.memory > limit.memory
    }

    fn limit_used_rate(&self, limit: &Self) -> f32 {
        let ai_core_rate = self.ai_core as f32 / limit.ai_core as f32;
        let ai_cpu_rate = self.ai_cpu as f32 / limit.ai_cpu as f32;
        let memory_rate = self.memory as f32 / limit.memory as f32;
        ai_core_rate.max(ai_cpu_rate).max(memory_rate)
    }
}

/// 华为 MIG 约束资源的类型别名
pub type ConstrainedHuaweiMigResource = ConstrainedResource<HuaweiMigResource>;

impl ConstrainedHuaweiMigResource {
    /// 根据设备最大容量创建约束资源
    ///
    /// # 参数
    /// * `current` - 当前资源
    /// * `max_ai_core` - 最大AI核心数
    /// * `max_ai_cpu` - 最大AI CPU数
    /// * `max_memory` - 最大内存（GB）
    pub fn new_with_limits(
        current: HuaweiMigResource,
        max_ai_core: u8,
        max_ai_cpu: u8,
        max_memory: u16,
    ) -> Self {
        let max_limit = HuaweiMigResource {
            ai_core: max_ai_core,
            ai_cpu: max_ai_cpu,
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
        current: HuaweiMigResource,
        device_full_capacity: HuaweiMigResource,
    ) -> Self {
        Self::new(current, device_full_capacity)
    }
}

// 为华为MIG约束资源提供特定的BuddyCellResource实现
impl BuddyCellResourceImplHelper for HuaweiMigResource {
    type ShareParam = VChipScale;

    fn find_optimal_combination(
        resource: &ConstrainedResource<Self>,
        candidates: &HashMap<Self::ShareParam, usize>,
    ) -> HashMap<Self::ShareParam, usize> {
        // 将候选者转换为搜索算法需要的格式
        let search_candidates: HashMap<VChipScale, (u32, u32, u32, usize)> = candidates
            .iter()
            .map(|(&scale, &count): (&VChipScale, &usize)| {
                let resource: HuaweiMigResource = (&scale).into();
                (
                    scale,
                    (
                        resource.ai_core as u32,
                        resource.ai_cpu as u32,
                        resource.memory as u32,
                        count,
                    ),
                )
            })
            .collect();

        // 使用三维搜索算法找到最优组合
        GpuResourceCombinationSearcher::search_3d_combination(
            resource
                .max_limit
                .ai_core
                .saturating_sub(resource.resource.ai_core) as u32,
            resource
                .max_limit
                .ai_cpu
                .saturating_sub(resource.resource.ai_cpu) as u32,
            resource
                .max_limit
                .memory
                .saturating_sub(resource.resource.memory) as u32,
            &search_candidates,
            10, // 最大搜索深度
        )
    }
}
