use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
    time::{Duration, Instant},
};

use dashmap::DashMap;

use super::worker_registry::{ModelId, WorkerId, WorkerRegistry};
use crate::compute_device::vgpu::share_target::ShareTarget;

/// 模型状态返回类型别名
pub type ModelStateResult<ShareParam> = (HashMap<String, Vec<ShareParam>>, Vec<Vec<ShareParam>>);

/// 任务到达率跟踪器
#[derive(Debug)]
pub struct ArrivalRateTracker {
    recent_arrivals: VecDeque<Instant>,
    ema_arrival_rate: f32, // 任务/秒
}

impl Default for ArrivalRateTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl ArrivalRateTracker {
    /// 创建新的任务到达率跟踪器
    pub fn new() -> Self {
        Self {
            recent_arrivals: VecDeque::new(),
            ema_arrival_rate: 0.0,
        }
    }

    pub(super) fn cleanup_old_arrival_data(&mut self, cutoff_time: Instant) {
        while let Some(&front_time) = self.recent_arrivals.front() {
            if front_time < cutoff_time {
                self.recent_arrivals.pop_front();
            } else {
                break;
            }
        }
    }

    /// 记录任务到达
    pub fn record_arrival(&mut self, now: Instant) {
        // 保留最近5秒的到达记录
        let cutoff = now - Duration::from_secs(5);
        self.cleanup_old_arrival_data(cutoff);

        self.recent_arrivals.push_back(now);

        // 计算当前到达率（过去5秒内的任务数 / 5秒）
        let current_rate = self.recent_arrivals.len() as f32 / 5.0;

        // 使用EMA更新到达率
        let alpha = 0.3; // EMA平滑因子
        self.ema_arrival_rate = alpha * current_rate + (1.0 - alpha) * self.ema_arrival_rate;
    }

    /// 获取当前到达率
    pub fn get_arrival_rate(&self) -> f32 {
        self.ema_arrival_rate
    }

    /// 获取最近的到达时间统计
    pub fn get_recent_arrivals_count(&self) -> usize {
        self.recent_arrivals.len()
    }
}

/// 模型服务管理器，专注于模型级别的业务逻辑和到达率跟踪
pub struct ModelServiceManager<ShareParam: ShareTarget> {
    /// Worker注册表的引用
    registry: Arc<WorkerRegistry<ShareParam>>,
    /// 每个模型的任务到达速率跟踪
    model_arrival_rates: DashMap<ModelId, ArrivalRateTracker>,
}

impl<ShareParam: ShareTarget> ModelServiceManager<ShareParam> {
    /// 创建新的模型服务管理器
    ///
    /// # 参数
    /// * `registry` - Worker注册表的引用
    pub fn new(registry: Arc<WorkerRegistry<ShareParam>>) -> Self {
        Self {
            registry,
            model_arrival_rates: DashMap::new(),
        }
    }

    /// 记录模型任务到达
    pub fn record_task_arrival(&self, model_id: &ModelId) {
        let arrival_time = Instant::now();
        self.model_arrival_rates
            .entry(model_id.clone())
            .or_default()
            .record_arrival(arrival_time);
    }

    /// 获取模型到达率
    pub fn get_arrival_rate(&self, model_id: &ModelId) -> Option<f32> {
        self.model_arrival_rates
            .get(model_id)
            .map(|tracker| tracker.get_arrival_rate())
    }

    /// 获取所有模型的到达率
    pub fn get_all_arrival_rates(&self) -> HashMap<ModelId, f32> {
        self.model_arrival_rates
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().get_arrival_rate()))
            .collect()
    }

    /// 获取模型的Worker列表 - 通过注册表
    pub fn get_model_workers(&self, model_id: &ModelId) -> Vec<WorkerId> {
        self.registry.get_model_workers(model_id)
    }

    /// 列出所有模型
    pub fn list_models(&self) -> Vec<ModelId> {
        self.registry.get_all_models()
    }

    /// 获取模型的Worker数量
    pub fn get_model_worker_count(&self, model_id: &ModelId) -> usize {
        self.registry.get_model_worker_count(model_id)
    }

    /// 获取所有模型的Worker统计
    pub fn get_worker_counts(&self) -> HashMap<ModelId, usize> {
        self.registry.get_worker_counts_by_model()
    }

    /// 获取模型状态信息
    pub fn get_model_state(&self) -> ModelStateResult<ShareParam> {
        let mut model_allocation = HashMap::new();
        let mut device_groups: HashMap<uuid::Uuid, Vec<_>> = HashMap::new();

        for model_id in self.registry.get_all_models() {
            let worker_ids = self.registry.get_model_workers(&model_id);
            let mut model_params = Vec::new();

            for worker_id in worker_ids {
                if let (Some(device_params), Some(worker_info)) = (
                    self.registry.get_worker_device_params(worker_id),
                    self.registry.get_worker_info(worker_id),
                ) {
                    model_params.push(device_params.clone());
                    device_groups
                        .entry(worker_info.device_id)
                        .or_default()
                        .push(device_params);
                }
            }

            if !model_params.is_empty() {
                model_allocation.insert(model_id, model_params);
            }
        }

        let device_slices = device_groups.into_values().collect();
        (model_allocation, device_slices)
    }

    /// 清理指定时间之前的到达率数据
    pub fn cleanup_old_arrival_data(&self, cutoff_time: Instant) {
        for mut tracker in self.model_arrival_rates.iter_mut() {
            tracker.cleanup_old_arrival_data(cutoff_time);
        }
    }

    /// 获取到达率跟踪器的统计信息
    pub fn get_arrival_stats(&self) -> HashMap<ModelId, (f32, usize)> {
        self.model_arrival_rates
            .iter()
            .map(|entry| {
                (
                    entry.key().clone(),
                    (
                        entry.value().get_arrival_rate(),
                        entry.value().get_recent_arrivals_count(),
                    ),
                )
            })
            .collect()
    }

    /// 重置模型的到达率跟踪
    pub fn reset_model_arrival_tracking(&self, model_id: &ModelId) {
        if let Some(mut tracker) = self.model_arrival_rates.get_mut(model_id) {
            tracker.recent_arrivals.clear();
            tracker.ema_arrival_rate = 0.0;
        }
    }

    /// 移除模型的到达率跟踪
    pub fn remove_model_arrival_tracking(&self, model_id: &ModelId) -> bool {
        self.model_arrival_rates.remove(model_id).is_some()
    }
}
