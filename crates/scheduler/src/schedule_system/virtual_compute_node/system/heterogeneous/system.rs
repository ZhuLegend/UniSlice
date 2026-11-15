//! Defines the top-level `HeterogeneousSystem` that integrates the allocator and scheduler.
//!
//! # Usage Example
//!
//! ```rust,ignore
//! use std::time::Duration;
//! use tokio::time;
//!
//! // 创建异构系统
//! let system = HeterogeneousSystem::new(devices, predictor, batch_range);
//!
//! // 初始化设备缓存（系统启动时调用一次）
//! system.initialize_device_cache().await;
//!
//! // 启动定期权重更新任务
//! let system_clone = system.clone();
//! tokio::spawn(async move {
//!     let mut interval = time::interval(Duration::from_secs(10)); // 每10秒更新一次
//!     loop {
//!         interval.tick().await;
//!         system_clone.update_all_weights().await;
//!     }
//! });
//!
//! // 在任务处理中快速查询权重信息
//! let supporting_devices = system.get_supporting_devices("model_name");
//! let selected_device = system.select_device_by_weight("model_name", &supporting_devices);
//!
//! // 记录延迟（在任务完成后）
//! system.record_latency(device_id, "model_name", latency_seconds);
//!
//! // 执行设备命令（会自动刷新缓存）
//! system.execute_commands(commands).await.unwrap();
//! ```

use std::{
    collections::HashMap,
    iter,
    ops::RangeInclusive,
    sync::{
        Arc, LazyLock, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use compute_node::{
    compute_node::r#virtual::virtual_compute_node::{
        WorkTask, components::request_scheduler::pool_queue::POOL_UNAVAILABLE,
    },
    util::{Input, Output},
};
use dashmap::DashMap;
use futures::future::JoinAll;
use rand::Rng;
use tokio::{sync::watch, task};
#[cfg(feature = "tracing")]
use tracing::warn;
use uuid::Uuid;

use crate::{
    SchedulableSystem, Scheduler, SystemContext,
    schedule_system::{
        allocator::RankableShareTarget,
        schedulers::greedy::GreedyScheduler,
        virtual_compute_node::system::{
            ModelRequestRatePredictor,
            heterogeneous::{allocator::BidAllocator, bid::Bid, device::BoxedShareableDevice},
        },
    },
};

/// 延迟记录条目，用于存储单次任务执行的延迟信息
#[derive(Debug, Clone)]
struct LatencyRecord {
    /// 延迟时间（秒），表示从任务开始到完成的实际处理时间
    latency: f32,
    /// 记录时间戳，用于清理过期记录
    timestamp: Instant,
}

/// The main system for heterogeneous scheduling.
pub struct HeterogeneousSystem {
    devices: Arc<HashMap<Uuid, Arc<dyn BoxedShareableDevice<Target = Bid>>>>,
    queue_coordinator: QueueCoordinator,
    model_request_rate_predictor: ModelRequestRatePredictor,
    batch_size_range: RangeInclusive<u32>,
    request_timeout: Duration,
    /// 设备EMA延迟缓存：device_id -> model_name -> ema_latency
    /// 使用指数移动平均（EMA）来平滑延迟数据，更好地应对异常值
    device_ema_latency: Arc<DashMap<Uuid, DashMap<String, f32>>>,
    /// 设备权重缓存：model_name -> device_id -> weight
    /// 缓存基于延迟计算的权重，权重越高表示该设备对该模型的处理能力越强
    device_weights: Arc<DashMap<String, DashMap<Uuid, f32>>>,
    /// Successful applied bids retain the shared online predictor for routing refreshes.
    applied_allocations: Mutex<HashMap<Uuid, HashMap<String, Vec<Bid>>>>,
    /// 设备支持的模型缓存：device_id -> supported_models
    /// 缓存每个设备支持的模型列表，避免频繁查询
    device_supported_models: Arc<DashMap<Uuid, Vec<String>>>,
    /// 模型到设备的映射缓存：model_name -> supporting_device_ids
    /// 反向索引，快速查找支持特定模型的设备
    model_to_devices: Arc<DashMap<String, Vec<Uuid>>>,
    /// 延迟记录的最大保留数量
    /// 控制每个设备-模型组合的历史记录数量，避免内存无限增长
    max_latency_records: usize,
}

struct QueueCoordinator {
    started: AtomicBool,
    changes: watch::Sender<u64>,
    tasks: Mutex<Vec<tokio::task::JoinHandle<()>>>,
}

impl Default for QueueCoordinator {
    fn default() -> Self {
        Self {
            started: AtomicBool::new(false),
            changes: watch::channel(0).0,
            tasks: Mutex::new(Vec::new()),
        }
    }
}
impl Drop for QueueCoordinator {
    fn drop(&mut self) {
        for task in self.tasks.get_mut().unwrap().drain(..) {
            task.abort();
        }
    }
}

impl HeterogeneousSystem {
    /// Creates a new heterogeneous system.
    pub fn new(
        devices: HashMap<Uuid, Arc<dyn BoxedShareableDevice<Target = Bid>>>,
        model_request_rate_predictor: ModelRequestRatePredictor,
        batch_size_range: RangeInclusive<u32>,
    ) -> Self {
        Self {
            devices: Arc::new(devices),
            queue_coordinator: QueueCoordinator::default(),
            model_request_rate_predictor,
            batch_size_range,
            request_timeout: Duration::from_secs(300),
            device_ema_latency: Arc::new(DashMap::new()),
            device_weights: Arc::new(DashMap::new()),
            applied_allocations: Mutex::new(HashMap::new()),
            device_supported_models: Arc::new(DashMap::new()),
            model_to_devices: Arc::new(DashMap::new()),
            max_latency_records: 100, // 默认保留最近100个延迟记录
        }
    }

    /// Creates a new heterogeneous system with custom latency record limit.
    ///
    /// # Arguments
    /// * `devices` - 可用的计算设备映射
    /// * `model_request_rate_predictor` - 模型请求速率预测器
    /// * `batch_size_range` - 批次大小范围
    /// * `max_latency_records` - 每个设备-模型组合最多保留的延迟记录数
    pub fn with_latency_record_limit(
        devices: HashMap<Uuid, Arc<dyn BoxedShareableDevice<Target = Bid>>>,
        model_request_rate_predictor: ModelRequestRatePredictor,
        batch_size_range: RangeInclusive<u32>,
        max_latency_records: usize,
    ) -> Self {
        let mut system = Self::new(devices, model_request_rate_predictor, batch_size_range);
        system.max_latency_records = max_latency_records;
        system
    }

    /// Bound queueing and execution together; an expired future cancels its result channel.
    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        assert!(!timeout.is_zero(), "request timeout must be positive");
        self.request_timeout = timeout;
        self
    }

    fn ensure_queue_coordinator(&self) {
        if self.queue_coordinator.started.swap(true, Ordering::AcqRel) {
            return;
        }
        let queues = self
            .devices
            .iter()
            .filter_map(|(id, device)| device.request_queue().map(|queue| (*id, queue)))
            .collect::<Vec<_>>();
        let mut tasks = self.queue_coordinator.tasks.lock().unwrap();
        for (_, queue) in &queues {
            let mut events = queue.subscribe();
            let changed = self.queue_coordinator.changes.clone();
            tasks.push(tokio::spawn(async move {
                while events.changed().await.is_ok() {
                    changed.send_modify(|value| *value = value.wrapping_add(1));
                }
            }));
        }
        let mut changes = self.queue_coordinator.changes.subscribe();
        let devices = Arc::downgrade(&self.devices);
        let weights = self.device_weights.clone();
        tasks.push(tokio::spawn(async move {
            loop {
                // No source/destination locks or strong device ownership survive await.
                if let Some(devices) = devices.upgrade() {
                    Self::handoff_unavailable(&queues, &devices, &weights);
                } else {
                    break;
                }
                if changes.changed().await.is_err() {
                    break;
                }
            }
        }));
    }

    /// At startup all pools can initialize together; otherwise replace one pool
    /// while preferring models that another pool can continue serving.
    fn next_apply_wave(
        &self,
        pending: &mut HashMap<Uuid, HashMap<String, Vec<Bid>>>,
    ) -> HashMap<Uuid, HashMap<String, Vec<Bid>>> {
        let active = self
            .devices
            .iter()
            .filter_map(|(id, device)| {
                device
                    .request_queue()
                    .map(|queue| (*id, queue.serving_models()))
            })
            .collect::<HashMap<_, _>>();
        let mut candidates = pending.keys().copied().collect::<Vec<_>>();
        candidates.sort();
        if active.values().all(Vec::is_empty) {
            std::mem::take(pending)
        } else {
            let id = *candidates
                .iter()
                .min_by_key(|id| {
                    let unprotected = active
                        .get(id)
                        .into_iter()
                        .flatten()
                        .filter(|model| {
                            !active
                                .iter()
                                .any(|(other, models)| other != *id && models.contains(model))
                        })
                        .count();
                    let replacements = pending[id]
                        .iter()
                        .filter(|(model, bids)| {
                            !bids.is_empty()
                                && active
                                    .iter()
                                    .any(|(other, models)| other != *id && models.contains(model))
                        })
                        .count();
                    (unprotected, std::cmp::Reverse(replacements), **id)
                })
                .unwrap();
            #[cfg(feature = "tracing")]
            tracing::info!(target: "allocation", event = "pool_apply_started", pool_id = %id);
            HashMap::from([(id, pending.remove(&id).unwrap())])
        }
    }

    /// Move only unclaimed fragments, returning ownership if every target rejects.
    fn handoff_unavailable(
        queues: &[(Uuid, compute_node::compute_node::r#virtual::virtual_compute_node::components::request_scheduler::PoolRequestQueue)],
        devices: &HashMap<Uuid, Arc<dyn BoxedShareableDevice<Target = Bid>>>,
        weights: &DashMap<String, DashMap<Uuid, f32>>,
    ) {
        for (source, queue) in queues {
            for model in queue.models() {
                let Some(pending) = queue.take_unavailable(&model) else {
                    continue;
                };
                let mut targets = queues
                    .iter()
                    .filter_map(|(id, target)| {
                        let (serving, epoch) = target.availability(&model);
                        (*id != *source && serving).then_some((*id, epoch))
                    })
                    .collect::<Vec<_>>();
                targets.sort_by(|(a, _), (b, _)| {
                    let weight = |id| {
                        weights
                            .get(&model)
                            .and_then(|map| map.get(&id).map(|v| *v))
                            .unwrap_or(1.0)
                    };
                    weight(*b).total_cmp(&weight(*a)).then_with(|| a.cmp(b))
                });
                #[cfg(feature = "tracing")]
                let ids = pending.task_ids();
                let mut pending = Some(pending);
                for (target, epoch) in targets {
                    match devices[&target].accept_pending(pending.take().unwrap(), epoch) {
                        Ok(()) => {
                            #[cfg(feature = "tracing")]
                            let timestamp_ms = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_millis() as u64;
                            #[cfg(feature = "tracing")]
                            for id in &ids {
                                tracing::info!(target: "virtual_compute_node::task_tracker", event = "task_transferred", task_id = %id, model_id = %model, source_pool = %source, target_pool = %target, timestamp_ms);
                            }
                            break;
                        }
                        Err(returned) => pending = Some(returned),
                    }
                }
                if let Some(pending) = pending {
                    queue.restore(pending);
                }
            }
        }
    }

    /// 初始化设备支持模型的缓存
    /// 这个方法应该在系统启动时调用，以预热缓存
    pub async fn initialize_device_cache(&self) {
        self.refresh_device_model_support().await;
    }

    /// 记录设备处理任务的延迟, 并更新EMA延迟
    ///
    /// # Arguments
    /// * `device_id` - 设备ID
    /// * `model_name` - 模型名称
    /// * `latency` - 延迟时间（秒）
    pub fn record_latency(&self, device_id: Uuid, model_name: &str, latency: f32) {
        let device_ema_latency = self.device_ema_latency.clone();
        let model_name = model_name.to_string();

        task::spawn_blocking(move || {
            // --- 更新EMA延迟 ---
            const EMA_ALPHA: f32 = 0.3; // 平滑因子，越高则新数据权重越大
            let device_ema_stats = device_ema_latency.entry(device_id).or_default();
            let mut ema_latency_entry = device_ema_stats.entry(model_name).or_default();
            let current_ema = *ema_latency_entry;

            if current_ema == 0.0 {
                // 如果是第一次记录，直接使用当前延迟作为初始EMA值
                *ema_latency_entry = latency;
            } else {
                // 使用EMA公式更新
                *ema_latency_entry = (latency * EMA_ALPHA) + (current_ema * (1.0 - EMA_ALPHA));
            }
        });
    }

    /// 计算特定设备对特定模型的平均延迟（使用EMA）
    ///
    /// # Arguments
    /// * `device_id` - 设备ID
    /// * `model_name` - 模型名称
    ///
    /// # Returns
    /// 平均延迟（秒），如果没有记录则返回None
    fn calculate_average_latency(&self, device_id: Uuid, model_name: &str) -> Option<f32> {
        self.device_ema_latency
            .get(&device_id)
            .and_then(|device_stats| device_stats.get(model_name).map(|v| *v.value()))
    }

    fn allocation_weights(allocation: &HashMap<String, Vec<Bid>>) -> HashMap<String, f32> {
        allocation
            .iter()
            .filter_map(|(model, bids)| {
                if bids.is_empty() {
                    return None;
                }
                const POINTS: [u32; 4] = [1, 4, 8, 16];
                let weight = bids
                    .iter()
                    .map(|bid| {
                        POINTS
                            .iter()
                            .map(|&batch| batch as f32 / bid.evaluate_latency(model, batch))
                            .sum::<f32>()
                            / POINTS.len() as f32
                    })
                    .sum::<f32>();
                (weight.is_finite() && weight > 0.0).then(|| (model.clone(), weight))
            })
            .collect()
    }

    /// Copy successfully applied bids for passive prediction diagnostics; never refresh weights.
    #[cfg(feature = "scheduler_daemon")]
    pub(crate) fn get_applied_bids(&self, model: &str) -> HashMap<Uuid, Vec<Bid>> {
        self.applied_allocations
            .lock()
            .unwrap()
            .iter()
            .filter_map(|(&pool, allocation)| {
                allocation.get(model).map(|bids| (pool, bids.clone()))
            })
            .collect()
    }

    /// Refresh capacities from the same execution-calibrated predictors used by allocation.
    /// Request JCT EMA is diagnostic only: it includes queueing and is not execution capacity.
    pub async fn update_all_weights(&self) {
        // Serialize publication with apply results, including partial failures.
        let applied = self.applied_allocations.lock().unwrap();
        for (&device_id, allocation) in applied.iter() {
            for (model, weight) in Self::allocation_weights(allocation) {
                if self.get_supporting_devices(&model).contains(&device_id) {
                    self.device_weights
                        .entry(model)
                        .or_default()
                        .insert(device_id, weight);
                }
            }
        }
    }

    /// 刷新设备支持的模型信息（并行查询所有设备）
    ///
    /// 采用全量更新策略，避免复杂的增量更新逻辑，保证数据一致性。
    async fn refresh_device_model_support(&self) {
        // 并行查询所有设备支持的模型
        let device_models: Vec<_> = self
            .devices
            .iter()
            .map(|(device_id, device)| async move {
                let supported_models = device.support_models().await;
                (*device_id, supported_models)
            })
            .collect::<JoinAll<_>>()
            .await;

        // 清空旧的缓存
        self.device_supported_models.clear();
        self.model_to_devices.clear();

        // 重建缓存
        for (device_id, supported_models) in device_models {
            // 更新设备支持的模型缓存
            self.device_supported_models
                .insert(device_id, supported_models.clone());

            // 更新模型到设备的反向映射
            for model_name in supported_models {
                self.model_to_devices
                    .entry(model_name)
                    .or_default()
                    .push(device_id);
            }
        }
    }

    /// 获取特定模型的归一化权重（快速查询，不触发更新）
    ///
    /// 此方法直接从缓存中读取权重信息，不会触发设备查询或权重重新计算。
    /// 如果需要最新的权重信息，请先调用 `update_all_weights()`。
    ///
    /// # Arguments
    /// * `model_name` - 模型名称
    ///
    /// # Returns
    /// 归一化后的设备权重映射，权重之和为1.0
    pub fn get_normalized_weights(&self, model_name: &str) -> HashMap<Uuid, f32> {
        let model_weights = match self.device_weights.get(model_name) {
            Some(weights) => weights,
            None => return HashMap::new(),
        };

        // 计算总权重
        let total_weight: f32 = model_weights.iter().map(|entry| *entry.value()).sum();

        if total_weight <= 0.0 {
            return HashMap::new();
        }

        // 归一化权重
        model_weights
            .iter()
            .map(|entry| (*entry.key(), *entry.value() / total_weight))
            .collect()
    }

    /// 获取历史请求 JCT EMA 的倒数，仅供诊断，不表示样本服务能力。
    ///
    /// # Arguments
    /// * `device_id` - 设备ID
    /// * `model_name` - 模型名称
    ///
    /// # Returns
    /// JCT 倒数（1/s），如果没有记录则返回 0.0；路由不使用该值。
    pub fn get_device_throughput(&self, device_id: Uuid, model_name: &str) -> f32 {
        self.calculate_average_latency(device_id, model_name)
            .map(|latency| 1.0 / latency.max(0.001))
            .unwrap_or(0.0)
    }

    /// 获取所有设备对特定模型的权重信息
    ///
    /// # Arguments
    /// * `model_name` - 模型名称
    ///
    /// # Returns
    /// 设备权重映射和总权重
    pub fn get_model_device_weights(&self, model_name: &str) -> (HashMap<Uuid, f32>, f32) {
        let normalized_weights = self.get_normalized_weights(model_name);
        let total_weight = normalized_weights.values().sum();
        (normalized_weights, total_weight)
    }

    /// 在设备apply操作完成后刷新全部设备缓存
    ///
    /// 此方法已集成到 `execute_commands` 中，通常不需要手动调用。
    pub async fn refresh_all_device_cache(&self) {
        self.refresh_device_model_support().await;
        // Preserve weights from successful allocations, pruning only unsupported pairs.
        self.device_weights.retain(|model, weights| {
            let supported = self.get_supporting_devices(model);
            weights.retain(|device_id, _| supported.contains(device_id));
            !weights.is_empty()
        });
    }

    /// 获取支持特定模型的所有设备（快速查询缓存）
    ///
    /// 此方法直接从缓存中读取，速度很快。如果缓存为空，
    /// 请先调用 `initialize_device_cache()` 或 `update_all_weights()`。
    ///
    /// # Arguments
    /// * `model_name` - 模型名称
    ///
    /// # Returns
    /// 支持该模型的设备ID列表
    pub fn get_supporting_devices(&self, model_name: &str) -> Vec<Uuid> {
        let cached = self
            .model_to_devices
            .get(model_name)
            .map(|v| v.clone())
            .unwrap_or_default();
        self.devices
            .iter()
            .filter_map(|(id, device)| {
                let serving = device.request_queue().map_or_else(
                    || cached.contains(id),
                    |queue| queue.availability(model_name).0,
                );
                serving.then_some(*id)
            })
            .collect()
    }

    /// 基于权重进行加权随机选择设备
    ///
    /// 按已成功分配切片的预测样本吞吐量计算权重，再加权随机选择。
    /// 预测使用在线校正；请求 JCT EMA 只用于诊断，不参与容量权重。
    ///
    /// # Arguments
    /// * `model_name` - 模型名称
    /// * `supported_devices` - 支持该模型的设备ID列表
    ///
    /// # Returns
    /// 选择的设备ID，如果没有可用设备则返回None
    pub fn select_device_by_weight(
        &self,
        model_name: &str,
        supported_devices: &[Uuid],
    ) -> Option<Uuid> {
        self.select_device_with_draw(model_name, supported_devices, None)
    }

    fn select_device_with_draw(
        &self,
        model_name: &str,
        supported_devices: &[Uuid],
        draw: Option<f32>,
    ) -> Option<Uuid> {
        let mut stable_devices = supported_devices.to_vec();
        if draw.is_some() {
            stable_devices.sort_unstable();
        }
        let supported_devices = stable_devices.as_slice();
        if supported_devices.is_empty() {
            return None;
        }

        // 如果只有一个设备，直接返回
        if supported_devices.len() == 1 {
            return Some(supported_devices[0]);
        }

        // 获取权重信息
        let normalized_weights = self.get_normalized_weights(model_name);

        // 收集有权重的设备和对应的权重
        let mut devices_with_weights = Vec::new();
        let mut weights = Vec::new();

        for &device_id in supported_devices {
            if let Some(&weight) = normalized_weights.get(&device_id) {
                devices_with_weights.push(device_id);
                weights.push(weight);
            } else {
                // 如果设备没有权重记录，给它一个默认的小权重
                devices_with_weights.push(device_id);
                weights.push(0.001); // 很小的默认权重
            }
        }

        // 如果所有权重都为0，则均匀随机选择
        if weights.iter().all(|&w| w <= 0.0) {
            let mut rng = rand::rng();
            let index = draw.map_or_else(
                || rng.random_range(0..supported_devices.len()),
                |v| (v * supported_devices.len() as f32) as usize,
            );
            return Some(supported_devices[index]);
        }

        // 手动实现加权随机选择
        let total_weight: f32 = weights.iter().sum();
        if total_weight <= 0.0 {
            let mut rng = rand::rng();
            let index = draw.map_or_else(
                || rng.random_range(0..supported_devices.len()),
                |v| (v * supported_devices.len() as f32) as usize,
            );
            return Some(supported_devices[index]);
        }

        let mut rng = rand::rng();
        let random_value = draw.unwrap_or_else(|| rng.random::<f32>()) * total_weight;
        let mut cumulative_weight = 0.0;

        for (i, &weight) in weights.iter().enumerate() {
            cumulative_weight += weight;
            if random_value <= cumulative_weight {
                return Some(devices_with_weights[i]);
            }
        }

        // 回退到最后一个设备（理论上不应该到达这里）
        #[cfg(feature = "tracing")]
        warn!("Weighted random selection fell through, returning last device");
        devices_with_weights.last().cloned()
    }

    /// 从候选设备列表中移除指定设备
    ///
    /// # Arguments
    /// * `devices` - 设备列表
    /// * `device_to_remove` - 要移除的设备ID
    ///
    /// # Returns
    /// 移除指定设备后的新列表
    pub fn remove_device_from_candidates(devices: Vec<Uuid>, device_to_remove: Uuid) -> Vec<Uuid> {
        devices
            .into_iter()
            .filter(|&id| id != device_to_remove)
            .collect()
    }

    /// 在异构系统中执行计算任务
    ///
    /// 这个方法实现了基于延迟权重的设备选择、容错执行和重试机制。
    ///
    /// # 处理流程
    /// 1. 提取任务的模型名称和批次大小
    /// 2. 查找支持该模型的所有设备
    /// 3. 使用加权随机算法选择最优设备
    /// 4. 执行任务并记录延迟
    /// 5. 如果失败，选择其他设备重试
    ///
    /// # Arguments
    /// * `task` - 要执行的工作任务
    ///
    /// # Returns
    /// 成功时返回计算结果，失败时返回错误信息
    ///
    /// # Examples
    /// ```rust,ignore
    /// let result = system.compute(work_task).await?;
    /// ```
    pub async fn compute(&self, task: WorkTask<Input>) -> Result<WorkTask<Output>, String> {
        self.ensure_queue_coordinator();
        tokio::time::timeout(self.request_timeout, self.compute_with_admission(task))
            .await
            .map_err(|_| "request deadline exceeded while waiting for pool service".to_owned())?
    }

    async fn compute_with_admission(
        &self,
        task: WorkTask<Input>,
    ) -> Result<WorkTask<Output>, String> {
        let model_name = &task.model;

        // 提取批次大小
        let batch_size = task.work.data().dimensions().first().copied().unwrap_or(1) as u32;

        #[cfg(feature = "tracing")]
        tracing::debug!(
            "Starting compute for model: {}, batch_size: {}",
            model_name,
            batch_size
        );

        // 查找支持该模型的设备
        'admission: loop {
            let mut changed = self.queue_coordinator.changes.subscribe();
            let mut supporting_devices = self.get_supporting_devices(model_name);
            while supporting_devices.is_empty()
                && self
                    .devices
                    .values()
                    .any(|d| d.request_queue().is_some_and(|q| q.knows_model(model_name)))
            {
                changed
                    .changed()
                    .await
                    .map_err(|_| "pool availability channel closed".to_owned())?;
                supporting_devices = self.get_supporting_devices(model_name);
            }
            if supporting_devices.is_empty() {
                let error_msg = format!("No devices support model: {}", model_name);
                #[cfg(feature = "tracing")]
                warn!("{}", error_msg);
                return Err(error_msg);
            }

            // 验证批次大小是否在支持范围内
            if !self.batch_size_range.contains(&batch_size) {
                let error_msg = format!(
                    "Batch size {} is outside supported range: {:?}",
                    batch_size, self.batch_size_range
                );
                #[cfg(feature = "tracing")]
                warn!("{}", error_msg);
                return Err(error_msg);
            }

            let mut remaining_devices = supporting_devices;
            let mut last_error = String::new();

            // 重试机制：尝试不同的设备，直到成功或所有设备都失败
            while !remaining_devices.is_empty() {
                // 使用加权随机选择设备
                if let Some(selected_device_id) = self.select_device_with_draw(
                    model_name,
                    &remaining_devices,
                    compute_node::experiment_random::seed(&format!("routing/{model_name}")).map(
                        |seed| {
                            compute_node::experiment_random::uniform(
                                seed,
                                task.work.id().as_u128() as u64,
                            )
                        },
                    ),
                ) {
                    if let Some(device) = self.devices.get(&selected_device_id) {
                        let start_time = Instant::now();
                        let epoch = device
                            .request_queue()
                            .map(|queue| queue.availability(model_name).1);

                        #[cfg(feature = "tracing")]
                        tracing::debug!(
                            "Attempting compute on device: {} for model: {}",
                            selected_device_id,
                            model_name
                        );

                        // 执行计算任务
                        match device.compute(task.clone()).await {
                            Ok(result) => {
                                // 记录成功的延迟
                                let latency = start_time.elapsed().as_secs_f32();
                                if epoch
                                    == device
                                        .request_queue()
                                        .map(|queue| queue.availability(model_name).1)
                                {
                                    self.record_latency(selected_device_id, model_name, latency);
                                }

                                #[cfg(feature = "tracing")]
                                tracing::debug!(
                                    "Compute successful on device: {}, latency: {:.3}s",
                                    selected_device_id,
                                    latency
                                );

                                return Ok(result);
                            }
                            Err(err) => {
                                // A live pool rejection is safe to retry: admission never
                                // enqueued it. An execution error may have done partial work.
                                if device.request_queue().is_some() && err != POOL_UNAVAILABLE {
                                    return Err(err);
                                }
                                // 记录失败的延迟（可选，用于统计）
                                let latency = start_time.elapsed().as_secs_f32();

                                #[cfg(feature = "tracing")]
                                warn!(
                                    "Compute failed on device: {}, error: {}, latency: {:.3}s",
                                    selected_device_id, err, latency
                                );

                                last_error = format!("Device {}: {}", selected_device_id, err);

                                // 从候选列表中移除失败的设备
                                remaining_devices = Self::remove_device_from_candidates(
                                    remaining_devices,
                                    selected_device_id,
                                );
                                if remaining_devices.is_empty() && err == POOL_UNAVAILABLE {
                                    continue 'admission;
                                }
                            }
                        }
                    } else {
                        #[cfg(feature = "tracing")]
                        warn!(
                            "Selected device {} not found in device map",
                            selected_device_id
                        );

                        // 从候选列表中移除不存在的设备
                        remaining_devices = Self::remove_device_from_candidates(
                            remaining_devices,
                            selected_device_id,
                        );
                    }
                } else {
                    // 无法选择设备（权重计算问题）
                    let error_msg = format!(
                        "Failed to select device for model: {} from {} available devices",
                        model_name,
                        remaining_devices.len()
                    );
                    #[cfg(feature = "tracing")]
                    warn!("{}", error_msg);
                    return Err(error_msg);
                }
            }

            // 所有设备都失败了
            let final_error = format!(
                "All devices failed for model: {}. Last error: {}",
                model_name, last_error
            );

            #[cfg(feature = "tracing")]
            warn!("{}", final_error);

            return Err(final_error);
        }
    }

    /// 执行延迟更新
    pub async fn perform_latency_update(&self) {
        self.devices
            .values()
            .for_each(|device| device.perform_latency_update());
        self.update_all_weights().await;
    }
}

/// [HeterogeneousSystem]'s [SystemContext].
pub struct HeterogeneousSystemContext {
    allocator_creator: Box<dyn Fn() -> BidAllocator + Send + Sync>,
    model_request_rate_predictor: ModelRequestRatePredictor,
    batch_size_range: RangeInclusive<u32>,
}

impl HeterogeneousSystemContext {
    /// Fresh decision-only resource budgets; calling this does not apply a plan.
    pub(crate) fn fresh_allocator(&self) -> BidAllocator {
        (self.allocator_creator)()
    }
}

impl AsRef<HeterogeneousSystemContext> for HeterogeneousSystemContext {
    fn as_ref(&self) -> &HeterogeneousSystemContext {
        self
    }
}

impl SystemContext for HeterogeneousSystemContext {
    type State = HashMap<Uuid, HashMap<String, Vec<Bid>>>;

    fn state(&self) -> &Self::State {
        static EMPTY: LazyLock<HashMap<Uuid, HashMap<String, Vec<Bid>>>> =
            LazyLock::new(HashMap::new);
        &EMPTY
    }
}

impl SchedulableSystem for HeterogeneousSystem {
    type Command = HashMap<Uuid, HashMap<String, Vec<Bid>>>;
    type Context = HeterogeneousSystemContext;
    type Error = String;

    async fn query_context(&self) -> Self::Context {
        let devices = self.devices.values().cloned().collect::<Vec<_>>();

        let creator = move || {
            BidAllocator::from_iter(devices.clone().into_iter().map(|d| d.boxed_allocator()))
        };

        HeterogeneousSystemContext {
            allocator_creator: Box::new(creator),
            model_request_rate_predictor: self.model_request_rate_predictor.clone(),
            batch_size_range: self.batch_size_range.clone(),
        }
    }

    fn generate_command(
        _current: &Self::Context,
        target: <Self::Context as SystemContext>::State,
    ) -> impl IntoIterator<Item = Self::Command, IntoIter: Send> + Send {
        iter::once(target)
    }

    /// 执行设备命令并处理部分失败的情况
    ///
    /// 此方法会并行执行所有设备的apply操作，不论成功与否，缓存均会更新
    ///
    /// # 错误处理策略
    /// - 成功的操作：设备状态正常更新，缓存得到刷新
    /// - 失败的操作：记录错误但不影响其他设备的状态更新
    /// - 返回值：如果有任何失败，返回聚合的错误信息；如果全部成功，返回Ok(())
    ///
    /// # Arguments
    /// * `commands` - 要执行的设备命令列表
    ///
    /// # Returns
    /// * `Ok(())` - 所有操作都成功
    /// * `Err(String)` - 有操作失败，包含所有失败的详细信息
    async fn execute_commands<I: IntoIterator<Item = Self::Command> + Send>(
        &self,
        commands: I,
    ) -> Result<(), Self::Error>
    where
        <I as IntoIterator>::IntoIter: Send,
    {
        self.ensure_queue_coordinator();
        let mut errors = Vec::new();

        for command in commands {
            let mut pending = command;
            while !pending.is_empty() {
                let wave = self.next_apply_wave(&mut pending);
                let results = wave
                    .into_iter()
                    .map(|(device_id, model_allocation)| async move {
                        let Some(device) = self.devices.get(&device_id) else {
                            return (device_id, Err(format!("Device {device_id} not found")));
                        };
                        if !model_allocation
                            .values()
                            .flatten()
                            .all(|bid| bid.device_id() == device_id)
                        {
                            return (
                                device_id,
                                Err(format!(
                                    "Allocation contains bids for another device: {device_id}"
                                )),
                            );
                        }

                        (
                            device_id,
                            device
                                .apply(model_allocation.clone())
                                .await
                                .map(|()| model_allocation),
                        )
                    })
                    .collect::<JoinAll<_>>()
                    .await;

                for (device_id, result) in results {
                    let mut applied = self.applied_allocations.lock().unwrap();
                    applied.remove(&device_id);
                    // apply can fail after partially changing the device. Its old and
                    // proposed capacities are then both untrusted: use the routing fallback.
                    self.device_weights.retain(|_, weights| {
                        weights.remove(&device_id);
                        !weights.is_empty()
                    });
                    match result {
                        Ok(allocation) => {
                            for (model, weight) in Self::allocation_weights(&allocation) {
                                self.device_weights
                                    .entry(model)
                                    .or_default()
                                    .insert(device_id, weight);
                            }
                            applied.insert(device_id, allocation);
                        }
                        Err(err) => {
                            #[cfg(feature = "tracing")]
                            warn!("Device apply operation failed: {}", err);
                            errors.push(err);
                        }
                    }
                }
            }
        }

        #[cfg(feature = "tracing")]
        tracing::debug!("Refreshing device cache after apply operations");
        self.refresh_all_device_cache().await;

        // 如果有错误，返回聚合的错误信息
        if !errors.is_empty() {
            let error_msg = format!(
                "Some device operations failed ({} errors): {}",
                errors.len(),
                errors.join("; ")
            );
            #[cfg(feature = "tracing")]
            warn!("{}", error_msg);
            return Err(error_msg);
        }

        Ok(())
    }
}

impl Scheduler<HeterogeneousSystemContext> for GreedyScheduler {
    type Error = String;

    async fn schedule(
        &self,
        context: impl AsRef<HeterogeneousSystemContext>,
    ) -> Result<<HeterogeneousSystemContext as SystemContext>::State, Self::Error> {
        let context = context.as_ref();

        let allocator = (*context.allocator_creator)();
        let model_request_rate = (context.model_request_rate_predictor)().await;
        let res = GreedyScheduler::try_schedule(
            allocator,
            model_request_rate,
            context.batch_size_range.clone(),
        )?;

        Ok(res
            .into_iter()
            .flat_map(|(model_name, bids)| {
                iter::repeat(model_name).zip(bids.into_iter().map(|bid| (bid.device_id(), bid)))
            })
            .fold(HashMap::new(), |mut acc, (model_name, (device_id, bid))| {
                acc.entry(device_id)
                    .or_default()
                    .entry(model_name)
                    .or_default()
                    .push(bid);

                acc
            }))
    }
}

#[cfg(test)]
mod routing_tests;
