use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, RwLock},
    time::Instant,
};

use uuid::Uuid;

use crate::{
    compute_device::vgpu::share_target::ShareTarget,
    compute_node::r#virtual::virtual_compute_node::components::{
        StateListenerHandle, WorkerStateNotifier,
    },
};

/// Worker唯一标识符类型别名
pub type WorkerId = Uuid;
/// 模型标识符类型别名
pub type ModelId = String;
/// 节点标识符类型别名
pub type NodeId = Uuid;

/// Successful inference measurements, excluding queue and batch-collection waits.
pub type PerformanceObserver<ShareParam> = Arc<dyn Fn(&str, &ShareParam, usize, f32) + Send + Sync>;

/// Worker的完整信息 - 数据层
pub struct WorkerInfo<ShareParam: ShareTarget> {
    /// Worker唯一标识符
    pub worker_id: WorkerId,
    /// 模型名称
    pub model_name: ModelId,
    /// 节点ID
    pub node_id: NodeId,
    /// 设备ID
    pub device_id: Uuid,
    /// 设备共享参数
    pub device_params: ShareParam,
    /// 创建时间
    pub created_at: Instant,
    /// Worker状态
    pub status: WorkerStatus,
}

impl<ShareParam: ShareTarget + Clone> Clone for WorkerInfo<ShareParam> {
    fn clone(&self) -> Self {
        Self {
            worker_id: self.worker_id,
            model_name: self.model_name.clone(),
            node_id: self.node_id,
            device_id: self.device_id,
            device_params: self.device_params.clone(),
            created_at: self.created_at,
            status: self.status,
        }
    }
}

/// Worker状态枚举
#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
pub enum WorkerStatus {
    /// 启动中
    Starting,
    /// 运行中
    Running,
    /// 暂停中，等待卸载
    Pausing,
    /// 已暂停，等待进一步指令
    Paused,
    /// 停止中
    Stopping,
    /// 已停止
    Stopped,
}

/// Owned registry membership. Its owner must finish/abort its pull task before
/// dropping this lease; hardware resource release stays with the node's proxy.
pub(super) struct WorkerRegistration<ShareParam: ShareTarget> {
    registry: Arc<WorkerRegistry<ShareParam>>,
    worker_id: WorkerId,
}

impl<ShareParam: ShareTarget> Drop for WorkerRegistration<ShareParam> {
    fn drop(&mut self) {
        self.registry.unregister_worker(self.worker_id);
    }
}

pub(super) struct WorkerTaskGuard<ShareParam: ShareTarget> {
    registry: Arc<WorkerRegistry<ShareParam>>,
    worker_id: WorkerId,
}

#[cfg(all(test, feature = "mock", feature = "vgpu"))]
mod liveness_tests {
    use super::*;
    use crate::{
        compute_device::vgpu::resource_spec::nv::mig::{A100Template, VDeviceScale},
        compute_node::r#virtual::virtual_compute_node::components::request_scheduler::{
            RequestScheduler, batch_size_extractor::BatchExtractorStrategy,
        },
    };
    #[tokio::test]
    async fn aborted_worker_task_closes_admission() {
        let scheduler =
            RequestScheduler::with_strategy(BatchExtractorStrategy::split_and_satisfy());
        let queue = scheduler.pool_queue();
        let registry = Arc::new(WorkerRegistry::with_request_queue(queue.clone()));
        let id = Uuid::new_v4();
        registry.register_worker(WorkerInfo {
            worker_id: id,
            model_name: "m".into(),
            node_id: Uuid::new_v4(),
            device_id: Uuid::new_v4(),
            device_params: VDeviceScale::A100(A100Template::Tiny),
            created_at: Instant::now(),
            status: WorkerStatus::Running,
        });
        let guard = registry.task_guard(id);
        let running = tokio::spawn(async move {
            let _guard = guard;
            std::future::pending::<()>().await;
        });
        assert!(queue.availability("m").0);
        running.abort();
        let _ = running.await;
        assert!(!queue.availability("m").0);
        assert_eq!(
            registry.get_worker_info(id).unwrap().status,
            WorkerStatus::Stopped
        );
    }
}
impl<ShareParam: ShareTarget> Drop for WorkerTaskGuard<ShareParam> {
    fn drop(&mut self) {
        self.registry
            .update_worker_status(self.worker_id, WorkerStatus::Stopped);
    }
}

/// Worker注册表 - 统一的数据存储和索引管理
pub struct WorkerRegistry<ShareParam: ShareTarget> {
    request_queue: Option<super::request_scheduler::PoolRequestQueue>,
    performance_observer: RwLock<Option<PerformanceObserver<ShareParam>>>,
    /// Worker的完整信息映射
    workers: Arc<RwLock<HashMap<WorkerId, WorkerInfo<ShareParam>>>>,
    /// 模型到Worker的索引映射
    model_to_workers: Arc<RwLock<HashMap<ModelId, HashSet<WorkerId>>>>,
    /// 设备到Worker的索引映射
    device_to_workers: Arc<RwLock<HashMap<Uuid, HashSet<WorkerId>>>>,
    /// 节点到Worker的索引映射
    node_to_workers: Arc<RwLock<HashMap<NodeId, HashSet<WorkerId>>>>,
    state_notifier: WorkerStateNotifier,
}

impl<ShareParam: ShareTarget> WorkerRegistry<ShareParam> {
    /// 创建新的Worker注册表
    pub fn new() -> Self {
        Self {
            request_queue: None,
            performance_observer: RwLock::new(None),
            workers: Default::default(),
            model_to_workers: Default::default(),
            device_to_workers: Default::default(),
            node_to_workers: Default::default(),
            state_notifier: WorkerStateNotifier::new(),
        }
    }

    /// Publish lifecycle transitions directly to the pool's admission boundary.
    pub fn with_request_queue(queue: super::request_scheduler::PoolRequestQueue) -> Self {
        queue.enable_worker_tracking();
        let mut registry = Self::new();
        registry.request_queue = Some(queue);
        registry
    }

    /// Observe valid successful inference calls from currently running workers.
    pub fn set_performance_observer(&self, observer: PerformanceObserver<ShareParam>) {
        *self.performance_observer.write().unwrap() = Some(observer);
    }

    pub(super) fn record_inference(
        &self,
        worker_id: WorkerId,
        model: &str,
        batch: usize,
        seconds: f32,
    ) where
        ShareParam: Clone,
    {
        if batch == 0 || !seconds.is_finite() || seconds <= 0.0 {
            return;
        }
        let Some(info) = self.get_worker_info(worker_id) else {
            return;
        };
        // Retiring or reassigned workers must not calibrate another model's profile.
        if info.status != WorkerStatus::Running || info.model_name != model {
            return;
        }
        let observer = self.performance_observer.read().unwrap().clone();
        if let Some(observer) = observer {
            observer(model, &info.device_params, batch, seconds);
        }
    }

    pub(super) fn register_owned(
        self: &Arc<Self>,
        info: WorkerInfo<ShareParam>,
    ) -> Result<WorkerRegistration<ShareParam>, String> {
        let worker_id = info.worker_id;
        if !self.register_worker(info) {
            return Err(format!("Worker {worker_id} already exists"));
        }
        Ok(WorkerRegistration {
            registry: self.clone(),
            worker_id,
        })
    }

    pub(super) fn task_guard(self: &Arc<Self>, worker_id: WorkerId) -> WorkerTaskGuard<ShareParam> {
        WorkerTaskGuard {
            registry: self.clone(),
            worker_id,
        }
    }

    /// 注册Worker
    pub fn register_worker(&self, worker_info: WorkerInfo<ShareParam>) -> bool {
        let worker_id = worker_info.worker_id;
        let model_name = worker_info.model_name.clone();
        let device_id = worker_info.device_id;
        let node_id = worker_info.node_id;

        // 检查是否已存在
        if self.workers.read().unwrap().contains_key(&worker_id) {
            return false;
        }

        // 存储Worker信息
        {
            let mut workers = self.workers.write().unwrap();
            if workers.contains_key(&worker_id) {
                return false;
            }
            if let Some(queue) = &self.request_queue {
                queue.worker_changed(
                    &model_name,
                    worker_id,
                    worker_info.status == WorkerStatus::Running,
                );
            }
            let status = worker_info.status;
            workers.insert(worker_id, worker_info);
            let _ = self.state_notifier.notify_state_change(worker_id, status);
        }

        // 更新索引
        self.model_to_workers
            .write()
            .unwrap()
            .entry(model_name)
            .or_default()
            .insert(worker_id);

        self.device_to_workers
            .write()
            .unwrap()
            .entry(device_id)
            .or_default()
            .insert(worker_id);

        self.node_to_workers
            .write()
            .unwrap()
            .entry(node_id)
            .or_default()
            .insert(worker_id);

        true
    }

    /// 注销Worker
    pub fn unregister_worker(&self, worker_id: WorkerId) -> Option<WorkerInfo<ShareParam>> {
        let worker_info = {
            let mut workers = self.workers.write().unwrap();
            let info = workers.remove(&worker_id)?;
            if let Some(queue) = &self.request_queue {
                queue.worker_changed(&info.model_name, worker_id, false);
            }
            let _ = self
                .state_notifier
                .notify_state_change(worker_id, WorkerStatus::Stopped);
            info
        };

        // 清理索引
        if let Some(workers) = self
            .model_to_workers
            .write()
            .unwrap()
            .get_mut(&worker_info.model_name)
        {
            workers.remove(&worker_id);
        }

        if let Some(workers) = self
            .device_to_workers
            .write()
            .unwrap()
            .get_mut(&worker_info.device_id)
        {
            workers.remove(&worker_id);
        }

        if let Some(workers) = self
            .node_to_workers
            .write()
            .unwrap()
            .get_mut(&worker_info.node_id)
        {
            workers.remove(&worker_id);
        }

        Some(worker_info)
    }

    /// 获取Worker信息
    pub fn get_worker_info(&self, worker_id: WorkerId) -> Option<WorkerInfo<ShareParam>>
    where
        ShareParam: Clone,
    {
        self.workers.read().unwrap().get(&worker_id).cloned()
    }

    /// 检查Worker是否存在
    pub fn contains_worker(&self, worker_id: WorkerId) -> bool {
        self.workers.read().unwrap().contains_key(&worker_id)
    }

    /// 获取模型的所有Worker
    pub fn get_model_workers(&self, model_id: &ModelId) -> Vec<WorkerId> {
        self.model_to_workers
            .read()
            .unwrap()
            .get(model_id)
            .map(|set| set.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// 获取设备的所有Worker
    pub fn get_device_workers(&self, device_id: Uuid) -> Vec<WorkerId> {
        self.device_to_workers
            .read()
            .unwrap()
            .get(&device_id)
            .map(|set| set.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// 获取节点的所有Worker
    pub fn get_node_workers(&self, node_id: NodeId) -> Vec<WorkerId> {
        self.node_to_workers
            .read()
            .unwrap()
            .get(&node_id)
            .map(|set| set.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// 更新Worker状态
    pub fn update_worker_status(&self, worker_id: WorkerId, status: WorkerStatus) -> bool {
        if let Some(worker_info) = self.workers.write().unwrap().get_mut(&worker_id) {
            if let Some(queue) = &self.request_queue {
                queue.worker_changed(
                    &worker_info.model_name,
                    worker_id,
                    status == WorkerStatus::Running,
                );
            }
            worker_info.status = status;
            // Publish before releasing the registry write lock. Independent
            // spawned notifications can otherwise publish Pausing after Paused.
            if let Err(error) = self.state_notifier.notify_state_change(worker_id, status) {
                tracing::warn!(%worker_id, %error, "Failed to publish worker state");
            }
            true
        } else {
            false
        }
    }

    /// 获取Worker状态监听器
    pub fn get_status_listener(&self, worker_id: WorkerId) -> StateListenerHandle {
        self.state_notifier.create_listener(worker_id)
    }

    /// 获取Worker的设备参数
    pub fn get_worker_device_params(&self, worker_id: WorkerId) -> Option<ShareParam> {
        self.workers
            .read()
            .unwrap()
            .get(&worker_id)
            .map(|info| info.device_params.clone())
    }

    /// 获取Worker的模型名称
    pub fn get_worker_model(&self, worker_id: WorkerId) -> Option<ModelId> {
        self.workers
            .read()
            .unwrap()
            .get(&worker_id)
            .map(|info| info.model_name.clone())
    }

    /// 获取Worker的节点ID
    pub fn get_worker_node(&self, worker_id: WorkerId) -> Option<NodeId> {
        self.workers
            .read()
            .unwrap()
            .get(&worker_id)
            .map(|info| info.node_id)
    }

    /// 获取所有Worker的ID列表
    pub fn get_all_worker_ids(&self) -> Vec<WorkerId> {
        self.workers.read().unwrap().keys().cloned().collect()
    }

    /// 获取所有模型的列表
    pub fn get_all_models(&self) -> Vec<ModelId> {
        self.model_to_workers
            .read()
            .unwrap()
            .keys()
            .cloned()
            .collect()
    }

    /// 获取模型的Worker数量
    pub fn get_model_worker_count(&self, model_id: &ModelId) -> usize {
        self.model_to_workers
            .read()
            .unwrap()
            .get(model_id)
            .map(|set| set.len())
            .unwrap_or(0)
    }

    /// 获取所有模型的Worker统计
    pub fn get_worker_counts_by_model(&self) -> HashMap<ModelId, usize> {
        self.model_to_workers
            .read()
            .unwrap()
            .iter()
            .map(|(model_id, workers)| (model_id.clone(), workers.len()))
            .collect()
    }

    /// 获取按状态分组的Worker
    pub fn get_workers_by_status(&self, status: WorkerStatus) -> Vec<WorkerId> {
        self.workers
            .read()
            .unwrap()
            .iter()
            .filter(|(_, info)| info.status == status)
            .map(|(worker_id, _)| *worker_id)
            .collect()
    }

    /// 清理空的索引条目
    pub fn cleanup_empty_indices(&self) {
        // 按照固定顺序获取所有写锁，避免死锁
        let mut model_to_workers = self.model_to_workers.write().unwrap();
        let mut device_to_workers = self.device_to_workers.write().unwrap();
        let mut node_to_workers = self.node_to_workers.write().unwrap();

        // 清理空的索引条目
        model_to_workers.retain(|_, workers| !workers.is_empty());
        device_to_workers.retain(|_, workers| !workers.is_empty());
        node_to_workers.retain(|_, workers| !workers.is_empty());

        // 锁在这里自动释放
    }

    /// 获取注册表统计信息
    pub fn get_statistics(&self) -> RegistryStatistics {
        let workers = self.workers.read().unwrap();
        let model_count = self.model_to_workers.read().unwrap().len();
        let device_count = self.device_to_workers.read().unwrap().len();
        let node_count = self.node_to_workers.read().unwrap().len();

        let status_counts = workers.values().fold(HashMap::new(), |mut acc, info| {
            *acc.entry(info.status).or_insert(0) += 1;
            acc
        });

        RegistryStatistics {
            total_workers: workers.len(),
            model_count,
            device_count,
            node_count,
            status_counts,
        }
    }
}

impl<ShareParam: ShareTarget> Default for WorkerRegistry<ShareParam> {
    fn default() -> Self {
        Self::new()
    }
}

/// 注册表统计信息
#[derive(Debug, Clone)]
pub struct RegistryStatistics {
    /// 总Worker数量
    pub total_workers: usize,
    /// 模型数量
    pub model_count: usize,
    /// 设备数量
    pub device_count: usize,
    /// 节点数量
    pub node_count: usize,
    /// 各状态的Worker数量统计
    pub status_counts: HashMap<WorkerStatus, usize>,
}
