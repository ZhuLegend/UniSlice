use std::{
    collections::HashMap,
    fmt::Debug,
    sync::Arc,
    time::{Duration, Instant},
};

use bincode::{Decode, Encode};
use dashmap::DashMap;
use serde::{Serialize, de::DeserializeOwned};
use tokio::{
    sync::{mpsc, oneshot},
    task::JoinHandle,
    time::timeout,
};
use worker::model::tensor::DynTensor;

use super::worker_registry::{
    ModelId, NodeId, WorkerId, WorkerInfo, WorkerRegistration, WorkerRegistry, WorkerStatus,
};
use crate::{
    compute_device::vgpu::share_target::ShareTarget,
    compute_node::r#virtual::{
        BaseWork, base,
        virtual_compute_node::base_virtual_compute_node_ext::DynBaseVirtualComputeNode,
    },
    util::{Input, Output},
};

/// 批大小类型别名
pub type BatchSize = usize;

/// 批量工作列表类型别名
pub type BatchWorkList = Vec<(BaseWork<Input>, oneshot::Sender<BaseWork<Output>>)>;

/// 虚拟Worker句柄
pub struct VirtualWorkerHandle<ShareParam: ShareTarget> {
    /// Worker唯一标识符
    pub worker_id: WorkerId,
    /// 模型名称
    pub model_name: ModelId,
    /// 节点ID
    pub node_id: NodeId,
    /// 异步任务句柄
    pub handle: JoinHandle<()>,
    _registration: WorkerRegistration<ShareParam>,
}

impl<ShareParam: ShareTarget> Drop for VirtualWorkerHandle<ShareParam> {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

/// Worker请求结构，用于请求批量处理任务
pub struct WorkerRequest {
    /// Worker唯一标识符
    pub worker_id: WorkerId,
    /// 模型名称
    pub model_name: ModelId,
    /// 期望的批大小
    pub desired_batch: BatchSize,
    /// 响应发送器，用于接收批量工作列表
    pub resp: oneshot::Sender<BatchWorkList>,
}

/// 批处理数据结构，用于内部处理批量数据
pub(super) struct BatchData {
    /// 输入张量列表
    pub(super) input_tensors: Vec<DynTensor>,
    /// 工作ID列表
    pub(super) work_ids: Vec<uuid::Uuid>,
    /// 各个工作的批大小列表
    pub(super) batch_sizes: Vec<usize>,
    /// 结果发送器列表
    pub(super) senders: Vec<oneshot::Sender<BaseWork<Output>>>,
}

/// Worker生命周期管理器，专注于Worker的创建、销毁和任务执行
pub struct WorkerManager<ShareParam: ShareTarget, DeviceState: base::DeviceState> {
    random_namespace: Option<String>,
    /// Worker注册表
    pub(super) registry: Arc<WorkerRegistry<ShareParam>>,
    /// 管理所有活跃的Worker及其执行句柄
    pub(super) worker_handles: DashMap<WorkerId, VirtualWorkerHandle<ShareParam>>,
    /// 底层节点的引用
    pub(in crate::compute_node::r#virtual::virtual_compute_node) nodes: Arc<
        DashMap<
            NodeId,
            Arc<dyn DynBaseVirtualComputeNode<ShareParam = ShareParam, DeviceState = DeviceState>>,
        >,
    >,
}

impl<
    ShareParam: ShareTarget + Serialize + DeserializeOwned + Debug + Send + Sync + 'static,
    DeviceState: base::DeviceState + Encode + Decode<()> + Send + Sync + 'static,
> WorkerManager<ShareParam, DeviceState>
{
    /// 创建新的 Worker 管理器
    ///
    /// # 参数
    /// * `nodes` - 底层计算节点映射
    pub fn new(
        nodes: HashMap<
            NodeId,
            Arc<dyn DynBaseVirtualComputeNode<ShareParam = ShareParam, DeviceState = DeviceState>>,
        >,
    ) -> Self {
        let registry = Arc::new(WorkerRegistry::new());

        Self {
            registry,
            random_namespace: None,
            worker_handles: DashMap::new(),
            nodes: Arc::new(DashMap::from_iter(nodes)),
        }
    }

    /// 使用共享注册表创建 Worker 管理器
    ///
    /// # 参数
    /// * `registry` - 共享的 Worker 注册表
    /// * `nodes` - 底层计算节点映射
    pub fn with_shared_registry(
        registry: Arc<WorkerRegistry<ShareParam>>,
        nodes: Arc<
            DashMap<
                NodeId,
                Arc<
                    dyn DynBaseVirtualComputeNode<
                            ShareParam = ShareParam,
                            DeviceState = DeviceState,
                        >,
                >,
            >,
        >,
    ) -> Self {
        Self {
            registry,
            random_namespace: None,
            worker_handles: DashMap::new(),
            nodes,
        }
    }

    /// 添加新的Worker
    pub(crate) fn set_random_namespace(&mut self, namespace: String) {
        self.random_namespace = Some(namespace);
    }

    /// 添加新的Worker
    pub fn add_worker(
        &self,
        worker_id: WorkerId,
        model_name: String,
        compute_node_id: NodeId,
        device_id: uuid::Uuid,
        device_params: ShareParam,
        worker_req_tx: mpsc::UnboundedSender<WorkerRequest>,
    ) -> Result<(), String> {
        self.add_worker_with_noise_slot(
            worker_id,
            model_name,
            compute_node_id,
            device_id,
            device_params,
            worker_req_tx,
            None,
        )
    }

    pub(crate) fn add_worker_with_noise_slot(
        &self,
        worker_id: WorkerId,
        model_name: String,
        compute_node_id: NodeId,
        device_id: uuid::Uuid,
        device_params: ShareParam,
        worker_req_tx: mpsc::UnboundedSender<WorkerRequest>,
        noise_slot: Option<&str>,
    ) -> Result<(), String> {
        // 验证计算节点存在
        if !self.nodes.contains_key(&compute_node_id) {
            return Err(format!("Compute node {compute_node_id} not found"));
        }

        // 在注册表中注册Worker
        let worker_info = WorkerInfo {
            worker_id,
            model_name: model_name.clone(),
            node_id: compute_node_id,
            device_id,
            device_params: device_params.clone(),
            created_at: Instant::now(),
            status: WorkerStatus::Starting,
        };
        let registration = self.registry.register_owned(worker_info)?;
        let random_seed = if self.random_namespace.is_some() && crate::experiment_random::enabled()
        {
            let slot = noise_slot.ok_or("missing explicit worker noise slot for controller")?;
            let key = format!("rounding/{slot}/{model_name}");
            let seed = crate::experiment_random::seed(&key);
            if let Some(seed) = seed {
                tracing::info!(target: "allocation", event="controller_random_seed_injected", worker_id=%worker_id, model=%model_name, profile=%device_params.name(), stream=%key, seed);
            }
            seed
        } else {
            None
        };

        // 创建并启动Worker任务（使用WorkerHandler替换原有spawn_worker_task）
        let mut handler = super::worker_handler::WorkerHandler::new(
            worker_id,
            model_name.clone(),
            compute_node_id,
            self.registry.clone(),
            self.nodes.clone(),
            worker_req_tx,
        );
        if let Some(seed) = random_seed {
            handler.set_random_seed(seed);
        }
        // Publish Running before spawning; a task that exits immediately must
        // not be marked Running again after its exit guard closes admission.
        self.registry
            .update_worker_status(worker_id, WorkerStatus::Running);
        let guard = self.registry.task_guard(worker_id);
        let handle = tokio::spawn(async move {
            let _guard = guard;
            handler.run().await;
        });

        // 保存Worker句柄
        self.worker_handles.insert(
            worker_id,
            VirtualWorkerHandle {
                worker_id,
                model_name,
                node_id: compute_node_id,
                handle,
                _registration: registration,
            },
        );

        Ok(())
    }

    /// 移除Worker
    pub async fn remove_worker(&self, worker_id: WorkerId) -> Result<(), String> {
        let mut worker = self
            .worker_handles
            .remove(&worker_id)
            .map(|(_, worker)| worker)
            .ok_or_else(|| format!("Worker {worker_id} not found"))?;
        self.registry
            .update_worker_status(worker_id, WorkerStatus::Stopping);
        match timeout(Duration::from_secs(10), &mut worker.handle).await {
            Ok(result) => {
                result.map_err(|error| format!("Worker {worker_id} task failed: {error}"))
            }
            Err(_) => {
                // Keep ownership until the task has stopped; a retry can await it.
                self.worker_handles.insert(worker_id, worker);
                Err(format!("Timeout waiting for worker {worker_id} to stop"))
            }
        }
    }

    /// 获取模型的所有Worker
    pub fn get_model_workers(&self, model_id: &ModelId) -> Vec<WorkerId> {
        self.registry.get_model_workers(model_id)
    }

    /// 获取模型的Worker句柄
    pub fn get_model_worker_handles(&self, model_id: &ModelId) -> Vec<(WorkerId, String)> {
        self.worker_handles
            .iter()
            .filter(|entry| &entry.value().model_name == model_id)
            .map(|entry| (*entry.key(), entry.value().model_name.clone()))
            .collect()
    }

    /// 获取Worker注册表的引用
    pub fn get_registry(&self) -> Arc<WorkerRegistry<ShareParam>> {
        self.registry.clone()
    }
}

#[cfg(test)]
mod ownership_tests {
    use uuid::Uuid;

    use super::*;
    use crate::compute_node::r#virtual::virtual_compute_node::tests::MockResponse;

    #[tokio::test(start_paused = true)]
    async fn shutdown_timeout_retains_registration_until_task_stops() {
        let manager = WorkerManager::<String, MockResponse>::new(HashMap::new());
        let id = uuid::Uuid::new_v4();
        let registration = manager
            .registry
            .register_owned(WorkerInfo {
                worker_id: id,
                model_name: "m".into(),
                node_id: Uuid::new_v4(),
                device_id: Uuid::new_v4(),
                device_params: "slice".into(),
                created_at: Instant::now(),
                status: WorkerStatus::Running,
            })
            .unwrap();
        let guard = manager.registry.task_guard(id);
        manager.worker_handles.insert(
            id,
            VirtualWorkerHandle {
                worker_id: id,
                model_name: "m".into(),
                node_id: Uuid::nil(),
                handle: tokio::spawn(async move {
                    let _guard = guard;
                    std::future::pending::<()>().await;
                }),
                _registration: registration,
            },
        );
        assert!(
            manager
                .remove_worker(id)
                .await
                .unwrap_err()
                .contains("Timeout")
        );
        assert!(manager.registry.contains_worker(id));
        manager.worker_handles.get(&id).unwrap().handle.abort();
        assert!(
            manager
                .remove_worker(id)
                .await
                .unwrap_err()
                .contains("task failed")
        );
        assert!(!manager.registry.contains_worker(id));
        assert!(manager.worker_handles.is_empty());
    }
}
