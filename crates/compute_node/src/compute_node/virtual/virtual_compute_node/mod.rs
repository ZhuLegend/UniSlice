//! VirtualComputeNode模块 - 虚拟计算节点的主要实现
//!
//! 这个模块提供了虚拟计算节点的主要功能，它是一个高层次的协调器，
//! 将各个组件（WorkerManager、RequestScheduler、ModelServiceManager等）
//! 组合在一起，提供统一的计算节点接口。

use std::{collections::HashMap, fmt::Debug, sync::Arc, time::Instant};

pub use base_virtual_compute_node_ext::DynBaseVirtualComputeNode;
use bincode::{Decode, Encode};
use components::{
    BatchSize, ModelServiceManager, NodeId, RequestScheduler, WorkerId, WorkerRegistry,
    worker_manager::WorkerManager,
};
use dashmap::DashMap;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tokio::sync::Semaphore;
use tracing::{Subscriber, trace};
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::Layer;
use uuid::Uuid;

use crate::{
    ComputeNode,
    compute_node::r#virtual::{BaseWork, base::Command as BaseCommand},
    util::{Input, Output, WorkState},
};

mod base_virtual_compute_node_ext;
pub mod components;
mod scheduler;

#[cfg(test)]
pub mod tests;

pub use base_virtual_compute_node_ext::RemoteBaseVirtualComputeNode;

use crate::{
    compute_device::vgpu::share_target::ShareTarget,
    compute_node::r#virtual::{
        base,
        virtual_compute_node::components::request_scheduler::batch_size_extractor::BatchExtractorStrategy,
    },
};

/// 虚拟计算节点 - 使用组件化架构的新实现
pub struct VirtualComputeNode<
    ShareParam: ShareTarget + Serialize + DeserializeOwned + Debug + Send + Sync + Encode + Decode<()>,
    DeviceState: base::DeviceState + Encode + Decode<()> + Send + Sync,
> {
    total_gpu_num: usize,

    /// Worker管理器 - 负责Worker生命周期和性能指标
    worker_manager: WorkerManager<ShareParam, DeviceState>,

    /// 请求调度器 - 负责任务调度和批处理分发
    request_scheduler: RequestScheduler,

    /// 模型服务管理器 - 负责模型管理和到达率跟踪
    model_service_manager: ModelServiceManager<ShareParam>,

    /// 状态变更信号量 - 确保同时只有一个 change_state 操作
    change_state_semaphore: Semaphore,
}

/// 节点状态信息
pub struct State<ShareParam: ShareTarget> {
    /// 模型到设备参数的分配
    pub model_allocation: HashMap<String, Vec<ShareParam>>,
    /// 设备切片信息
    pub slices: Vec<Vec<ShareParam>>,
}

impl<ShareParam: ShareTarget + Debug> Debug for State<ShareParam> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("State")
            .field("model_allocation", &self.model_allocation)
            .field("slices", &self.slices)
            .finish()
    }
}

/// 节点状态统计
#[derive(Debug)]
pub struct Status {
    /// 各模型的请求队列batch size总和
    pub model_req_batch_size: HashMap<String, usize>,
    /// 各模型的吞吐量
    pub model_throughput: HashMap<String, f32>,
}

/// 计算任务包装
#[derive(Debug, Serialize, Deserialize, Encode, Decode, Clone)]
#[serde(bound = "State: WorkState")]
pub struct WorkTask<State: WorkState> {
    /// 目标模型名称
    pub model: String,
    /// 实际的计算任务
    pub work: BaseWork<State>,
}

/// 迁移命令类型别名
type MigrationCommand<ShareParam> = (NodeId, Vec<BaseCommand<ShareParam>>);

/// 节点管理命令
pub enum Command<ShareParam: ShareTarget> {
    /// 状态变更命令
    ChangeState {
        /// 目标状态
        target_state: State<ShareParam>,
    },
    /// 状态查询命令
    QueryState,
}

impl<ShareParam: ShareTarget + Debug> Debug for Command<ShareParam> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Command::ChangeState { target_state } => f
                .debug_struct("Command::ChangeState")
                .field("target_state", target_state)
                .finish(),
            Command::QueryState => f.debug_struct("Command::QueryState").finish(),
        }
    }
}

impl<
    ShareParam: ShareTarget + Serialize + DeserializeOwned + Debug + Send + Sync + Encode + Decode<()>,
    DeviceState: base::DeviceState + Encode + Decode<()> + Send + Sync,
> VirtualComputeNode<ShareParam, DeviceState>
{
    /// 获取GPU数量
    pub fn gpu_num(&self) -> usize {
        self.total_gpu_num
    }
}

impl<
    ShareParam: ShareTarget
        + Serialize
        + DeserializeOwned
        + Debug
        + Send
        + Sync
        + Encode
        + Decode<()>
        + 'static,
    DeviceState: base::DeviceState + Encode + Decode<()> + Send + Sync + 'static,
> VirtualComputeNode<ShareParam, DeviceState>
{
    /// 创建新的虚拟计算节点
    pub fn new(
        batch_extractor_strategy: BatchExtractorStrategy,
        _latency_predictor: impl Fn(&str, &ShareParam, BatchSize) -> f32 + Send + Sync + 'static,
    ) -> Self {
        let nodes: Arc<
            DashMap<
                NodeId,
                Arc<
                    dyn DynBaseVirtualComputeNode<
                            ShareParam = ShareParam,
                            DeviceState = DeviceState,
                        >,
                >,
            >,
        > = Arc::new(DashMap::new());

        let request_scheduler = RequestScheduler::with_strategy(batch_extractor_strategy);
        // Lifecycle changes close admission synchronously before Pausing.
        let registry = Arc::new(WorkerRegistry::with_request_queue(
            request_scheduler.pool_queue(),
        ));

        // 创建Worker管理器
        let worker_manager = WorkerManager::with_shared_registry(registry.clone(), nodes);

        // 创建模型服务管理器
        let model_service_manager = ModelServiceManager::new(registry.clone());

        Self {
            total_gpu_num: 0,
            worker_manager,
            request_scheduler,
            model_service_manager,
            change_state_semaphore: Semaphore::new(1),
        }
    }

    /// 添加Worker到指定模型
    pub fn set_experiment_random_namespace(&mut self, namespace: String) {
        self.request_scheduler
            .set_random_namespace(namespace.clone());
        self.worker_manager.set_random_namespace(namespace);
    }

    /// 添加Worker到指定模型
    pub fn request_queue(&self) -> components::request_scheduler::PoolRequestQueue {
        self.request_scheduler.pool_queue()
    }

    /// Share execution measurements with the allocator's online predictor.
    pub fn set_performance_observer(
        &self,
        observer: components::worker_registry::PerformanceObserver<ShareParam>,
    ) {
        self.worker_manager
            .get_registry()
            .set_performance_observer(observer);
    }

    /// Accept pending fragments with their original completion channels.
    pub fn accept_pending(
        &self,
        pending: components::request_scheduler::PendingRequests,
        epoch: u64,
    ) -> Result<(), components::request_scheduler::PendingRequests> {
        self.request_scheduler.accept_pending(pending, epoch)
    }

    /// 添加Worker到指定模型
    pub fn add_worker(
        &self,
        worker_id: WorkerId,
        model_name: String,
        compute_node_id: NodeId,
        device_id: Uuid,
        device_params: ShareParam,
    ) -> Result<(), String> {
        self.add_worker_with_noise_slot(
            worker_id,
            model_name,
            compute_node_id,
            device_id,
            device_params,
            None,
        )
    }

    fn add_worker_with_noise_slot(
        &self,
        worker_id: WorkerId,
        model_name: String,
        compute_node_id: NodeId,
        device_id: Uuid,
        device_params: ShareParam,
        noise_slot: Option<&str>,
    ) -> Result<(), String> {
        // 获取Worker请求发送通道
        let worker_req_tx = self.request_scheduler.get_worker_request_sender();

        // 通过WorkerManager添加Worker
        self.worker_manager.add_worker_with_noise_slot(
            worker_id,
            model_name,
            compute_node_id,
            device_id,
            device_params,
            worker_req_tx,
            noise_slot,
        )
    }

    /// 移除Worker
    pub async fn remove_worker(&self, worker_id: WorkerId) -> Result<(), String> {
        let model = self
            .worker_manager
            .get_registry()
            .get_worker_info(worker_id)
            .map(|v| v.model_name.clone());
        self.request_scheduler.cleanup_worker_requests(worker_id);
        self.worker_manager.remove_worker(worker_id).await?;
        if let Some(model) = model {
            let queued_samples = self
                .request_scheduler
                .get_all_model_queue_batch_sizes()
                .get(&model)
                .copied()
                .unwrap_or(0);
            if queued_samples > 0 && self.worker_manager.get_model_workers(&model).is_empty() {
                tracing::warn!(target: "allocation", event = "model_queue_without_worker", %worker_id, %model, queued_samples);
            }
        }
        Ok(())
    }

    /// 添加虚拟计算节点
    ///
    /// 此方法会：
    /// 1. 获取节点当前的 worker 状态
    /// 2. 将节点添加到内部映射中
    /// 3. 为节点中的每个现有 worker 调用 add_worker 方法进行注册
    ///
    /// # 参数
    /// * `node_id` - 节点唯一标识符
    /// * `node` - 虚拟计算节点的动态 trait 对象
    ///
    /// # 返回值
    /// * `Ok(())` - 成功添加节点和所有 workers
    /// * `Err(String)` - 添加过程中发生错误
    pub async fn add_virtual_compute_node(
        &mut self,
        node_id: NodeId,
        node: Arc<
            dyn DynBaseVirtualComputeNode<ShareParam = ShareParam, DeviceState = DeviceState>,
        >,
    ) -> Result<(), String> {
        // 获取节点当前的状态，包含所有 workers
        let node_state = node
            .state()
            .await
            .map_err(|e| format!("Failed to get state from node {node_id}: {e}"))?;
        let gpu_num = node
            .status()
            .await
            .map_err(|e| format!("Failed to get status from node {node_id} for GPU count: {e}"))?
            .devices
            .len();

        self.total_gpu_num += gpu_num;

        // 将节点添加到内部映射
        self.worker_manager.nodes.insert(node_id, node.clone());

        // 为每个现有的 worker 调用 add_worker
        for (worker_id, worker_info) in node_state.workers {
            // 从 worker_info 中提取必要的信息
            let model_name = match worker_info.model_name {
                Some(name) => name,
                None => {
                    tracing::warn!(
                        "Worker {} has no model name, which is unexpected",
                        worker_id
                    );
                    return Err(format!(
                        "Worker {worker_id} in node {node_id} has no model name"
                    ));
                }
            };

            let device_id = worker_info.device_id;
            let device_params = worker_info.params;

            // 添加 worker 到管理中
            if let Err(e) = self.add_worker_with_noise_slot(
                worker_id,
                model_name.clone(),
                node_id,
                device_id,
                device_params,
                worker_info.experiment_noise_slot.as_deref(),
            ) {
                tracing::error!(
                    "Failed to add worker {} from node {} to management: {}",
                    worker_id,
                    node_id,
                    e
                );
                // 继续处理其他 workers，不要因为一个失败就停止
            } else {
                tracing::trace!(
                    "Successfully added worker {} (model: {}) from node {}",
                    worker_id,
                    model_name,
                    node_id
                );
            }
        }

        Ok(())
    }

    /// 获取模型状态
    fn get_state(&self) -> State<ShareParam> {
        let (model_allocation, mut slices) = self.model_service_manager.get_model_state();
        if slices.len() < self.total_gpu_num {
            slices.extend(vec![Default::default(); self.total_gpu_num - slices.len()]);
        }
        State {
            model_allocation,
            slices,
        }
    }

    /// 获取状态统计
    fn get_status(&self) -> Status {
        let model_req_batch_size = self.request_scheduler.get_all_model_queue_batch_sizes();

        // 模型吞吐量现在由各个Worker内部管理，不需要外部查询
        let model_throughput = HashMap::new();

        Status {
            model_req_batch_size,
            model_throughput,
        }
    }

    /// 清理无效的Worker请求
    fn cleanup_invalid_requests(&self) {
        let valid_workers: std::collections::HashSet<_> = self
            .worker_manager
            .get_registry()
            .get_all_worker_ids()
            .into_iter()
            .collect();

        self.request_scheduler
            .cleanup_invalid_worker_requests(&valid_workers);
    }

    /// 定期维护任务
    pub fn perform_maintenance(&self) {
        // 清理无效的Worker请求
        self.cleanup_invalid_requests();

        // 清理旧的到达率数据
        let cutoff_time = Instant::now() - std::time::Duration::from_secs(300); // 5分钟前
        self.model_service_manager
            .cleanup_old_arrival_data(cutoff_time);
    }
}

impl<
    ShareParam: ShareTarget
        + Serialize
        + DeserializeOwned
        + Debug
        + Send
        + Sync
        + Encode
        + Decode<()>
        + Ord
        + 'static,
    DeviceState: base::DeviceState + Encode + Decode<()> + Send + Sync + 'static,
> ComputeNode for VirtualComputeNode<ShareParam, DeviceState>
{
    type State = State<ShareParam>;
    type Status = Status;
    type Command = Command<ShareParam>;
    type InputData = WorkTask<Input>;
    type OutputData = WorkTask<Output>;

    async fn handle(&self, command: Self::Command) -> Result<Self::State, String> {
        match command {
            Command::ChangeState { target_state } => {
                // 获取信号量许可，确保同时只有一个 change_state 操作
                let _permit = self
                    .change_state_semaphore
                    .acquire()
                    .await
                    .map_err(|e| format!("Failed to acquire change_state semaphore: {e}"))?;

                trace!(self.total_gpu_num, ?target_state, "Changing State!");

                self.handle_change_state(target_state).await
            }
            Command::QueryState => self.state().await,
        }
    }

    async fn state(&self) -> Result<Self::State, String> {
        Ok(self.get_state())
    }

    async fn status(&self) -> Result<Self::Status, String> {
        Ok(self.get_status())
    }

    async fn compute(&self, work: Self::InputData) -> Result<Self::OutputData, String> {
        let model_name = work.model.clone();
        let base_work = work.work;

        // 通过请求调度器处理任务
        let result = self
            .request_scheduler
            .add_compute_task_on_admission(model_name.clone(), base_work, || {
                self.model_service_manager.record_task_arrival(&model_name)
            })
            .await?;

        Ok(WorkTask {
            model: model_name,
            work: result,
        })
    }
}

/// 任务日志Layer构造函数
pub fn task_tracking_layer<
    S: Subscriber + for<'span> tracing_subscriber::registry::LookupSpan<'span>,
>(
    log_dir: impl AsRef<std::path::Path>,
) -> (impl Layer<S>, WorkerGuard) {
    use components::task_tracker::TASK_TRACKER_TARGET;
    use tracing_appender::non_blocking;
    use tracing_subscriber::{filter::filter_fn, fmt};

    let file_appender = tracing_appender::rolling::daily(log_dir, "task_tracking.jsonl");
    let (non_blocking_appender, guard) = non_blocking(file_appender);

    let layer = fmt::layer()
        .json()
        .with_writer(non_blocking_appender)
        .with_filter(filter_fn(|metadata| {
            metadata.target() == TASK_TRACKER_TARGET
        }));

    (layer, guard)
}

/// 任务日志Layer构造函数
pub fn worker_perf_layer<
    S: Subscriber + for<'span> tracing_subscriber::registry::LookupSpan<'span>,
>(
    log_dir: impl AsRef<std::path::Path>,
) -> (impl Layer<S>, WorkerGuard) {
    use tracing_appender::non_blocking;
    use tracing_subscriber::{filter::filter_fn, fmt};

    let file_appender = tracing_appender::rolling::daily(log_dir, "worker_perf.jsonl");
    let (non_blocking_appender, guard) = non_blocking(file_appender);

    let layer = fmt::layer()
        .json()
        .with_writer(non_blocking_appender)
        .with_filter(filter_fn(|metadata| {
            metadata.target() == components::WORKER_PERF_TARGET
        }));

    (layer, guard)
}

/// Structured lifecycle/drain and connection observations for run analysis.
pub fn lifecycle_events_layer<
    S: Subscriber + for<'span> tracing_subscriber::registry::LookupSpan<'span>,
>(
    log_dir: impl AsRef<std::path::Path>,
) -> (impl Layer<S>, WorkerGuard) {
    let file_appender = tracing_appender::rolling::never(log_dir, "lifecycle_events.jsonl");
    let (writer, guard) = tracing_appender::non_blocking(file_appender);
    let layer = tracing_subscriber::fmt::layer()
        .json()
        .with_writer(writer)
        .with_filter(tracing_subscriber::filter::filter_fn(|m| {
            m.target() == "allocation"
                || (m.target() == "rpc_timing" && *m.level() <= tracing::Level::INFO)
        }));
    (layer, guard)
}
