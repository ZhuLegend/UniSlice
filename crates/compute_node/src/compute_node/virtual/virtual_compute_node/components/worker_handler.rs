//! The `WorkerHandler` module defines the [WorkerHandler] struct,
//! which is responsible for managing the full lifecycle
//! and task loop of a single worker in a virtual compute node.
//!
//! `WorkerHandler` coordinates the:
//! - execution of tasks assigned to a worker
//! - handles communication with the worker registry
//! - monitors performance metrics (integrated performance monitoring)
//! - manages interactions with compute devices
//! - adaptive batch size control based on historical data and arrival rates
//!
//! It ensures that each worker is properly registered, receives tasks,
//! processes input and output data, and updates its status and performance information.
//! Each worker autonomously adjusts its batch size based on its own performance history,
//! arrival rate estimation, and intelligent batch size modeling.
//!
//! The struct interacts with other components such as:
//! - `WorkerRegistry`: for tracking worker status and registration.
//! - Compute devices: for executing assigned tasks.
//! - Task trackers and managers: for handling task assignment and completion.
//!
//! This module is intended for internal use within the virtual compute node implementation and is not exposed as part of the public API.

use std::{
    fmt::Debug,
    sync::Arc,
    time::{Duration, Instant},
};

use bincode::{Decode, Encode};
use dashmap::DashMap;
use serde::{Serialize, de::DeserializeOwned};
use tokio::sync::{mpsc, oneshot};
use tracing::{trace, warn};
use worker::model::tensor::DynTensor;

use super::{
    global_task_tracker,
    worker_manager::{BatchData, BatchWorkList, WorkerRequest},
    worker_registry::{ModelId, NodeId, WorkerId, WorkerRegistry, WorkerStatus},
};
use crate::{
    compute_device::vgpu::share_target::ShareTarget,
    compute_node::r#virtual::{
        BaseWork, base, base::Work,
        virtual_compute_node::base_virtual_compute_node_ext::DynBaseVirtualComputeNode,
    },
    util::{Input, Output},
};

pub const WORKER_PERF_TARGET: &str = "WorkerHandlerPerf";

#[cfg(feature = "mock")]
fn parse_fixed_experiment_batch(value: Option<&str>, model: &str) -> Result<Option<usize>, String> {
    let Some(value) = value else { return Ok(None) };
    let targets: std::collections::BTreeMap<String, usize> =
        serde_json::from_str(value).map_err(|error| error.to_string())?;
    if targets.is_empty() || targets.values().any(|batch| !(1..=32).contains(batch)) {
        return Err("fixed targets must be in 1..=32".into());
    }
    targets
        .get(model)
        .copied()
        .map(Some)
        .ok_or_else(|| format!("missing fixed target for {model}"))
}

#[cfg(all(test, feature = "mock"))]
mod fixed_batch_experiment_tests {
    use super::parse_fixed_experiment_batch;

    #[test]
    fn fixed_targets_are_explicit_and_validated() {
        assert_eq!(parse_fixed_experiment_batch(None, "m").unwrap(), None);
        assert_eq!(
            parse_fixed_experiment_batch(Some(r#"{"m":7}"#), "m").unwrap(),
            Some(7)
        );
        for invalid in [
            r#"{}"#,
            r#"{"m":0}"#,
            r#"{"m":33}"#,
            r#"{"other":2}"#,
            r#"{"m":true}"#,
        ] {
            assert!(parse_fixed_experiment_batch(Some(invalid), "m").is_err());
        }
    }
}

/// Worker处理器，负责单个Worker的完整生命周期和任务循环
pub(super) struct WorkerHandler<ShareParam: ShareTarget, DeviceState: base::DeviceState> {
    worker_id: WorkerId,
    model_name: ModelId,
    compute_node_id: NodeId,
    registry: Arc<WorkerRegistry<ShareParam>>,
    nodes: Arc<
        DashMap<
            NodeId,
            Arc<dyn DynBaseVirtualComputeNode<ShareParam = ShareParam, DeviceState = DeviceState>>,
        >,
    >,
    worker_req_tx: mpsc::UnboundedSender<WorkerRequest>,
    /// 集成的性能监控和批大小控制器
    performance_controller: PerformanceController,
    experiment_fixed_batch: bool,
}

impl<
    ShareParam: ShareTarget + Serialize + DeserializeOwned + Debug + Send + Sync,
    DeviceState: base::DeviceState + Encode + Decode<()> + Send + Sync,
> WorkerHandler<ShareParam, DeviceState>
{
    pub(super) fn set_random_seed(&mut self, seed: u64) {
        self.performance_controller.random_stream =
            Some(crate::experiment_random::SeededStream::new(seed));
    }
    /// 创建新的Worker处理器
    pub(super) fn new(
        worker_id: WorkerId,
        model_name: ModelId,
        compute_node_id: NodeId,
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
        worker_req_tx: mpsc::UnboundedSender<WorkerRequest>,
    ) -> Self {
        #[cfg(feature = "mock")]
        let fixed_batch = parse_fixed_experiment_batch(
            std::env::var("HSGS_EXPERIMENT_FIXED_BATCHES")
                .ok()
                .as_deref(),
            &model_name,
        )
        .expect("invalid fixed-batch simulation experiment");
        #[cfg(not(feature = "mock"))]
        let fixed_batch: Option<usize> = None;
        let mut performance_controller = PerformanceController::new();
        if let Some(batch) = fixed_batch {
            performance_controller.current_batch_size = batch;
        }
        #[cfg(feature = "mock")]
        tracing::info!(target: "batch_ablation", event = "batch_policy",
            worker_id = %worker_id, model = %model_name,
            fixed_target_batch = fixed_batch.unwrap_or(0));
        Self {
            worker_id,
            model_name,
            compute_node_id,
            registry,
            nodes,
            worker_req_tx,
            performance_controller,
            experiment_fixed_batch: fixed_batch.is_some(),
        }
    }

    /// Worker的主任务循环
    pub(super) async fn run(&mut self) {
        let mut state_listener = self.registry.get_status_listener(self.worker_id);

        loop {
            // 从注册表获取当前状态，这将决定worker的行为
            let current_status = self
                .registry
                .get_worker_info(self.worker_id)
                .map(|info| info.status);

            match current_status {
                Some(WorkerStatus::Running) => {
                    // 状态为Running，正常处理任务
                }
                Some(WorkerStatus::Pausing) => {
                    // 状态为Pausing，完成当前任务后转换为Paused状态
                    // 如果当前没有正在执行的任务，立即转换为Paused
                    self.registry
                        .update_worker_status(self.worker_id, WorkerStatus::Paused);

                    trace!(
                        "Worker {} transitioned from Pausing to Paused",
                        self.worker_id
                    );
                    continue;
                }
                Some(WorkerStatus::Paused) => {
                    // 状态为Paused，等待进一步指令，不接受新任务
                    let _ = state_listener
                        .wait_for_status_change(Some(Duration::from_secs(30)))
                        .await;
                    continue;
                }
                Some(WorkerStatus::Stopping) => {
                    self.registry
                        .update_worker_status(self.worker_id, WorkerStatus::Stopped);
                    break;
                }
                None => {
                    break;
                }
                Some(WorkerStatus::Starting) => {
                    // 其他状态（如Starting），等待状态变为Running
                    let _ = state_listener
                        .wait_for_status_change(Some(Duration::from_secs(30)))
                        .await;
                    continue;
                }
                Some(WorkerStatus::Stopped) => {
                    // 已停止，退出循环
                    break;
                }
            }

            // 获取当前批量大小
            let current_batch = self.performance_controller.get_current_batch_size();

            let loop_start_time = Instant::now();

            // Keep ownership of the receiver when pause races with assignment.
            let Some(mut batch_rx) = self.request_batch_work(current_batch) else {
                break;
            };
            let works = tokio::select! {
                biased;
                batch = &mut batch_rx => match batch {
                    Ok(works) => works,
                    Err(_) => continue,
                },
                _ = state_listener.wait_for_status_change(None) => {
                    // Closing prevents new assignments. Drain any batch sent
                    // between polling the receiver and observing the pause;
                    // later sends fail and the scheduler returns them to its queue.
                    batch_rx.close();
                    match batch_rx.try_recv() {
                        Ok(works) => works,
                        Err(_) => continue,
                    }
                },
            };

            let idle_time = loop_start_time.elapsed().as_secs_f32();

            if works.is_empty() {
                warn!("Worker {} received empty batch work", self.worker_id);
                continue;
            }

            let instant = Instant::now();

            // 准备批处理输入数据
            let batch_data = Self::prepare_batch_input(works);

            trace!(
                "Worker {} processing batch of size {} for model {}, use: {:?}",
                self.worker_id,
                batch_data.batch_sizes.iter().sum::<usize>(),
                self.model_name,
                instant.elapsed()
            );

            // 记录任务分配事件
            let total_batch_size: usize = batch_data.batch_sizes.iter().sum();
            global_task_tracker().track_batch_tasks_assigned(
                &batch_data.work_ids,
                self.worker_id,
                &self.model_name,
                total_batch_size,
            );

            // 执行推理并处理结果
            let processing_start = Instant::now();
            if let Some((processing_end, total_batch_size)) = self
                .execute_inference_and_distribute_results(batch_data, processing_start)
                .await
            {
                self.update_performance_metrics(
                    idle_time,
                    processing_start,
                    processing_end,
                    total_batch_size,
                );

                let is_underloaded = total_batch_size < current_batch;
                self.update_batch_size_controller(is_underloaded);
            }
        }
    }

    /// 更新批大小控制器
    fn update_batch_size_controller(&mut self, is_underloaded: bool) {
        if self.experiment_fixed_batch {
            return;
        }
        self.performance_controller
            .update_batch_size(is_underloaded);
    }

    /// 获取批量任务
    fn request_batch_work(&self, desired_batch: usize) -> Option<oneshot::Receiver<BatchWorkList>> {
        let (tx, rx) = oneshot::channel();
        let req = WorkerRequest {
            model_name: self.model_name.to_string(),
            worker_id: self.worker_id,
            desired_batch,
            resp: tx,
        };

        if self.worker_req_tx.send(req).is_err() {
            warn!("Worker request channel unexpectedly closed");
            return None;
        }

        Some(rx)
    }

    /// 准备批处理输入数据
    fn prepare_batch_input(works: BatchWorkList) -> BatchData {
        let len = works.len();

        let (input_tensors, work_ids, batch_sizes, senders) = works.into_iter().fold(
            (
                Vec::with_capacity(len),
                Vec::with_capacity(len),
                Vec::with_capacity(len),
                Vec::with_capacity(len),
            ),
            |(mut input_tensors, mut work_ids, mut batch_sizes, mut senders), (work, sender)| {
                work_ids.push(work.id());
                batch_sizes.push(work.data().dimensions()[0] as usize);
                senders.push(sender);
                input_tensors.push(work.data);

                (input_tensors, work_ids, batch_sizes, senders)
            },
        );

        BatchData {
            input_tensors,
            work_ids,
            batch_sizes,
            senders,
        }
    }

    /// 执行推理并分发结果
    async fn execute_inference_and_distribute_results(
        &self,
        batch_data: BatchData,
        processing_start: Instant,
    ) -> Option<(Instant, usize)> {
        let total_batch_size: usize = batch_data.batch_sizes.iter().sum();

        // 合并为批量张量
        let batched_input = match DynTensor::concat_batch(batch_data.input_tensors) {
            Ok(tensor) => tensor,
            Err(e) => {
                warn!("Concat batch batch input failed: {e:?}");
                return None;
            }
        };

        // 调用指定节点执行推理
        let node = self.nodes.get(&self.compute_node_id)?;
        let base_work = BaseWork::<Input>::new(self.worker_id, batched_input);
        let batch_work = Work {
            target_worker: self.worker_id,
            work: base_work,
        };

        let output = match node.compute(batch_work).await {
            Ok(res) => res,
            Err(e) => {
                warn!("Failed to compute on node {}: {}", self.compute_node_id, e);
                global_task_tracker().track_batch_compute_failed(
                    &batch_data.work_ids,
                    self.worker_id,
                    &self.model_name,
                    e,
                );
                return None;
            }
        };
        let processing_end = Instant::now();
        let (_, result_tensor) = output.work.take();

        // 记录计算完成事件
        let processing_duration_ms =
            processing_end.duration_since(processing_start).as_millis() as u64;
        global_task_tracker().track_batch_compute_completed(
            &batch_data.work_ids,
            self.worker_id,
            &self.model_name,
            total_batch_size,
            processing_duration_ms,
        );

        // 分发结果
        tokio::task::spawn(async move {
            Self::distribute_batch_results(
                result_tensor,
                batch_data.batch_sizes,
                batch_data.work_ids,
                batch_data.senders,
            );
        });

        Some((processing_end, total_batch_size))
    }

    /// 分发批处理结果给各个请求
    fn distribute_batch_results(
        result_tensor: DynTensor,
        batch_sizes: Vec<usize>,
        work_ids: Vec<uuid::Uuid>,
        senders: Vec<oneshot::Sender<BaseWork<Output>>>,
    ) {
        // 计算分割点
        let mut split_points = Vec::new();
        let mut sum = 0;
        // 迭代除最后一个元素外的所有 size
        for size in batch_sizes.iter().take(batch_sizes.len().saturating_sub(1)) {
            sum += size;
            split_points.push(sum);
        }

        match result_tensor.split_batch(split_points) {
            Ok(result_tensors) => {
                assert_eq!(result_tensors.len(), batch_sizes.len());
                assert_eq!(result_tensors.len(), work_ids.len());
                assert_eq!(result_tensors.len(), senders.len());
                for ((work_id, sender), result_tensor) in
                    work_ids.into_iter().zip(senders).zip(result_tensors)
                {
                    let output_work = BaseWork::<Output>::new(work_id, result_tensor);
                    let _ = sender.send(output_work);
                }
            }
            Err(e) => warn!("Failed to split batch results, {e:?}"),
        };
    }

    /// 更新性能指标
    fn update_performance_metrics(
        &mut self,
        idle_time: f32,
        processing_start: Instant,
        processing_end: Instant,
        total_batch_size: usize,
    ) {
        let processing_time = processing_end
            .duration_since(processing_start)
            .as_secs_f32();
        let throughput = total_batch_size as f32 / processing_time.max(0.001);

        self.registry.record_inference(
            self.worker_id,
            &self.model_name,
            total_batch_size,
            processing_time,
        );

        self.performance_controller.update_processing_completed(
            throughput,
            idle_time,
            processing_time,
            processing_end,
            total_batch_size,
        );

        trace!(
            target: WORKER_PERF_TARGET,
            worker_id = %self.worker_id,
            model_name = %self.model_name,
            batch_size = total_batch_size,
            processing_time_ms = processing_time * 1000.0,
            idle_time_ms = idle_time * 1000.0,
            throughput_tps = throughput, // samples/s; retain the historical log field name
        );
    }
}

/// 内部Worker性能指标（仅用于算法内部）
#[derive(Debug, Clone)]
struct WorkerMetrics {
    /// 最后一次完成任务的时间
    last_completion_time: Option<Instant>,
    /// 指数移动平均物理吞吐量（samples/s）
    ema_throughput: f32,
    /// 指数移动平均有效吞吐量（samples/s）
    ema_effective_throughput: f32,
    /// 指数移动平均空闲时间（秒）
    ema_idle_time: f32, // 秒
    /// 指数移动平均处理时间（秒/批次）
    ema_processing_time: f32, // 秒/批次
    /// 最后一次空闲时间（秒）
    last_idle_time: f32, // 秒
    /// 总处理任务数量
    total_tasks_processed: u64,
    /// 总处理时间
    total_processing_time: f32,
    /// 平均批大小
    average_batch_size: f32,
}

impl WorkerMetrics {
    /// 创建新的 Worker 指标实例
    fn new() -> Self {
        Self {
            last_completion_time: None,
            ema_throughput: 0.0,
            ema_effective_throughput: 0.0,
            ema_idle_time: 0.0,
            ema_processing_time: 0.1,
            last_idle_time: 0.0,
            total_tasks_processed: 0,
            total_processing_time: 0.0,
            average_batch_size: 1.0,
        }
    }
}

/// 随机化取整
/// 将一个浮点数 f 概率性地取整为 floor(f) 或 ceil(f)
/// 例如，3.5 有 50% 概率变为 3，50% 概率变为 4
fn stochastic_round(
    value: f32,
    random_stream: Option<&crate::experiment_random::SeededStream>,
) -> usize {
    if value <= 0.0 {
        return 0;
    }
    let floor_value = value.floor();
    let remainder = value.fract(); // 获取小数部分
    let random_val: f32 = random_stream.map_or_else(rand::random, |stream| stream.next_f32());

    if random_val < remainder {
        (floor_value + 1.0) as usize
    } else {
        floor_value as usize
    }
}

/// 集成的性能控制器，包含性能监控和智能批大小控制
#[derive(Debug)]
struct PerformanceController {
    random_stream: Option<crate::experiment_random::SeededStream>,
    /// Worker性能指标
    metrics: WorkerMetrics,

    /// 当前批大小
    current_batch_size: usize,

    // AIMD 配置参数
    /// 当idle_time低于此阈值时，可能表示任务堆积
    congestion_idle_threshold: f32,
    /// 连续出现拥塞的次数
    consecutive_congestion_count: u32,
    /// 触发加性递增所需的连续拥塞次数
    congestion_threshold_count: u32,
    /// 连续出现负载不足的次数
    consecutive_underload_count: u32,
    /// 触发乘性递减所需的连续负载不足次数
    underload_threshold_count: u32,
    /// 加性递增的步长
    additive_increase_step: usize,
    /// 乘性递减的因子
    multiplicative_decrease_factor: f32,

    /// 慢启动状态标志
    is_in_slow_start: bool,

    /// 长时间空闲阈值，用于决定何时重置回慢启动状态
    restart_idle_threshold: f32,

    /// 配置参数
    min_batch_size: usize,
    max_batch_size: usize,

    // 周期性批大小更新控制
    /// 任务完成计数器
    task_counter: u32,
    /// 批大小更新间隔（任务完成次数）
    update_interval: u32,
    /// 空闲时间缓冲区
    idle_time_buffer: Vec<f32>,
    /// 负载不足缓冲区
    underload_buffer: Vec<bool>,
}

impl PerformanceController {
    fn new() -> Self {
        Self {
            metrics: WorkerMetrics::new(),
            random_stream: None,
            current_batch_size: 4, // 初始批大小

            // 初始化AIMD参数
            congestion_idle_threshold: 0.005, // 5ms
            consecutive_congestion_count: 0,
            congestion_threshold_count: 1, // 连续2次低空闲时间则增加batch
            consecutive_underload_count: 0,
            underload_threshold_count: 3, // 连续2次高空闲时间则减少batch
            additive_increase_step: 4,
            multiplicative_decrease_factor: 0.9,

            // 默认启动时进入慢启动模式
            is_in_slow_start: true,

            // 长时间空闲阈值，用于重置慢启动
            restart_idle_threshold: 1.5, // 1.5秒

            min_batch_size: 1,
            max_batch_size: 32,

            // 初始化周期性批大小更新参数
            task_counter: 0,
            update_interval: 2, // 每2次任务完成更新一次批大小
            idle_time_buffer: Vec::with_capacity(2),
            underload_buffer: Vec::with_capacity(2),
        }
    }

    /// 获取当前批大小
    fn get_current_batch_size(&self) -> usize {
        self.current_batch_size
    }

    /// 更新处理完成的指标
    fn update_processing_completed(
        &mut self,
        throughput: f32,
        idle_time: f32,
        processing_time: f32,
        completion_time: Instant,
        batch_size: usize,
    ) {
        const ALPHA: f32 = 0.3; // 平滑因子

        // 更新物理吞吐率等基本指标
        self.metrics.ema_throughput =
            ALPHA * throughput + (1.0 - ALPHA) * self.metrics.ema_throughput;
        self.metrics.ema_idle_time = ALPHA * idle_time + (1.0 - ALPHA) * self.metrics.ema_idle_time;
        self.metrics.ema_processing_time =
            ALPHA * processing_time + (1.0 - ALPHA) * self.metrics.ema_processing_time;
        self.metrics.last_idle_time = idle_time;
        self.metrics.last_completion_time = Some(completion_time);

        // 计算并更新有效吞吐率
        let effective_throughput = (batch_size as f32) / (processing_time + idle_time).max(0.001);
        self.metrics.ema_effective_throughput =
            ALPHA * effective_throughput + (1.0 - ALPHA) * self.metrics.ema_effective_throughput;

        // 更新累计统计
        self.metrics.total_tasks_processed += batch_size as u64;
        self.metrics.total_processing_time += processing_time;

        // 更新平均批次大小
        let batch_alpha = 0.1;
        self.metrics.average_batch_size =
            batch_alpha * batch_size as f32 + (1.0 - batch_alpha) * self.metrics.average_batch_size;
    }

    /// 更新批大小（智能调整）
    fn update_batch_size(&mut self, is_underloaded: bool) {
        // 将当前数据存入缓冲区
        self.idle_time_buffer.push(self.metrics.last_idle_time);
        self.underload_buffer.push(is_underloaded);
        self.task_counter += 1;

        // 检查是否达到更新间隔
        if self.task_counter >= self.update_interval {
            // 计算平均值
            let avg_idle_time =
                self.idle_time_buffer.iter().sum::<f32>() / self.idle_time_buffer.len() as f32;
            let underload_ratio = self.underload_buffer.iter().filter(|&&x| x).count() as f32
                / self.underload_buffer.len() as f32;

            // 基于平均值执行批大小调整
            self.perform_batch_size_update(avg_idle_time, underload_ratio > 0.5);

            // 重置缓冲区和计数器
            self.idle_time_buffer.clear();
            self.underload_buffer.clear();
            self.task_counter = 0;
        }
    }

    /// 基于平均数据执行批大小更新
    fn perform_batch_size_update(&mut self, avg_idle_time: f32, is_underloaded: bool) {
        let mut new_batch_size = self.current_batch_size;
        let reason: &str;

        if self.is_in_slow_start {
            // 当前处于"慢启动"阶段
            if avg_idle_time < self.congestion_idle_threshold {
                // 负载很高，进行指数增长
                new_batch_size = self.current_batch_size * 2;
                reason = "慢启动 (指数增长)";
            } else {
                // 第一次遇到非拥塞信号（空闲或负载不足），退出慢启动
                self.is_in_slow_start = false;
                // 如果是负载不足，则正常执行乘性递减
                if is_underloaded {
                    let fractional_batch_size =
                        self.current_batch_size as f32 * self.multiplicative_decrease_factor;
                    new_batch_size =
                        stochastic_round(fractional_batch_size, self.random_stream.as_ref());
                    reason = "退出慢启动 (负载不足)";
                } else {
                    reason = "退出慢启动 -> 进入拥塞避免";
                }
            }
        } else {
            // 当前处于"拥塞避免"阶段 (即标准AIMD)
            if avg_idle_time > self.restart_idle_threshold {
                // 1. 长时间空闲：乘性递减，并准备在下次负载回升时进入慢启动
                let fractional_batch_size =
                    self.current_batch_size as f32 * self.multiplicative_decrease_factor;
                new_batch_size =
                    stochastic_round(fractional_batch_size, self.random_stream.as_ref());
                self.is_in_slow_start = true;
                self.consecutive_underload_count = 0; // 重置计数
                reason = "乘性递减 (长时间空闲，重置慢启动)";
            } else if is_underloaded {
                // 2. 短暂空闲：达到阈值后执行乘性递减
                self.consecutive_underload_count += 1;
                if self.consecutive_underload_count >= self.underload_threshold_count {
                    let fractional_batch_size =
                        self.current_batch_size as f32 * self.multiplicative_decrease_factor;
                    new_batch_size =
                        stochastic_round(fractional_batch_size, self.random_stream.as_ref());
                    self.consecutive_underload_count = 0; // 重置计数
                    reason = "乘性递减 (负载不足，返回批次小于请求)";
                } else {
                    new_batch_size = self.current_batch_size;
                    reason = "观望 (潜在负载不足)";
                }
            } else if avg_idle_time < self.congestion_idle_threshold {
                // 3. 负载很高：加性递增
                self.consecutive_congestion_count += 1;
                self.consecutive_underload_count = 0; // 重置计数
                if self.consecutive_congestion_count >= self.congestion_threshold_count {
                    new_batch_size = self.current_batch_size + self.additive_increase_step;
                    self.consecutive_congestion_count = 0;
                    reason = "加性递增 (任务堆积)";
                } else {
                    new_batch_size = self.current_batch_size;
                    reason = "观望 (潜在拥塞)";
                }
            } else {
                // 4. 健康状态，保持稳定
                new_batch_size = self.current_batch_size;
                self.consecutive_congestion_count = 0;
                self.consecutive_underload_count = 0; // 重置计数
                reason = "保持稳定";
            }
        }

        // 确保批大小在合法范围内
        let new_batch_size = new_batch_size.clamp(self.min_batch_size, self.max_batch_size);

        if new_batch_size != self.current_batch_size {
            trace!(
                "批大小调整: {} -> {}, 原因: {}, 平均空闲={:.3}s, 负载不足比例={:.1}%",
                self.current_batch_size,
                new_batch_size,
                reason,
                avg_idle_time,
                self.underload_buffer.iter().filter(|&&x| x).count() as f32
                    / self.underload_buffer.len() as f32
                    * 100.0
            );
        }
        self.current_batch_size = new_batch_size;
    }
}
