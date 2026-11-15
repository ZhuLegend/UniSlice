use std::{
    collections::VecDeque,
    ops::Deref,
    sync::{Arc, RwLock},
};

use dashmap::DashMap;
use model_queue::{RequestEntry, Requests};
use tokio::{
    sync::{mpsc, oneshot},
    task,
    task::JoinHandle,
};
use tracing::trace;

use super::{
    global_task_tracker,
    worker_manager::WorkerRequest,
    worker_registry::{ModelId, WorkerId},
};
use crate::{
    compute_node::r#virtual::{
        BaseWork,
        virtual_compute_node::components::request_scheduler::batch_size_extractor::{
            BatchExtractorStrategy, BatchSizeExtractor,
        },
    },
    util::{Input, Output},
};

pub mod batch_size_extractor;
pub mod pool_queue;
pub use pool_queue::{PendingRequests, PoolRequestQueue};

mod model_queue;

/// 待处理Worker请求类型别名
type PendingWorkerRequests = Arc<DashMap<ModelId, RwLock<VecDeque<WorkerRequest>>>>;

/// 负责请求调度、批处理分发和任务队列管理
pub struct RequestScheduler {
    random_namespace: Option<String>,
    pool_queue: PoolRequestQueue,
    /// 模型输入请求队列: model_id -> 请求队列
    model_reqs: Arc<DashMap<ModelId, Requests>>,
    /// 各模型待满足的Worker请求: model_id -> 请求列表
    pending_worker_reqs: PendingWorkerRequests,
    /// WorkerRequest提交通道
    worker_req_tx: mpsc::UnboundedSender<WorkerRequest>,
    /// 每个模型的事件通道发送端: model_id -> sender
    model_event_txs: Arc<DashMap<ModelId, mpsc::UnboundedSender<()>>>,
    /// 每个模型的处理任务句柄: model_id -> task_handle
    model_handles: Arc<DashMap<ModelId, JoinHandle<()>>>,
    /// 管理pending WorkerRequests的调度任务句柄
    worker_req_handle: JoinHandle<()>,
    empty_queue_checker_handle: JoinHandle<()>,
    /// Batch size提取策略
    batch_size_extractor: BatchExtractorStrategy,
}

impl RequestScheduler {
    /// 创建新的请求调度器，指定batch size提取策略
    pub fn with_strategy(batch_size_extractor: BatchExtractorStrategy) -> Self {
        let model_reqs: Arc<DashMap<ModelId, Requests>> = Arc::new(DashMap::new());
        let pending_worker_reqs: PendingWorkerRequests = Arc::new(DashMap::new());
        let (worker_req_tx, worker_req_rx) = mpsc::unbounded_channel::<WorkerRequest>();
        let model_event_txs: Arc<DashMap<ModelId, mpsc::UnboundedSender<()>>> =
            Arc::new(DashMap::new());
        let model_handles: Arc<DashMap<ModelId, JoinHandle<()>>> = Arc::new(DashMap::new());

        let worker_req_handle = Self::spawn_worker_request_handler(
            worker_req_rx,
            pending_worker_reqs.clone(),
            model_event_txs.clone(),
        );

        // 启动定时检查器，确保空队列的模型也能被处理
        let empty_queue_checker_handle = Self::spawn_empty_queue_checker(
            model_reqs.clone(),
            pending_worker_reqs.clone(),
            model_event_txs.clone(),
        );

        Self {
            random_namespace: None,
            pool_queue: PoolRequestQueue::new(model_reqs.clone(), model_event_txs.clone()),
            model_reqs,
            pending_worker_reqs,
            worker_req_tx,
            model_event_txs,
            model_handles,
            worker_req_handle,
            empty_queue_checker_handle,
            batch_size_extractor,
        }
    }

    /// 获取Worker请求发送通道
    pub(crate) fn set_random_namespace(&mut self, namespace: String) {
        self.random_namespace = Some(namespace);
    }

    /// 获取Worker请求发送通道
    pub fn get_worker_request_sender(&self) -> mpsc::UnboundedSender<WorkerRequest> {
        self.worker_req_tx.clone()
    }

    /// Live admission state and movable pending queues for this pool.
    pub fn pool_queue(&self) -> PoolRequestQueue {
        self.pool_queue.clone()
    }

    /// Accept existing fragments only if the target generation is still serving.
    pub fn accept_pending(
        &self,
        pending: PendingRequests,
        epoch: u64,
    ) -> Result<(), PendingRequests> {
        self.ensure_model_queue_exists(&pending.model);
        self.ensure_model_handler(&pending.model);
        self.pool_queue.accept(pending, epoch)
    }

    /// 添加计算任务到队列
    pub async fn add_compute_task(
        &self,
        model_id: ModelId,
        work: BaseWork<Input>,
    ) -> Result<BaseWork<Output>, String> {
        self.add_compute_task_on_admission(model_id, work, || {})
            .await
    }

    /// Run a short arrival-accounting callback only after admission succeeds.
    pub async fn add_compute_task_on_admission(
        &self,
        model_id: ModelId,
        work: BaseWork<Input>,
        on_admitted: impl FnOnce(),
    ) -> Result<BaseWork<Output>, String> {
        let (tx, rx) = oneshot::channel();
        let task_id = work.id();
        let batch_size = work.data().dimensions().first().copied().unwrap_or(0) as usize;
        let entry = RequestEntry::new(work, tx)?;

        // Admission and whole batch extraction share the pool's model boundary.
        // A rejected admission has not executed or emitted an arrival event.
        let queued_at_ms = {
            let state = self.pool_queue.state(&model_id);
            let state = state.lock().unwrap();
            if !self.pool_queue.serving_locked(&state) {
                return Err(pool_queue::POOL_UNAVAILABLE.to_owned());
            }
            let queued_at_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            self.with_model_queue(&model_id, |queue| {
                queue.push_back(entry);
            });
            on_admitted();
            queued_at_ms
        };
        // 记录任务到达事件
        let model_name = model_id.clone();
        task::spawn_blocking(move || {
            global_task_tracker().track_task_arrived_at(
                task_id,
                &model_name,
                batch_size,
                queued_at_ms,
            );
        });

        // 确保模型有对应的处理任务
        self.ensure_model_handler(&model_id);

        // 通知对应模型的处理任务
        if let Some(event_tx) = self.model_event_txs.get(&model_id) {
            let _ = event_tx.send(());
        }

        // 等待处理结果
        rx.await
            .map_err(|_| "Task was unexpectedly dropped".to_string())
    }

    /// 请求批量任务（由Worker调用）
    ///
    /// 任务请求会以FIFO的方式处理，保证返回的batch size是请求的desired_batch大小。
    pub async fn request_batch(&self, request: WorkerRequest) -> Result<(), String> {
        self.worker_req_tx
            .send(request)
            .map_err(|_| "Failed to send worker request".to_string())
    }

    /// 获取模型队列的总batch size
    pub fn get_model_queue_batch_size(&self, model_id: &ModelId) -> usize {
        self.model_reqs
            .get(model_id)
            .map(|lock| lock.total_batch_size())
            .unwrap_or(0)
    }

    /// 获取所有模型的总batch size
    pub fn get_all_model_queue_batch_sizes(&self) -> std::collections::HashMap<ModelId, usize> {
        self.model_reqs
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().total_batch_size()))
            .collect()
    }

    /// 清理指定Worker的pending请求
    pub fn cleanup_worker_requests(&self, worker_id: WorkerId) {
        for entry in self.pending_worker_reqs.iter() {
            entry
                .value()
                .write()
                .unwrap()
                .retain(|req| req.worker_id != worker_id);
        }
    }

    /// 清理所有无效的Worker请求
    pub fn cleanup_invalid_worker_requests(
        &self,
        valid_workers: &std::collections::HashSet<WorkerId>,
    ) {
        for entry in self.pending_worker_reqs.iter() {
            entry
                .value()
                .write()
                .unwrap()
                .retain(|req| valid_workers.contains(&req.worker_id));
        }
    }

    // 私有辅助方法

    /// 确保模型队列存在
    fn ensure_model_queue_exists(&self, model_id: &ModelId) {
        // 使用 DashMap 的 entry API 原子性地检查并插入
        self.model_reqs
            .entry(model_id.clone())
            .or_insert_with(|| Requests::new(VecDeque::new()));
    }

    /// 获取模型队列的引用
    fn with_model_queue<T, F>(&self, model_id: &ModelId, f: F) -> T
    where
        F: FnOnce(&Requests) -> T,
    {
        self.ensure_model_queue_exists(model_id);
        let queue = self.model_reqs.get(model_id).unwrap();
        f(queue.value())
    }

    fn spawn_empty_queue_checker(
        model_reqs: Arc<DashMap<ModelId, Requests>>,
        pending_worker_reqs: PendingWorkerRequests,
        model_event_txs: Arc<DashMap<ModelId, mpsc::UnboundedSender<()>>>,
    ) -> JoinHandle<()> {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_micros(200));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                interval.tick().await;
                for model_id in model_reqs.iter().map(|entry| entry.key().clone()) {
                    let has_model_reqs = model_reqs
                        .get(&model_id)
                        .map(|lock| !lock.is_empty())
                        .unwrap_or(false);
                    let has_pending_reqs = pending_worker_reqs
                        .get(&model_id)
                        .map(|lock| !lock.read().unwrap().is_empty())
                        .unwrap_or(false);
                    if (has_model_reqs && has_pending_reqs)
                        && let Some(event_tx) = model_event_txs.get(&model_id)
                    {
                        let _ = event_tx.send(());
                    }
                }
            }
        })
    }

    /// 生成Worker请求处理任务
    fn spawn_worker_request_handler(
        mut worker_req_rx: mpsc::UnboundedReceiver<WorkerRequest>,
        pending_worker_reqs: PendingWorkerRequests,
        model_event_txs: Arc<DashMap<ModelId, mpsc::UnboundedSender<()>>>,
    ) -> JoinHandle<()> {
        tokio::spawn(async move {
            while let Some(req) = worker_req_rx.recv().await {
                let model_name = req.model_name.clone();

                trace!(
                    "Received worker request for model: {model_name:?}, desired batch size: {}",
                    req.desired_batch
                );

                // 添加到pending队列
                if let Some(queue) = pending_worker_reqs.get(&model_name) {
                    queue.write().unwrap().push_back(req);
                } else {
                    pending_worker_reqs
                        .insert(model_name.clone(), RwLock::new(VecDeque::from(vec![req])));
                }

                // 触发对应模型的调度
                if let Some(event_tx) = model_event_txs.get(&model_name) {
                    let _ = event_tx.send(());
                }
            }
        })
    }

    /// 确保模型有对应的处理任务
    fn ensure_model_handler(&self, model_id: &ModelId) {
        // 快速路径：检查是否已经初始化
        if self.model_handles.contains_key(model_id) {
            return;
        }

        // 使用 DashMap 的 entry API 原子性地检查并创建
        self.model_handles
            .entry(model_id.clone())
            .or_insert_with(|| {
                // 创建事件通道
                let (event_tx, event_rx) = mpsc::unbounded_channel::<()>();

                // 先插入事件通道
                self.model_event_txs.insert(model_id.clone(), event_tx);

                // 创建处理任务
                let model_id_clone = model_id.clone();
                let model_reqs_clone = self.model_reqs.clone();
                let pending_worker_reqs_clone = self.pending_worker_reqs.clone();
                let seed = self.random_namespace.as_ref().and_then(|namespace| {
                    let key = format!("extraction/{namespace}/{model_id}");
                    let seed = crate::experiment_random::seed(&key);
                    if let Some(seed) = seed {
                        tracing::info!(target: "allocation", event="extraction_random_seed_injected", model=%model_id, stream=%key, seed);
                    }
                    seed
                });
                let batch_size_extractor_clone = Arc::new(self.batch_size_extractor.clone().with_random_seed(seed));
                let pool_queue = self.pool_queue.clone();

                tokio::spawn(async move {
                    let mut event_rx = event_rx;
                    while event_rx.recv().await.is_some() {
                        // Notifications request a scan of current state, not one scan per
                        // historical tick. Coalesce queued wakes before the blocking scan.
                        while event_rx.try_recv().is_ok() {}
                        let model_id_clone = model_id_clone.clone();
                        let model_reqs_clone = model_reqs_clone.clone();
                        let pending_worker_reqs_clone = pending_worker_reqs_clone.clone();
                        let batch_size_extractor_clone = Arc::clone(&batch_size_extractor_clone);
                        let pool_queue = pool_queue.clone();
                        trace!("try allocate model requests for {model_id_clone:?}");

                        let _ = task::spawn_blocking(move || {
                            Self::process_model_requests(
                                batch_size_extractor_clone,
                                &model_id_clone,
                                &model_reqs_clone,
                                &pending_worker_reqs_clone,
                                &pool_queue,
                            )
                        })
                        .await;
                    }
                })
            });
    }

    fn process_model_requests(
        batch_size_extractor: Arc<BatchExtractorStrategy>,
        model_id: &ModelId,
        model_reqs: &Arc<DashMap<ModelId, Requests>>,
        pending_worker_reqs: &PendingWorkerRequests,
        pool_queue: &PoolRequestQueue,
    ) {
        let state = pool_queue.state(model_id);
        let state = state.lock().unwrap();
        if !pool_queue.serving_locked(&state) {
            if let Some(queue) = pending_worker_reqs.get(model_id) {
                queue.write().unwrap().clear();
            }
            return;
        }
        let mut processed = 0;
        while let Some(desired_batch_size) = pending_worker_reqs.get(model_id).and_then(|queue| {
            let mut queue = queue.write().ok()?;
            queue.retain(|req| {
                !req.resp.is_closed() && pool_queue.worker_eligible(&state, req.worker_id)
            });
            queue.front().map(|req| req.desired_batch)
        }) {
            if processed >= 64 {
                break;
            }
            processed += 1;
            let start = std::time::Instant::now();
            // 获取模型请求队列的锁
            let model_reqs_lock = match model_reqs.get(model_id) {
                Some(lock) => lock,
                None => break, // 没有模型请求队列，这理论上不应该发生
            };

            let should_proceed =
                { batch_size_extractor.should_extract(desired_batch_size, &model_reqs_lock) }; // 读锁在这里释放

            if should_proceed {
                let pending_queue = match pending_worker_reqs.get(model_id) {
                    Some(queue) => queue,
                    None => break, // 没有pending请求队列，这理论上不应该发生
                };
                // 可以满足，提取任务
                let req = pending_queue.write().unwrap().pop_front().unwrap();
                drop(pending_queue); // 提前释放pending队列的锁

                // 收集批量任务
                let works_batch =
                    batch_size_extractor.extract_batch(req.desired_batch, model_reqs_lock.deref());

                trace!("批次分配完成，耗时：{:?}", start.elapsed());

                if let Err(batch) = req.resp.send(works_batch) {
                    // 发送失败，将任务放回队列
                    Self::return_tasks_to_queue(batch, model_reqs_lock.value());
                    pool_queue.notify_change();
                }
            } else {
                // 无法满足请求，释放锁并等待更多任务
                break;
            }
        }
    }

    /// 将任务返回到队列（当Worker断开连接时）
    fn return_tasks_to_queue(
        batch: Vec<(BaseWork<Input>, oneshot::Sender<BaseWork<Output>>)>,
        model_reqs_lock: &Requests,
    ) {
        // 按相反顺序插入，保持原有顺序
        for (work, response_tx) in batch.into_iter().rev() {
            model_reqs_lock.push_front(RequestEntry::new(work, response_tx).unwrap());
        }
    }

    /// 获取pending请求统计信息
    pub fn get_pending_requests_stats(&self) -> std::collections::HashMap<ModelId, usize> {
        self.pending_worker_reqs
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().read().unwrap().len()))
            .collect()
    }
}

impl Drop for RequestScheduler {
    fn drop(&mut self) {
        self.worker_req_handle.abort();
        self.empty_queue_checker_handle.abort();

        // 终止所有模型处理任务
        for handle in self.model_handles.iter() {
            handle.value().abort();
        }
    }
}

#[cfg(test)]
impl RequestScheduler {
    pub(super) fn clone_for_test(&self) -> Self {
        Self {
            pool_queue: self.pool_queue.clone(),
            random_namespace: self.random_namespace.clone(),
            model_reqs: self.model_reqs.clone(),
            pending_worker_reqs: self.pending_worker_reqs.clone(),
            worker_req_tx: self.worker_req_tx.clone(),
            model_event_txs: self.model_event_txs.clone(),
            model_handles: self.model_handles.clone(),
            worker_req_handle: tokio::spawn(async {}), // 占位符
            empty_queue_checker_handle: tokio::spawn(async {}),
            batch_size_extractor: self.batch_size_extractor.clone(),
        }
    }
}

#[cfg(test)]
mod lifecycle_tests {
    use std::time::Duration;

    use uuid::Uuid;

    use super::*;

    fn input() -> BaseWork<Input> {
        BaseWork::<Input>::new(Uuid::new_v4(), ndarray::Array2::<f32>::zeros((1, 1)).into())
    }

    async fn fetch(
        scheduler: &RequestScheduler,
        model: &str,
        batch: usize,
    ) -> super::super::worker_manager::BatchWorkList {
        let (resp, rx) = oneshot::channel();
        scheduler
            .request_batch(WorkerRequest {
                worker_id: Uuid::new_v4(),
                model_name: model.into(),
                desired_batch: batch,
                resp,
            })
            .await
            .unwrap();
        rx.await.unwrap()
    }

    fn complete(batch: super::super::worker_manager::BatchWorkList) {
        for (work, tx) in batch {
            tx.send(BaseWork::<Output>::new(work.id(), work.data().clone()))
                .unwrap();
        }
    }

    async fn queued(scheduler: &RequestScheduler, size: usize) {
        tokio::time::timeout(Duration::from_secs(2), async {
            while scheduler.get_model_queue_batch_size(&"m".into()) != size {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    async fn claim(
        scheduler: &RequestScheduler,
        worker: Uuid,
        size: usize,
    ) -> super::super::worker_manager::BatchWorkList {
        let (resp, rx) = oneshot::channel();
        scheduler
            .request_batch(WorkerRequest {
                worker_id: worker,
                model_name: "m".into(),
                desired_batch: size,
                resp,
            })
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(2), rx)
            .await
            .unwrap()
            .unwrap()
    }

    fn tracked() -> (Arc<RequestScheduler>, Uuid) {
        let scheduler = Arc::new(RequestScheduler::with_strategy(
            BatchExtractorStrategy::split_and_satisfy(),
        ));
        let worker = Uuid::new_v4();
        scheduler.pool_queue.enable_worker_tracking();
        scheduler.pool_queue.test_worker_changed("m", worker, true);
        (scheduler, worker)
    }

    #[tokio::test]
    async fn partial_batch_handoff_preserves_original_result_and_order() {
        let (source, old_worker) = tracked();
        let (target, new_worker) = tracked();
        let id = Uuid::new_v4();
        let work = BaseWork::<Input>::new(
            id,
            ndarray::Array2::from_shape_vec((4, 1), vec![1f32, 2., 3., 4.])
                .unwrap()
                .into(),
        );
        let original = work.data().clone();
        let result = {
            let source = source.clone();
            tokio::spawn(async move { source.add_compute_task("m".into(), work).await })
        };
        queued(&source, 4).await;
        let claimed = claim(&source, old_worker, 2).await;
        source
            .pool_queue
            .test_worker_changed("m", old_worker, false);
        let pending = source.pool_queue.take_unavailable("m").unwrap();
        assert_eq!(pending.len(), 1);
        let fragment_id = pending.task_ids()[0];
        assert_eq!(source.get_model_queue_batch_size(&"m".into()), 0);
        let (_, epoch) = target.pool_queue.availability("m");
        assert!(target.accept_pending(pending, epoch).is_ok());
        complete(claimed); // Retirement does not invalidate an already claimed batch.
        assert!(!result.is_finished());
        let remaining = claim(&target, new_worker, 2).await;
        assert_eq!(remaining[0].0.id(), fragment_id);
        complete(remaining);
        let output = result.await.unwrap().unwrap();
        assert_eq!(output.id(), id);
        assert_eq!(
            bincode::encode_to_vec(output.data(), bincode::config::standard()).unwrap(),
            bincode::encode_to_vec(&original, bincode::config::standard()).unwrap()
        );
        assert_eq!(target.get_model_queue_batch_size(&"m".into()), 0);
        assert!(
            source
                .add_compute_task("m".into(), input())
                .await
                .unwrap_err()
                .contains(pool_queue::POOL_UNAVAILABLE)
        );
    }

    #[tokio::test]
    async fn stale_target_returns_ownership_and_global_gap_can_resume() {
        let (source, old_worker) = tracked();
        let (target, new_worker) = tracked();
        let result = {
            let source = source.clone();
            tokio::spawn(async move { source.add_compute_task("m".into(), input()).await })
        };
        queued(&source, 1).await;
        source
            .pool_queue
            .test_worker_changed("m", old_worker, false);
        let (_, old_epoch) = target.pool_queue.availability("m");
        target
            .pool_queue
            .test_worker_changed("m", new_worker, false);
        target.pool_queue.test_worker_changed("m", new_worker, true);
        let pending = source.pool_queue.take_unavailable("m").unwrap();
        let returned = target
            .accept_pending(pending, old_epoch)
            .err()
            .expect("stale generation must reject");
        source.pool_queue.restore(returned);
        assert!(!result.is_finished());
        source.pool_queue.test_worker_changed("m", old_worker, true);
        complete(claim(&source, old_worker, 1).await);
        result.await.unwrap().unwrap();
        assert_eq!(target.get_model_queue_batch_size(&"m".into()), 0);
    }

    #[tokio::test]
    async fn cancelled_pending_requests_are_removed_without_execution() {
        let (source, worker) = tracked();
        let result = {
            let source = source.clone();
            tokio::spawn(async move { source.add_compute_task("m".into(), input()).await })
        };
        queued(&source, 1).await;
        result.abort();
        let _ = result.await;
        source.pool_queue.test_worker_changed("m", worker, false);
        assert!(source.pool_queue.take_unavailable("m").is_none());
        assert_eq!(source.get_model_queue_batch_size(&"m".into()), 0);
    }

    #[tokio::test]
    async fn busy_model_cannot_postpone_another_models_partial_batch() {
        let scheduler = Arc::new(RequestScheduler::with_strategy(
            BatchExtractorStrategy::timeout_split_and_satisfy(Duration::from_millis(80)),
        ));
        let cold = {
            let scheduler = scheduler.clone();
            tokio::spawn(async move {
                scheduler
                    .add_compute_task("cold".into(), input())
                    .await
                    .unwrap()
            })
        };
        let hot = {
            let scheduler = scheduler.clone();
            tokio::spawn(async move {
                loop {
                    let (result, batch) =
                        tokio::join!(scheduler.add_compute_task("hot".into(), input()), async {
                            complete(fetch(&scheduler, "hot", 1).await)
                        },);
                    result.unwrap();
                    let _ = batch;
                    tokio::time::sleep(Duration::from_millis(2)).await;
                }
            })
        };
        let batch =
            tokio::time::timeout(Duration::from_millis(600), fetch(&scheduler, "cold", 32)).await;
        hot.abort();
        let batch = batch.expect("other models must not reset cold model's timeout");
        assert_eq!(batch.len(), 1);
        complete(batch);
        cold.await.unwrap();
    }

    #[tokio::test]
    async fn dropping_scheduler_releases_checker_owners() {
        let scheduler =
            RequestScheduler::with_strategy(BatchExtractorStrategy::split_and_satisfy());
        let queues = Arc::downgrade(&scheduler.model_reqs);
        drop(scheduler);
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
        assert!(queues.upgrade().is_none());
    }
}
