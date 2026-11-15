use std::{collections::HashMap, sync::Arc, time::Duration};

use dashmap::DashMap;
use tokio::sync::mpsc;
use uuid::Uuid;
use worker::model::tensor::DynTensor;

use crate::{
    compute_device::vgpu::ShareableComputeDevice,
    compute_node::r#virtual::{
        BaseWork,
        virtual_compute_node::{
            base_virtual_compute_node_ext::{ComputeFuture, DynBaseVirtualComputeNode},
            components::{
                NodeId, WorkerId, WorkerRegistry,
                worker_manager::{BatchData, WorkerManager},
            },
            tests::{MockResponse, wait_for_completion},
        },
    },
    util::{Input, Output},
};

// Mock DynBaseVirtualComputeNode for testing
#[derive(Debug)]
struct MockComputeNode {
    _node_id: NodeId,
    response_delay: Duration,
    started: Option<mpsc::UnboundedSender<(WorkerId, usize)>>,
    gate: Option<Arc<tokio::sync::Semaphore>>,
}

impl MockComputeNode {
    fn new(node_id: NodeId) -> Self {
        Self {
            _node_id: node_id,
            response_delay: Duration::from_millis(10),
            started: None,
            gate: None,
        }
    }

    #[allow(dead_code)]
    fn with_delay(node_id: NodeId, delay: Duration) -> Self {
        Self {
            _node_id: node_id,
            response_delay: delay,
            started: None,
            gate: None,
        }
    }
}

impl DynBaseVirtualComputeNode for MockComputeNode {
    type ShareParam = String;
    type DeviceState = MockResponse;

    fn command(
        &self,
        _command: crate::compute_node::r#virtual::base::Command<
            Self::ShareParam,
        >,
    ) -> crate::compute_node::r#virtual::virtual_compute_node::base_virtual_compute_node_ext::CommandFuture<
        '_,
        Self::ShareParam,
    >{
        unimplemented!("command not needed for worker_manager tests")
    }

    fn state(
        &self,
    ) -> crate::compute_node::r#virtual::virtual_compute_node::base_virtual_compute_node_ext::StateFuture<
        '_,
        Self::ShareParam,
    >{
        unimplemented!("state not needed for worker_manager tests")
    }

    fn status(
        &self,
    ) -> crate::compute_node::r#virtual::virtual_compute_node::base_virtual_compute_node_ext::StatusFuture<
        '_,
        Self::DeviceState,
    >{
        unimplemented!("status not needed for worker_manager tests")
    }

    fn compute(
        &self,
        work: crate::compute_node::r#virtual::base::Work<Input>,
    ) -> ComputeFuture<'_> {
        Box::pin(async move {
            if let Some(started) = &self.started {
                started
                    .send((
                        work.target_worker,
                        work.work.data().dimensions()[0] as usize,
                    ))
                    .unwrap();
            }
            if let Some(gate) = &self.gate {
                gate.acquire().await.unwrap().forget();
            }
            tokio::time::sleep(self.response_delay).await;

            // Create mock output tensor
            let input_tensor = work.work.data().clone();
            let output_tensor =
                create_mock_output_tensor(&input_tensor).map_err(|e| e.to_string())?;
            let output_work = BaseWork::<Output>::new(work.work.id(), output_tensor);

            Ok(crate::compute_node::r#virtual::base::Work {
                target_worker: work.target_worker,
                work: output_work,
            })
        })
    }
}

// Helper function to create mock DynTensor for testing
fn create_mock_tensor(
    batch_size: usize,
) -> Result<DynTensor, Box<dyn std::error::Error + Send + Sync>> {
    use ndarray::Array;

    // Create a mock tensor with the specified batch size
    // Shape: [batch_size, 1] - simple 2D tensor for testing
    let data: ndarray::Array2<f32> = Array::zeros((batch_size, 1));
    let tensor: DynTensor = data.into();
    Ok(tensor)
}

fn create_mock_output_tensor(
    input: &DynTensor,
) -> Result<DynTensor, Box<dyn std::error::Error + Send + Sync>> {
    create_mock_tensor(input.dimensions()[0] as usize)
}

// Helper function to create mock nodes
fn create_mock_nodes(
    count: usize,
) -> HashMap<
    NodeId,
    Arc<dyn DynBaseVirtualComputeNode<ShareParam = String, DeviceState = MockResponse>>,
> {
    let mut nodes = HashMap::new();
    for _i in 0..count {
        let node_id = Uuid::new_v4();
        let node = Arc::new(MockComputeNode::new(node_id))
            as Arc<dyn DynBaseVirtualComputeNode<ShareParam = String, DeviceState = MockResponse>>;
        nodes.insert(node_id, node);
    }
    nodes
}

#[tokio::test]
async fn test_worker_manager_creation() {
    let nodes = create_mock_nodes(2);
    let worker_manager = WorkerManager::<String, MockResponse>::new(nodes.clone());

    assert_eq!(worker_manager.nodes.len(), 2);
    assert!(worker_manager.worker_handles.is_empty());
}

#[tokio::test]
async fn test_worker_manager_with_shared_registry() {
    let registry = Arc::new(WorkerRegistry::<String>::new());

    let worker_manager = WorkerManager::<String, MockResponse>::with_shared_registry(
        registry.clone(),
        Arc::new(DashMap::new()),
    );

    // 验证共享组件
    // 验证注册表是共享的
    assert!(Arc::ptr_eq(&worker_manager.registry, &registry));
}

#[tokio::test]
async fn test_add_worker_success() {
    let nodes = create_mock_nodes(1);
    let node_id = *nodes.keys().next().unwrap();
    let worker_manager = WorkerManager::<String, MockResponse>::new(nodes);

    let worker_id = Uuid::new_v4();
    let device_id = Uuid::new_v4();
    let (tx, _rx) = mpsc::unbounded_channel();

    let result = worker_manager.add_worker(
        worker_id,
        "test_model".to_string(),
        node_id,
        device_id,
        "1024,4".to_string(),
        tx,
    );

    assert!(result.is_ok());

    // 验证worker已注册
    let worker_info = worker_manager.registry.get_worker_info(worker_id).unwrap();
    assert_eq!(worker_info.model_name, "test_model");
    assert_eq!(worker_info.node_id, node_id);

    // 验证worker句柄已存储
    assert!(worker_manager.worker_handles.contains_key(&worker_id));
}

#[tokio::test]
async fn test_add_worker_nonexistent_node() {
    let nodes = create_mock_nodes(1);
    let worker_manager = WorkerManager::<String, MockResponse>::new(nodes);

    let worker_id = Uuid::new_v4();
    let device_id = Uuid::new_v4();
    let nonexistent_node_id = Uuid::new_v4();
    let (tx, _rx) = mpsc::unbounded_channel();

    let result = worker_manager.add_worker(
        worker_id,
        "test_model".to_string(),
        nonexistent_node_id,
        device_id,
        "1024,4".to_string(),
        tx,
    );

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not found"));
}

#[tokio::test]
async fn test_add_duplicate_worker() {
    let nodes = create_mock_nodes(1);
    let node_id = *nodes.keys().next().unwrap();
    let worker_manager = WorkerManager::<String, MockResponse>::new(nodes);

    let worker_id = Uuid::new_v4();
    let device_id = Uuid::new_v4();
    let (tx1, _rx1) = mpsc::unbounded_channel();
    let (tx2, _rx2) = mpsc::unbounded_channel();

    // 添加第一个worker
    let result1 = worker_manager.add_worker(
        worker_id,
        "test_model".to_string(),
        node_id,
        device_id,
        "1024,4".to_string(),
        tx1,
    );
    assert!(result1.is_ok());

    // 尝试添加相同ID的worker
    let result2 = worker_manager.add_worker(
        worker_id,
        "test_model_2".to_string(),
        node_id,
        device_id,
        "1024,4".to_string(),
        tx2,
    );
    assert!(result2.is_err());
    assert!(result2.unwrap_err().contains("already exists"));
}

#[tokio::test]
async fn test_remove_worker_success() {
    let nodes = create_mock_nodes(1);
    let node_id = *nodes.keys().next().unwrap();
    let worker_manager = WorkerManager::<String, MockResponse>::new(nodes);

    let worker_id = Uuid::new_v4();
    let device_id = Uuid::new_v4();
    let (tx, _rx) = mpsc::unbounded_channel();

    // 添加worker
    worker_manager
        .add_worker(
            worker_id,
            "test_model".to_string(),
            node_id,
            device_id,
            "1024,4".to_string(),
            tx,
        )
        .unwrap();

    wait_for_completion().await;

    // 验证worker存在
    assert!(worker_manager.registry.get_worker_info(worker_id).is_some());
    assert!(worker_manager.worker_handles.contains_key(&worker_id));

    // 移除worker
    let result = worker_manager.remove_worker(worker_id).await;
    assert!(result.is_ok());

    // 验证worker已移除
    assert!(worker_manager.registry.get_worker_info(worker_id).is_none());
    assert!(!worker_manager.worker_handles.contains_key(&worker_id));
}

#[tokio::test]
async fn test_remove_nonexistent_worker() {
    let nodes = create_mock_nodes(1);
    let worker_manager = WorkerManager::<String, MockResponse>::new(nodes);

    let nonexistent_worker_id = Uuid::new_v4();

    let result = worker_manager.remove_worker(nonexistent_worker_id).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not found"));
}

#[tokio::test]
async fn test_get_model_workers() {
    let nodes = create_mock_nodes(1);
    let node_id = *nodes.keys().next().unwrap();
    let worker_manager = WorkerManager::<String, MockResponse>::new(nodes);

    let worker_id1 = Uuid::new_v4();
    let worker_id2 = Uuid::new_v4();
    let worker_id3 = Uuid::new_v4();
    let device_id = Uuid::new_v4();

    let (tx1, _rx1) = mpsc::unbounded_channel();
    let (tx2, _rx2) = mpsc::unbounded_channel();
    let (tx3, _rx3) = mpsc::unbounded_channel();

    // 添加两个相同模型的worker和一个不同模型的worker
    worker_manager
        .add_worker(
            worker_id1,
            "model_a".to_string(),
            node_id,
            device_id,
            "1024,4".to_string(),
            tx1,
        )
        .unwrap();
    worker_manager
        .add_worker(
            worker_id2,
            "model_a".to_string(),
            node_id,
            device_id,
            "1024,4".to_string(),
            tx2,
        )
        .unwrap();
    worker_manager
        .add_worker(
            worker_id3,
            "model_b".to_string(),
            node_id,
            device_id,
            "1024,4".to_string(),
            tx3,
        )
        .unwrap();

    wait_for_completion().await;

    let model_a_workers = worker_manager.get_model_workers(&"model_a".to_string());
    let model_b_workers = worker_manager.get_model_workers(&"model_b".to_string());
    let model_c_workers = worker_manager.get_model_workers(&"model_c".to_string());

    assert_eq!(model_a_workers.len(), 2);
    assert!(model_a_workers.contains(&worker_id1));
    assert!(model_a_workers.contains(&worker_id2));

    assert_eq!(model_b_workers.len(), 1);
    assert!(model_b_workers.contains(&worker_id3));

    assert_eq!(model_c_workers.len(), 0);
}

#[tokio::test]
async fn test_get_model_worker_handles() {
    let nodes = create_mock_nodes(1);
    let node_id = *nodes.keys().next().unwrap();
    let worker_manager = WorkerManager::<String, MockResponse>::new(nodes);

    let worker_id1 = Uuid::new_v4();
    let worker_id2 = Uuid::new_v4();
    let device_id = Uuid::new_v4();
    let (tx1, _rx1) = mpsc::unbounded_channel();
    let (tx2, _rx2) = mpsc::unbounded_channel();

    worker_manager
        .add_worker(
            worker_id1,
            "test_model".to_string(),
            node_id,
            device_id,
            "1024,4".to_string(),
            tx1,
        )
        .unwrap();
    worker_manager
        .add_worker(
            worker_id2,
            "test_model".to_string(),
            node_id,
            device_id,
            "1024,4".to_string(),
            tx2,
        )
        .unwrap();

    wait_for_completion().await;

    let handles = worker_manager.get_model_worker_handles(&"test_model".to_string());
    assert_eq!(handles.len(), 2);

    let worker_ids: Vec<WorkerId> = handles.iter().map(|(id, _)| *id).collect();
    assert!(worker_ids.contains(&worker_id1));
    assert!(worker_ids.contains(&worker_id2));
}

// 注意：性能指标现在完全集成在WorkerHandler内部，
// 不再需要外部访问接口，因此移除了相关测试

// 注意：WorkerManagerTrait已被删除，因为性能指标现在完全内部管理
// 不再需要外部访问接口，因此移除了相关测试

#[tokio::test]
async fn test_registry_access() {
    let nodes = create_mock_nodes(1);
    let worker_manager = WorkerManager::<String, MockResponse>::new(nodes);

    // 测试获取组件引用
    let registry = worker_manager.get_registry();

    // 验证这些是同一个实例
    assert!(Arc::ptr_eq(&worker_manager.registry, &registry));
}

#[test]
fn test_worker_manager_basic_construction() {
    // 这是一个不需要运行时环境的基本测试
    let nodes = Default::default();
    let worker_manager = WorkerManager::<String, MockResponse>::with_shared_registry(
        Arc::new(WorkerRegistry::new()),
        nodes,
    );

    // 验证基本状态
    assert_eq!(worker_manager.nodes.len(), 0);
    assert!(worker_manager.worker_handles.is_empty());
}

#[test]
fn test_batch_data_creation() {
    // 测试BatchData结构的基本功能
    let empty_tensors: Vec<DynTensor> = vec![];
    let work_ids = vec![uuid::Uuid::new_v4(), uuid::Uuid::new_v4()];
    let batch_sizes = vec![1, 2];
    let (tx1, _rx1) = tokio::sync::oneshot::channel();
    let (tx2, _rx2) = tokio::sync::oneshot::channel();
    let senders = vec![tx1, tx2];

    let batch_data = BatchData {
        input_tensors: empty_tensors,
        work_ids: work_ids.clone(),
        batch_sizes: batch_sizes.clone(),
        senders,
    };

    assert_eq!(batch_data.work_ids.len(), 2);
    assert_eq!(batch_data.batch_sizes.len(), 2);
    assert_eq!(batch_data.work_ids, work_ids);
    assert_eq!(batch_data.batch_sizes, batch_sizes);
}

#[tokio::test]
async fn idle_worker_can_pause_without_another_request_arriving() {
    use super::super::worker_registry::WorkerStatus;
    let nodes = create_mock_nodes(1);
    let node_id = *nodes.keys().next().unwrap();
    let manager = WorkerManager::<String, MockResponse>::new(nodes);
    let worker_id = Uuid::new_v4();
    let (tx, mut rx) = mpsc::unbounded_channel();
    manager
        .add_worker(
            worker_id,
            "idle".into(),
            node_id,
            Uuid::new_v4(),
            "slice".into(),
            tx,
        )
        .unwrap();
    let request = tokio::time::timeout(Duration::from_secs(1), rx.recv())
        .await
        .unwrap()
        .unwrap();
    manager
        .registry
        .update_worker_status(worker_id, WorkerStatus::Pausing);
    manager
        .registry
        .get_status_listener(worker_id)
        .wait_for_worker_status(WorkerStatus::Paused, Some(Duration::from_secs(1)))
        .await
        .unwrap();
    assert!(request.resp.is_closed());
    manager.remove_worker(worker_id).await.unwrap();
}

#[tokio::test]
async fn assigned_work_completes_before_worker_pauses() {
    use super::super::worker_registry::WorkerStatus;
    let nodes = create_mock_nodes(1);
    let node_id = *nodes.keys().next().unwrap();
    let manager = WorkerManager::<String, MockResponse>::new(nodes);
    let worker_id = Uuid::new_v4();
    let (tx, mut rx) = mpsc::unbounded_channel();
    manager
        .add_worker(
            worker_id,
            "busy".into(),
            node_id,
            Uuid::new_v4(),
            "slice".into(),
            tx,
        )
        .unwrap();
    let request = tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            let request = rx.recv().await.unwrap();
            // The initial Running notification can cancel the first fetch.
            if !request.resp.is_closed() {
                break request;
            }
        }
    })
    .await
    .unwrap();
    let (result_tx, result_rx) = tokio::sync::oneshot::channel();
    let work = BaseWork::<Input>::new(Uuid::new_v4(), create_mock_tensor(1).unwrap());
    // Both events are ready on the next worker poll: the accepted batch must drain.
    manager
        .registry
        .update_worker_status(worker_id, WorkerStatus::Pausing);
    request.resp.send(vec![(work, result_tx)]).unwrap();
    tokio::time::timeout(Duration::from_secs(1), result_rx)
        .await
        .unwrap()
        .unwrap();
    manager
        .registry
        .get_status_listener(worker_id)
        .wait_for_worker_status(WorkerStatus::Paused, Some(Duration::from_secs(1)))
        .await
        .unwrap();
    manager.remove_worker(worker_id).await.unwrap();
}

#[tokio::test]
async fn successful_inference_reports_actual_batch_and_ignores_retired_measurements() {
    use super::super::worker_registry::WorkerStatus;
    let nodes = create_mock_nodes(1);
    let node_id = *nodes.keys().next().unwrap();
    let manager = WorkerManager::<String, MockResponse>::new(nodes);
    let (measure_tx, mut measure_rx) = mpsc::unbounded_channel();
    manager
        .registry
        .set_performance_observer(Arc::new(move |model, profile, batch, seconds| {
            measure_tx
                .send((model.to_string(), profile.clone(), batch, seconds))
                .unwrap();
        }));
    let worker_id = Uuid::new_v4();
    let (tx, mut rx) = mpsc::unbounded_channel();
    manager
        .add_worker(
            worker_id,
            "m".into(),
            node_id,
            Uuid::new_v4(),
            "slice".into(),
            tx,
        )
        .unwrap();
    let request = tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            let request = rx.recv().await.unwrap();
            if !request.resp.is_closed() {
                break request;
            }
        }
    })
    .await
    .unwrap();
    let (result_tx, result_rx) = tokio::sync::oneshot::channel();
    request
        .resp
        .send(vec![(
            BaseWork::<Input>::new(Uuid::new_v4(), create_mock_tensor(3).unwrap()),
            result_tx,
        )])
        .unwrap();
    let (model, profile, batch, seconds) =
        tokio::time::timeout(Duration::from_secs(1), measure_rx.recv())
            .await
            .unwrap()
            .unwrap();
    assert_eq!((model.as_str(), profile.as_str(), batch), ("m", "slice", 3));
    assert!(seconds.is_finite() && seconds > 0.0);
    result_rx.await.unwrap();
    manager
        .registry
        .update_worker_status(worker_id, WorkerStatus::Paused);
    manager.registry.record_inference(worker_id, "m", 3, 0.1);
    assert!(measure_rx.try_recv().is_err());
    manager.remove_worker(worker_id).await.unwrap();
    manager.registry.record_inference(worker_id, "m", 3, 0.1);
    assert!(measure_rx.try_recv().is_err());
}

#[tokio::test]
async fn retiring_worker_drains_only_its_batch_and_other_node_drains_shared_queue() {
    use super::super::{
        request_scheduler::{RequestScheduler, batch_size_extractor::BatchExtractorStrategy},
        worker_registry::WorkerStatus,
    };
    let scheduler = Arc::new(RequestScheduler::with_strategy(
        BatchExtractorStrategy::timeout_split_and_satisfy(Duration::from_millis(8)),
    ));
    let (started_tx, mut started_rx) = mpsc::unbounded_channel();
    let node_ids = [Uuid::new_v4(), Uuid::new_v4()];
    let worker_ids = [Uuid::new_v4(), Uuid::new_v4()];
    let gates = [
        Arc::new(tokio::sync::Semaphore::new(0)),
        Arc::new(tokio::sync::Semaphore::new(0)),
    ];
    let nodes = node_ids
        .iter()
        .zip(&gates)
        .map(|(&id, gate)| {
            let mut node = MockComputeNode::new(id);
            node.started = Some(started_tx.clone());
            node.gate = Some(gate.clone());
            (
                id,
                Arc::new(node)
                    as Arc<
                        dyn DynBaseVirtualComputeNode<
                                ShareParam = String,
                                DeviceState = MockResponse,
                            >,
                    >,
            )
        })
        .collect();
    let manager = WorkerManager::new(nodes);
    let model = "shared".to_string();
    for i in 0..2 {
        manager
            .add_worker(
                worker_ids[i],
                model.clone(),
                node_ids[i],
                Uuid::new_v4(),
                "slice".into(),
                scheduler.get_worker_request_sender(),
            )
            .unwrap();
    }
    let mut tasks = Vec::new();
    for _ in 0..64 {
        let scheduler = scheduler.clone();
        let model = model.clone();
        tasks.push(tokio::spawn(async move {
            let id = Uuid::new_v4();
            let result = scheduler
                .add_compute_task(
                    model,
                    BaseWork::<Input>::new(id, create_mock_tensor(1).unwrap()),
                )
                .await
                .unwrap();
            assert_eq!(result.id(), id);
            assert_eq!(result.data().dimensions()[0], 1);
        }));
    }
    let queued = tokio::time::timeout(Duration::from_secs(2), async {
        let started_batches = vec![
            started_rx.recv().await.unwrap(),
            started_rx.recv().await.unwrap(),
        ];
        let queued = 64
            - started_batches
                .iter()
                .map(|(_, count)| count)
                .sum::<usize>();
        assert!(queued > 0);
        let mut started: Vec<_> = started_batches.iter().map(|(id, _)| *id).collect();
        started.sort();
        let mut expected = worker_ids.to_vec();
        expected.sort();
        assert_eq!(started, expected);
        while scheduler.get_model_queue_batch_size(&model) != queued {
            tokio::task::yield_now().await;
        }
        queued
    })
    .await
    .unwrap();
    // Stop new pulls, finish the assigned batch, then remove this worker.
    manager
        .registry
        .update_worker_status(worker_ids[0], WorkerStatus::Pausing);
    gates[0].add_permits(1);
    manager
        .registry
        .get_status_listener(worker_ids[0])
        .wait_for_worker_status(WorkerStatus::Paused, Some(Duration::from_secs(2)))
        .await
        .unwrap();
    assert_eq!(scheduler.get_model_queue_batch_size(&model), queued);
    manager.remove_worker(worker_ids[0]).await.unwrap();
    gates[1].add_permits(64);
    tokio::time::timeout(Duration::from_secs(2), async {
        for task in tasks {
            task.await.unwrap();
        }
    })
    .await
    .unwrap();
    assert_eq!(scheduler.get_model_queue_batch_size(&model), 0);
    let mut remaining_batches = 0;
    while let Ok((id, _)) = started_rx.try_recv() {
        assert_eq!(id, worker_ids[1]);
        remaining_batches += 1;
    }
    assert!(remaining_batches > 0);
    manager.remove_worker(worker_ids[1]).await.unwrap();
}

#[tokio::test]
async fn dropping_worker_owner_aborts_task_and_unregisters_indices() {
    let nodes = create_mock_nodes(1);
    let node_id = *nodes.keys().next().unwrap();
    let manager = WorkerManager::<String, MockResponse>::new(nodes);
    let registry = manager.get_registry();
    let worker_id = Uuid::new_v4();
    let device_id = Uuid::new_v4();
    let (tx, _rx) = mpsc::unbounded_channel();
    manager
        .add_worker(
            worker_id,
            "m".into(),
            node_id,
            device_id,
            "slice".into(),
            tx,
        )
        .unwrap();
    let owner = manager.worker_handles.remove(&worker_id).unwrap().1;
    let stopped = owner.handle.abort_handle();
    drop(owner);
    assert!(!registry.contains_worker(worker_id));
    assert!(registry.get_model_workers(&"m".into()).is_empty());
    assert!(registry.get_device_workers(device_id).is_empty());
    assert!(registry.get_node_workers(node_id).is_empty());
    tokio::time::timeout(Duration::from_secs(1), async {
        while !stopped.is_finished() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(!registry.contains_worker(worker_id));
}
