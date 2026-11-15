use uuid::Uuid;

use crate::compute_node::r#virtual::virtual_compute_node::{
    components::{WorkerRegistry, worker_registry::WorkerStatus},
    tests::create_test_worker_info,
};

#[test]
fn test_worker_registry_new() {
    let registry: WorkerRegistry<String> = WorkerRegistry::new();
    assert_eq!(registry.get_all_worker_ids().len(), 0);
    assert_eq!(registry.get_all_models().len(), 0);
}

#[test]
fn test_register_worker() {
    let registry: WorkerRegistry<String> = WorkerRegistry::new();
    let worker_id = Uuid::new_v4();
    let model_name = "test_model".to_string();
    let node_id = Uuid::new_v4();
    let device_id = Uuid::new_v4();
    let device_params = "1024".to_string(); // 内存大小参数

    let worker_info = create_test_worker_info(
        worker_id,
        model_name.clone(),
        node_id,
        device_id,
        device_params,
    );

    // 注册新worker应该成功
    assert!(registry.register_worker(worker_info.clone()));

    // 重复注册同一worker应该失败
    assert!(!registry.register_worker(worker_info));

    // 验证worker已注册
    assert!(registry.contains_worker(worker_id));
    assert_eq!(registry.get_all_worker_ids().len(), 1);
    assert_eq!(registry.get_model_workers(&model_name).len(), 1);
}

#[tokio::test]
async fn committed_status_is_visible_before_notification_tasks_can_run() {
    use super::super::worker_registry::WorkerStatus;
    let registry: WorkerRegistry<String> = WorkerRegistry::new();
    let id = Uuid::new_v4();
    registry.register_worker(create_test_worker_info(
        id,
        "m".into(),
        Uuid::new_v4(),
        Uuid::new_v4(),
        "slice".into(),
    ));
    registry.update_worker_status(id, WorkerStatus::Pausing);
    registry.update_worker_status(id, WorkerStatus::Paused);
    // A caller must observe the registry's already committed state without
    // depending on separately spawned notification tasks being scheduled.
    let mut listener = registry.get_status_listener(id);
    let waiting = listener.wait_for_worker_status(WorkerStatus::Paused, None);
    tokio::pin!(waiting);
    assert!(matches!(
        futures::poll!(&mut waiting),
        std::task::Poll::Ready(Ok(()))
    ));
}

#[test]
fn test_unregister_worker() {
    let registry: WorkerRegistry<String> = WorkerRegistry::new();
    let worker_id = Uuid::new_v4();
    let model_name = "test_model".to_string();
    let node_id = Uuid::new_v4();
    let device_id = Uuid::new_v4();

    let worker_info = create_test_worker_info(
        worker_id,
        model_name.clone(),
        node_id,
        device_id,
        "2048".to_string(),
    );

    // 注册worker
    registry.register_worker(worker_info);
    assert!(registry.contains_worker(worker_id));

    // 注销worker
    let removed_info = registry.unregister_worker(worker_id);
    assert!(removed_info.is_some());
    assert!(!registry.contains_worker(worker_id));
    assert_eq!(registry.get_model_workers(&model_name).len(), 0);

    // 再次注销应该返回None
    assert!(registry.unregister_worker(worker_id).is_none());
}

#[tokio::test]
async fn test_get_model_workers() {
    let registry: WorkerRegistry<String> = WorkerRegistry::new();
    let model_name = "test_model".to_string();
    let node_id = Uuid::new_v4();
    let device_id = Uuid::new_v4();

    // 创建多个worker
    let worker1_id = Uuid::new_v4();
    let worker2_id = Uuid::new_v4();
    let worker3_id = Uuid::new_v4();

    let worker1 = create_test_worker_info(
        worker1_id,
        model_name.clone(),
        node_id,
        device_id,
        "1024".to_string(),
    );

    let worker2 = create_test_worker_info(
        worker2_id,
        model_name.clone(),
        node_id,
        device_id,
        "2048".to_string(),
    );

    let worker3 = create_test_worker_info(
        worker3_id,
        "other_model".to_string(),
        node_id,
        device_id,
        "4096".to_string(),
    );

    registry.register_worker(worker1);
    registry.register_worker(worker2);
    registry.register_worker(worker3);

    let model_workers = registry.get_model_workers(&model_name);
    assert_eq!(model_workers.len(), 2);
    assert!(model_workers.contains(&worker1_id));
    assert!(model_workers.contains(&worker2_id));
    assert!(!model_workers.contains(&worker3_id));
}

#[tokio::test]
async fn test_update_worker_status() {
    let registry: WorkerRegistry<String> = WorkerRegistry::new();
    let worker_id = Uuid::new_v4();

    let worker_info = create_test_worker_info(
        worker_id,
        "test_model".to_string(),
        Uuid::new_v4(),
        Uuid::new_v4(),
        "1024".to_string(),
    );

    registry.register_worker(worker_info);

    // 更新状态应该成功
    assert!(registry.update_worker_status(worker_id, WorkerStatus::Running));

    // 验证状态已更新
    let info = registry.get_worker_info(worker_id).unwrap();
    assert_eq!(info.status, WorkerStatus::Running);

    // 更新不存在的worker应该失败
    let non_existent_id = Uuid::new_v4();
    assert!(!registry.update_worker_status(non_existent_id, WorkerStatus::Stopped));
}

#[test]
fn test_get_workers_by_status() {
    let registry: WorkerRegistry<String> = WorkerRegistry::new();
    let node_id = Uuid::new_v4();
    let device_id = Uuid::new_v4();

    let worker1_id = Uuid::new_v4();
    let worker2_id = Uuid::new_v4();
    let worker3_id = Uuid::new_v4();

    let mut worker1 = create_test_worker_info(
        worker1_id,
        "model1".to_string(),
        node_id,
        device_id,
        "1024".to_string(),
    );
    worker1.status = WorkerStatus::Starting;

    let worker2 = create_test_worker_info(
        worker2_id,
        "model2".to_string(),
        node_id,
        device_id,
        "2048".to_string(),
    );

    let worker3 = create_test_worker_info(
        worker3_id,
        "model3".to_string(),
        node_id,
        device_id,
        "4096".to_string(),
    );

    registry.register_worker(worker1);
    registry.register_worker(worker2);
    registry.register_worker(worker3);

    let running_workers = registry.get_workers_by_status(WorkerStatus::Running);
    assert_eq!(running_workers.len(), 2);
    assert!(running_workers.contains(&worker2_id));
    assert!(running_workers.contains(&worker3_id));

    let starting_workers = registry.get_workers_by_status(WorkerStatus::Starting);
    assert_eq!(starting_workers.len(), 1);
    assert!(starting_workers.contains(&worker1_id));
}

#[test]
fn test_get_statistics() {
    let registry: WorkerRegistry<String> = WorkerRegistry::new();
    let node_id = Uuid::new_v4();
    let device_id = Uuid::new_v4();

    // 添加一些workers
    for i in 0..5 {
        let mut worker_info = create_test_worker_info(
            Uuid::new_v4(),
            format!("model_{}", i % 2),
            node_id,
            device_id,
            "1024".to_string(),
        );
        worker_info.status = if i < 3 {
            WorkerStatus::Running
        } else {
            WorkerStatus::Starting
        };
        registry.register_worker(worker_info);
    }

    let stats = registry.get_statistics();
    assert_eq!(stats.total_workers, 5);
    assert_eq!(stats.model_count, 2); // model_0 和 model_1
    assert_eq!(stats.device_count, 1);
    assert_eq!(stats.node_count, 1);
    assert_eq!(stats.status_counts.get(&WorkerStatus::Running), Some(&3));
    assert_eq!(stats.status_counts.get(&WorkerStatus::Starting), Some(&2));
}

#[test]
fn test_cleanup_empty_indices() {
    let registry: WorkerRegistry<String> = WorkerRegistry::new();
    let worker_id = Uuid::new_v4();
    let model_name = "test_model".to_string();

    let worker_info = create_test_worker_info(
        worker_id,
        model_name.clone(),
        Uuid::new_v4(),
        Uuid::new_v4(),
        "1024".to_string(),
    );

    // 注册然后注销worker
    registry.register_worker(worker_info);
    registry.unregister_worker(worker_id);

    // 在清理前，模型索引可能仍然存在但为空
    registry.cleanup_empty_indices();

    // 验证空索引被清理
    assert_eq!(registry.get_model_workers(&model_name).len(), 0);
    assert_eq!(registry.get_all_models().len(), 0);
}

#[test]
fn test_device_and_node_indexing() {
    let registry: WorkerRegistry<String> = WorkerRegistry::new();
    let node_id = Uuid::new_v4();
    let device_id = Uuid::new_v4();
    let worker_id = Uuid::new_v4();

    let worker_info = create_test_worker_info(
        worker_id,
        "test_model".to_string(),
        node_id,
        device_id,
        "1024".to_string(),
    );

    registry.register_worker(worker_info);

    // 测试设备索引
    let device_workers = registry.get_device_workers(device_id);
    assert_eq!(device_workers.len(), 1);
    assert!(device_workers.contains(&worker_id));

    // 测试节点索引
    let node_workers = registry.get_node_workers(node_id);
    assert_eq!(node_workers.len(), 1);
    assert!(node_workers.contains(&worker_id));

    // 测试获取worker的设备和节点信息
    assert_eq!(registry.get_worker_node(worker_id), Some(node_id));
    assert_eq!(
        registry.get_worker_device_params(worker_id),
        Some("1024".to_string())
    );
}

#[test]
fn test_worker_counts_by_model() {
    let registry: WorkerRegistry<String> = WorkerRegistry::new();
    let node_id = Uuid::new_v4();
    let device_id = Uuid::new_v4();

    // 为不同模型添加不同数量的worker
    for _i in 0..3 {
        let worker_info = create_test_worker_info(
            Uuid::new_v4(),
            "model_a".to_string(),
            node_id,
            device_id,
            "1024".to_string(),
        );
        registry.register_worker(worker_info);
    }

    for _i in 0..2 {
        let worker_info = create_test_worker_info(
            Uuid::new_v4(),
            "model_b".to_string(),
            node_id,
            device_id,
            "2048".to_string(),
        );
        registry.register_worker(worker_info);
    }

    let worker_counts = registry.get_worker_counts_by_model();
    assert_eq!(worker_counts.get("model_a"), Some(&3));
    assert_eq!(worker_counts.get("model_b"), Some(&2));
    assert_eq!(registry.get_model_worker_count(&"model_a".to_string()), 3);
    assert_eq!(registry.get_model_worker_count(&"model_b".to_string()), 2);
    assert_eq!(
        registry.get_model_worker_count(&"non_existent".to_string()),
        0
    );
}

#[tokio::test]
async fn test_paused_status_handling() {
    let registry: WorkerRegistry<String> = WorkerRegistry::new();
    let worker_id = Uuid::new_v4();

    let worker_info = create_test_worker_info(
        worker_id,
        "test_model".to_string(),
        Uuid::new_v4(),
        Uuid::new_v4(),
        "1024".to_string(),
    );

    registry.register_worker(worker_info);

    // 测试状态转换：Running -> Pausing -> Paused
    assert!(registry.update_worker_status(worker_id, WorkerStatus::Running));
    let info = registry.get_worker_info(worker_id).unwrap();
    assert_eq!(info.status, WorkerStatus::Running);

    assert!(registry.update_worker_status(worker_id, WorkerStatus::Pausing));
    let info = registry.get_worker_info(worker_id).unwrap();
    assert_eq!(info.status, WorkerStatus::Pausing);

    assert!(registry.update_worker_status(worker_id, WorkerStatus::Paused));
    let info = registry.get_worker_info(worker_id).unwrap();
    assert_eq!(info.status, WorkerStatus::Paused);

    // 测试获取处于Paused状态的workers
    let paused_workers = registry.get_workers_by_status(WorkerStatus::Paused);
    assert_eq!(paused_workers.len(), 1);
    assert!(paused_workers.contains(&worker_id));
}

#[test]
fn test_worker_status_statistics_with_paused() {
    let registry: WorkerRegistry<String> = WorkerRegistry::new();
    let node_id = Uuid::new_v4();
    let device_id = Uuid::new_v4();

    // 创建不同状态的workers
    let statuses = vec![
        WorkerStatus::Running,
        WorkerStatus::Running,
        WorkerStatus::Pausing,
        WorkerStatus::Paused,
        WorkerStatus::Paused,
        WorkerStatus::Stopping,
    ];

    for (i, status) in statuses.iter().enumerate() {
        let mut worker_info = create_test_worker_info(
            Uuid::new_v4(),
            format!("model_{}", i),
            node_id,
            device_id,
            "1024".to_string(),
        );
        worker_info.status = *status;
        registry.register_worker(worker_info);
    }

    let stats = registry.get_statistics();
    assert_eq!(stats.total_workers, 6);
    assert_eq!(stats.status_counts.get(&WorkerStatus::Running), Some(&2));
    assert_eq!(stats.status_counts.get(&WorkerStatus::Pausing), Some(&1));
    assert_eq!(stats.status_counts.get(&WorkerStatus::Paused), Some(&2));
    assert_eq!(stats.status_counts.get(&WorkerStatus::Stopping), Some(&1));
}
