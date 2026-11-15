use std::{sync::Arc, time::Duration};

use uuid::Uuid;

use super::super::worker_registry::WorkerRegistry;
use crate::compute_node::r#virtual::virtual_compute_node::{
    components::{ModelServiceManager, model_service_manager::ArrivalRateTracker},
    tests::create_test_worker_info,
};

#[test]
fn test_arrival_rate_tracker_new() {
    let tracker = ArrivalRateTracker::new();
    assert_eq!(tracker.get_arrival_rate(), 0.0);
    assert_eq!(tracker.get_recent_arrivals_count(), 0);
}

#[test]
fn test_arrival_rate_tracker_record_arrival() {
    let mut tracker = ArrivalRateTracker::new();
    let now = std::time::Instant::now();

    // 记录一些到达事件
    tracker.record_arrival(now);
    tracker.record_arrival(now + Duration::from_millis(100));
    tracker.record_arrival(now + Duration::from_millis(200));

    assert_eq!(tracker.get_recent_arrivals_count(), 3);
    assert!(tracker.get_arrival_rate() > 0.0);
}

#[test]
fn test_arrival_rate_tracker_cleanup_old_data() {
    let mut tracker = ArrivalRateTracker::new();
    let old_time = std::time::Instant::now() - Duration::from_secs(10);
    let recent_time = std::time::Instant::now();

    // 添加旧数据
    tracker.record_arrival(old_time);
    tracker.record_arrival(old_time + Duration::from_millis(100));

    // 添加新数据
    tracker.record_arrival(recent_time);

    // 清理旧数据应该只保留新数据
    tracker.cleanup_old_arrival_data(std::time::Instant::now() - Duration::from_secs(5));
    assert_eq!(tracker.get_recent_arrivals_count(), 1);
}

#[test]
fn test_model_service_manager_new() {
    let registry = Arc::new(WorkerRegistry::<String>::new());
    let manager = ModelServiceManager::new(registry.clone());

    assert!(manager.get_all_arrival_rates().is_empty());
    assert!(manager.list_models().is_empty());
}

#[test]
fn test_record_task_arrival() {
    let registry = Arc::new(WorkerRegistry::<String>::new());
    let manager = ModelServiceManager::new(registry.clone());
    let model_id = "test_model".to_string();

    // 记录任务到达
    manager.record_task_arrival(&model_id);
    manager.record_task_arrival(&model_id);

    // 验证到达率被记录
    let arrival_rate = manager.get_arrival_rate(&model_id);
    assert!(arrival_rate.is_some());
    assert!(arrival_rate.unwrap() >= 0.0);

    let all_rates = manager.get_all_arrival_rates();
    assert_eq!(all_rates.len(), 1);
    assert!(all_rates.contains_key(&model_id));
}

#[test]
fn test_get_model_workers() {
    let registry = Arc::new(WorkerRegistry::<String>::new());
    let manager = ModelServiceManager::new(registry.clone());
    let model_id = "test_model".to_string();
    let node_id = Uuid::new_v4();
    let device_id = Uuid::new_v4();

    // 添加一些workers到注册表
    let worker1_id = Uuid::new_v4();
    let worker2_id = Uuid::new_v4();

    let worker1 = create_test_worker_info(
        worker1_id,
        model_id.clone(),
        node_id,
        device_id,
        "1024".to_string(),
    );

    let worker2 = create_test_worker_info(
        worker2_id,
        model_id.clone(),
        node_id,
        device_id,
        "2048".to_string(),
    );

    registry.register_worker(worker1);
    registry.register_worker(worker2);

    // 通过manager获取workers
    let workers = manager.get_model_workers(&model_id);
    assert_eq!(workers.len(), 2);
    assert!(workers.contains(&worker1_id));
    assert!(workers.contains(&worker2_id));
}

#[test]
fn test_list_models() {
    let registry = Arc::new(WorkerRegistry::<String>::new());
    let manager = ModelServiceManager::new(registry.clone());
    let node_id = Uuid::new_v4();
    let device_id = Uuid::new_v4();

    // 添加不同模型的workers
    let worker1 = create_test_worker_info(
        Uuid::new_v4(),
        "model_a".to_string(),
        node_id,
        device_id,
        "1024".to_string(),
    );

    let worker2 = create_test_worker_info(
        Uuid::new_v4(),
        "model_b".to_string(),
        node_id,
        device_id,
        "2048".to_string(),
    );

    registry.register_worker(worker1);
    registry.register_worker(worker2);

    let models = manager.list_models();
    assert_eq!(models.len(), 2);
    assert!(models.contains(&"model_a".to_string()));
    assert!(models.contains(&"model_b".to_string()));
}

#[test]
fn test_get_model_worker_count() {
    let registry = Arc::new(WorkerRegistry::<String>::new());
    let manager = ModelServiceManager::new(registry.clone());
    let model_id = "test_model".to_string();
    let node_id = Uuid::new_v4();
    let device_id = Uuid::new_v4();

    // 初始时应该没有workers
    assert_eq!(manager.get_model_worker_count(&model_id), 0);

    // 添加一些workers
    for i in 0..3 {
        let worker = create_test_worker_info(
            Uuid::new_v4(),
            model_id.clone(),
            node_id,
            device_id,
            format!("{}", 1024 * (i + 1)),
        );
        registry.register_worker(worker);
    }

    assert_eq!(manager.get_model_worker_count(&model_id), 3);
}

#[test]
fn test_get_worker_counts() {
    let registry = Arc::new(WorkerRegistry::<String>::new());
    let manager = ModelServiceManager::new(registry.clone());
    let node_id = Uuid::new_v4();
    let device_id = Uuid::new_v4();

    // 为不同模型添加不同数量的workers
    for i in 0..2 {
        let worker = create_test_worker_info(
            Uuid::new_v4(),
            "model_a".to_string(),
            node_id,
            device_id,
            format!("{}", 1024 * (i + 1)),
        );
        registry.register_worker(worker);
    }

    for i in 0..3 {
        let worker = create_test_worker_info(
            Uuid::new_v4(),
            "model_b".to_string(),
            node_id,
            device_id,
            format!("{}", 1024 * (i + 1)),
        );
        registry.register_worker(worker);
    }

    let worker_counts = manager.get_worker_counts();
    assert_eq!(worker_counts.get("model_a"), Some(&2));
    assert_eq!(worker_counts.get("model_b"), Some(&3));
}

#[test]
fn test_get_model_state() {
    let registry = Arc::new(WorkerRegistry::<String>::new());
    let manager = ModelServiceManager::new(registry.clone());
    let model_id = "test_model".to_string();
    let node_id = Uuid::new_v4();
    let device_id = Uuid::new_v4();

    // 添加一些workers
    let worker1 = create_test_worker_info(
        Uuid::new_v4(),
        model_id.clone(),
        node_id,
        device_id,
        "1024".to_string(),
    );

    let worker2 = create_test_worker_info(
        Uuid::new_v4(),
        model_id.clone(),
        node_id,
        device_id,
        "2048".to_string(),
    );

    registry.register_worker(worker1);
    registry.register_worker(worker2);

    let (model_allocation, device_slices) = manager.get_model_state();

    // 验证模型分配
    assert!(model_allocation.contains_key(&model_id));
    let model_params = model_allocation.get(&model_id).unwrap();
    assert_eq!(model_params.len(), 2);

    // 验证设备分组
    assert!(!device_slices.is_empty());
}

#[test]
fn test_cleanup_old_arrival_data() {
    let registry = Arc::new(WorkerRegistry::<String>::new());
    let manager = ModelServiceManager::new(registry.clone());
    let model_id = "test_model".to_string();

    // 记录一些到达事件
    manager.record_task_arrival(&model_id);
    manager.record_task_arrival(&model_id);

    let cutoff_time = std::time::Instant::now() + Duration::from_secs(10);
    manager.cleanup_old_arrival_data(cutoff_time);

    // 验证数据被清理
    let arrival_rate = manager.get_arrival_rate(&model_id);
    assert!(arrival_rate.is_some());
}

#[test]
fn test_get_arrival_stats() {
    let registry = Arc::new(WorkerRegistry::<String>::new());
    let manager = ModelServiceManager::new(registry.clone());
    let model_id1 = "model_1".to_string();
    let model_id2 = "model_2".to_string();

    // 为不同模型记录到达事件
    manager.record_task_arrival(&model_id1);
    manager.record_task_arrival(&model_id1);
    manager.record_task_arrival(&model_id2);

    let stats = manager.get_arrival_stats();
    assert_eq!(stats.len(), 2);
    assert!(stats.contains_key(&model_id1));
    assert!(stats.contains_key(&model_id2));

    let (rate1, count1) = stats.get(&model_id1).unwrap();
    let (rate2, count2) = stats.get(&model_id2).unwrap();
    assert!(*rate1 >= 0.0);
    assert!(*rate2 >= 0.0);
    assert_eq!(*count1, 2);
    assert_eq!(*count2, 1);
}

#[test]
fn test_reset_model_arrival_tracking() {
    let registry = Arc::new(WorkerRegistry::<String>::new());
    let manager = ModelServiceManager::new(registry.clone());
    let model_id = "test_model".to_string();

    // 记录一些到达事件
    manager.record_task_arrival(&model_id);
    manager.record_task_arrival(&model_id);

    // 验证有数据
    let arrival_rate = manager.get_arrival_rate(&model_id).unwrap();
    assert!(arrival_rate >= 0.0);

    // 重置跟踪
    manager.reset_model_arrival_tracking(&model_id);

    // 验证数据被重置
    let reset_rate = manager.get_arrival_rate(&model_id).unwrap();
    assert_eq!(reset_rate, 0.0);
}

#[test]
fn test_remove_model_arrival_tracking() {
    let registry = Arc::new(WorkerRegistry::<String>::new());
    let manager = ModelServiceManager::new(registry.clone());
    let model_id = "test_model".to_string();

    // 记录到达事件
    manager.record_task_arrival(&model_id);
    assert!(manager.get_arrival_rate(&model_id).is_some());

    // 移除跟踪
    assert!(manager.remove_model_arrival_tracking(&model_id));
    assert!(manager.get_arrival_rate(&model_id).is_none());

    // 再次移除应该返回false
    assert!(!manager.remove_model_arrival_tracking(&model_id));
}
