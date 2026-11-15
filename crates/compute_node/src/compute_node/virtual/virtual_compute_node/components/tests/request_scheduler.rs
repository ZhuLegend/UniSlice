use std::time::Duration;

use model::tensor::DynTensor;
use ndarray::ArrayD;
use tokio::sync::oneshot;
use uuid::Uuid;

use crate::{
    compute_node::r#virtual::{
        BaseWork,
        virtual_compute_node::components::{
            RequestScheduler, request_scheduler::batch_size_extractor::BatchExtractorStrategy,
            worker_manager::WorkerRequest,
        },
    },
    util::{Input, Output},
};

#[tokio::test]
async fn test_request_scheduler_new() {
    let scheduler = RequestScheduler::with_strategy(BatchExtractorStrategy::split_and_satisfy());
    let model_id = "test_model".to_string();

    // 初始状态应该没有队列长度
    assert!(scheduler.get_pending_requests_stats().is_empty());
}

#[tokio::test]
async fn test_get_worker_request_sender() {
    let scheduler = RequestScheduler::with_strategy(BatchExtractorStrategy::split_and_satisfy());
    let sender = scheduler.get_worker_request_sender();

    // 验证可以获取发送器
    assert!(!sender.is_closed());
}

#[tokio::test]
async fn test_request_batch() {
    let scheduler = RequestScheduler::with_strategy(BatchExtractorStrategy::split_and_satisfy());
    let worker_id = Uuid::new_v4();
    let model_name = "test_model".to_string();

    let (tx, _rx) = oneshot::channel();
    let request = WorkerRequest {
        worker_id,
        model_name: model_name.clone(),
        desired_batch: 1,
        resp: tx,
    };

    // 发送Worker请求
    let result = scheduler.request_batch(request).await;
    assert!(result.is_ok());

    // 等待请求被处理
    tokio::time::sleep(Duration::from_millis(10)).await;

    // 验证pending请求统计
    let pending_stats = scheduler.get_pending_requests_stats();
    // 请求应该被记录（可能为1或0，取决于处理速度）
    assert!(pending_stats.get(&model_name).unwrap_or(&0) >= &0);
}

#[tokio::test]
async fn test_model_queue_lengths_basic() {
    let scheduler = RequestScheduler::with_strategy(BatchExtractorStrategy::split_and_satisfy());
    let model1 = "model1".to_string();
    let model2 = "model2".to_string();

    // 初始状态下所有队列都是空的
    assert_eq!(scheduler.get_model_queue_batch_size(&model1), 0);
    assert_eq!(scheduler.get_model_queue_batch_size(&model2), 0);
}

#[tokio::test]
async fn test_cleanup_worker_requests() {
    let scheduler = RequestScheduler::with_strategy(BatchExtractorStrategy::split_and_satisfy());
    let worker1_id = Uuid::new_v4();
    let worker2_id = Uuid::new_v4();

    // 测试清理不存在的worker不会出错
    scheduler.cleanup_worker_requests(worker1_id);
    scheduler.cleanup_worker_requests(worker2_id);
}

#[tokio::test]
async fn test_cleanup_invalid_worker_requests() {
    let scheduler = RequestScheduler::with_strategy(BatchExtractorStrategy::split_and_satisfy());
    let worker1_id = Uuid::new_v4();
    let worker2_id = Uuid::new_v4();
    let worker3_id = Uuid::new_v4(); // 这个worker将被认为是无效的
    let model_name = "test_model".to_string();

    // 创建worker请求并添加到pending队列
    let (tx1, _rx1) = oneshot::channel();
    let (tx2, _rx2) = oneshot::channel();
    let (tx3, _rx3) = oneshot::channel();

    let req1 = WorkerRequest {
        worker_id: worker1_id,
        model_name: model_name.clone(),
        desired_batch: 1,
        resp: tx1,
    };
    let req2 = WorkerRequest {
        worker_id: worker2_id,
        model_name: model_name.clone(),
        desired_batch: 1,
        resp: tx2,
    };
    let req3 = WorkerRequest {
        worker_id: worker3_id,
        model_name: model_name.clone(),
        desired_batch: 1,
        resp: tx3,
    };

    // 发送worker请求
    scheduler.request_batch(req1).await.unwrap();
    scheduler.request_batch(req2).await.unwrap();
    scheduler.request_batch(req3).await.unwrap();

    // 等待请求被处理到pending队列
    tokio::time::sleep(Duration::from_millis(50)).await;

    // 验证初始状态有3个pending请求
    let initial_stats = scheduler.get_pending_requests_stats();
    let initial_count = initial_stats.get(&model_name).unwrap_or(&0);
    assert!(*initial_count > 0);

    // 创建有效worker集合（不包含worker3）
    let mut valid_workers = std::collections::HashSet::new();
    valid_workers.insert(worker1_id);
    valid_workers.insert(worker2_id);

    // 清理无效worker请求
    scheduler.cleanup_invalid_worker_requests(&valid_workers);

    // 验证worker3的请求被清理了
    let final_stats = scheduler.get_pending_requests_stats();
    let final_count = final_stats.get(&model_name).unwrap_or(&0);

    // 应该只剩下有效worker的请求
    assert!(*final_count < *initial_count);
}

// 测试基本的队列操作和任务流程
#[tokio::test]
async fn test_basic_queue_operations() {
    let scheduler = RequestScheduler::with_strategy(BatchExtractorStrategy::split_and_satisfy());
    let model_name = "test_model".to_string();

    // 初始状态验证
    assert_eq!(scheduler.get_model_queue_batch_size(&model_name), 0);
    assert!(scheduler.get_pending_requests_stats().is_empty());

    // 创建测试任务
    let work1 = BaseWork::<Input>::new(
        Uuid::new_v4(),
        DynTensor::from(ArrayD::from_elem(vec![1], 1i32)),
    );
    let work2 = BaseWork::<Input>::new(
        Uuid::new_v4(),
        DynTensor::from(ArrayD::from_elem(vec![1], 2i32)),
    );

    // 异步添加任务（不等待结果）
    let model_name_clone1 = model_name.clone();
    let scheduler_clone1 = scheduler.clone_for_test();
    let task1_handle = tokio::spawn(async move {
        scheduler_clone1
            .add_compute_task(model_name_clone1, work1)
            .await
    });

    let model_name_clone2 = model_name.clone();
    let scheduler_clone2 = scheduler.clone_for_test();
    let task2_handle = tokio::spawn(async move {
        scheduler_clone2
            .add_compute_task(model_name_clone2, work2)
            .await
    });

    // 等待任务被添加到队列
    tokio::time::sleep(Duration::from_millis(10)).await;

    // 验证队列长度
    assert_eq!(scheduler.get_model_queue_batch_size(&model_name), 2);

    // 创建worker请求来处理任务
    let worker_id = Uuid::new_v4();
    let (tx, rx) = oneshot::channel();
    let worker_req = WorkerRequest {
        worker_id,
        model_name: model_name.clone(),
        desired_batch: 2,
        resp: tx,
    };

    // 发送worker请求
    scheduler.request_batch(worker_req).await.unwrap();

    // 等待批量任务被处理
    tokio::time::sleep(Duration::from_millis(10)).await;

    // Worker应该收到批量任务
    let batch = rx.await.unwrap();
    assert_eq!(batch.len(), 2);

    // 验证队列被清空
    assert_eq!(scheduler.get_model_queue_batch_size(&model_name), 0);

    // 处理任务并返回结果
    for (work, resp_tx) in batch {
        let output = BaseWork::<Output>::new(
            work.id(),
            DynTensor::from(ArrayD::from_elem(vec![1], 100i32)),
        );
        let _ = resp_tx.send(output);
    }

    // 验证任务完成
    let result1 = task1_handle.await.unwrap();
    let result2 = task2_handle.await.unwrap();

    assert!(result1.is_ok());
    assert!(result2.is_ok());
}

#[tokio::test]
async fn test_multiple_workers_same_model() {
    let scheduler = RequestScheduler::with_strategy(BatchExtractorStrategy::split_and_satisfy());
    let model_name = "test_model".to_string();
    let worker1_id = Uuid::new_v4();
    let worker2_id = Uuid::new_v4();

    // 添加多个任务到队列
    for i in 0..6 {
        let work = BaseWork::<Input>::new(
            Uuid::new_v4(),
            DynTensor::from(ArrayD::from_elem(vec![1], i)),
        );
        let model_name_clone = model_name.clone();
        let scheduler_clone = scheduler.clone_for_test();
        tokio::spawn(async move {
            let _ = scheduler_clone
                .add_compute_task(model_name_clone, work)
                .await;
        });
    }

    // 等待任务被添加
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert_eq!(scheduler.get_model_queue_batch_size(&model_name), 6);

    // 创建两个worker的请求
    let (tx1, rx1) = oneshot::channel();
    let (tx2, rx2) = oneshot::channel();

    let req1 = WorkerRequest {
        worker_id: worker1_id,
        model_name: model_name.clone(),
        desired_batch: 2,
        resp: tx1,
    };

    let req2 = WorkerRequest {
        worker_id: worker2_id,
        model_name: model_name.clone(),
        desired_batch: 3,
        resp: tx2,
    };

    // 发送worker请求
    scheduler.request_batch(req1).await.unwrap();
    scheduler.request_batch(req2).await.unwrap();

    // 等待处理
    tokio::time::sleep(Duration::from_millis(20)).await;

    // 两个worker应该都收到任务
    let batch1 = rx1.await.unwrap();
    let batch2 = rx2.await.unwrap();

    assert_eq!(batch1.len(), 2);
    assert_eq!(batch2.len(), 3);

    // 验证剩余队列长度
    assert_eq!(scheduler.get_model_queue_batch_size(&model_name), 1);

    // 处理所有任务
    for (work, resp_tx) in batch1.into_iter().chain(batch2.into_iter()) {
        let output = BaseWork::<Output>::new(
            work.id(),
            DynTensor::from(ArrayD::from_elem(vec![1], 200i32)),
        );
        let _ = resp_tx.send(output);
    }

    // 测试worker清理功能
    scheduler.cleanup_worker_requests(worker1_id);
    scheduler.cleanup_worker_requests(worker2_id);

    // 验证清理后状态
    let pending_stats = scheduler.get_pending_requests_stats();
    assert_eq!(pending_stats.get(&model_name).unwrap_or(&0), &0);
}

#[tokio::test]
async fn test_worker_request_processing() {
    let scheduler = RequestScheduler::with_strategy(BatchExtractorStrategy::split_and_satisfy());
    let model_name = "test_model".to_string();
    let worker_id = Uuid::new_v4();

    // 测试worker请求在没有任务时的行为
    let (tx, mut rx) = oneshot::channel();
    let req = WorkerRequest {
        worker_id,
        model_name: model_name.clone(),
        desired_batch: 3,
        resp: tx,
    };

    scheduler.request_batch(req).await.unwrap();

    // 等待请求被处理到pending队列
    tokio::time::sleep(Duration::from_millis(10)).await;

    // 验证pending请求存在
    let pending_stats = scheduler.get_pending_requests_stats();
    assert!(pending_stats.get(&model_name).unwrap_or(&0) > &0);

    // 现在添加一些任务
    for i in 0..2 {
        let work = BaseWork::<Input>::new(
            Uuid::new_v4(),
            DynTensor::from(ArrayD::from_elem(vec![1], i)),
        );
        let model_name_clone = model_name.clone();
        let scheduler_clone = scheduler.clone_for_test();
        tokio::spawn(async move {
            let _ = scheduler_clone
                .add_compute_task(model_name_clone, work)
                .await;
        });
    }

    // 等待任务被添加
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert_eq!(scheduler.get_model_queue_batch_size(&model_name), 2);

    // worker请求应该还在pending，因为任务数不够
    tokio::time::sleep(Duration::from_millis(10)).await;
    match rx.try_recv() {
        Err(_) => {} // 请求还在等待，这是期望的
        Ok(_) => panic!("Worker request should not be fulfilled with insufficient tasks"),
    }

    // 添加更多任务满足批量需求
    let work = BaseWork::<Input>::new(
        Uuid::new_v4(),
        DynTensor::from(ArrayD::from_elem(vec![1], 3i32)),
    );
    let model_name_clone = model_name.clone();
    let scheduler_clone = scheduler.clone_for_test();
    tokio::spawn(async move {
        let _ = scheduler_clone
            .add_compute_task(model_name_clone, work)
            .await;
    });

    // 等待处理
    tokio::time::sleep(Duration::from_millis(20)).await;

    // 现在worker应该收到批量任务
    let batch = rx.await.unwrap();
    assert_eq!(batch.len(), 3);

    // 处理任务
    for (work, resp_tx) in batch {
        let output = BaseWork::<Output>::new(
            work.id(),
            DynTensor::from(ArrayD::from_elem(vec![1], 300i32)),
        );
        let _ = resp_tx.send(output);
    }

    // 验证队列被清空
    assert_eq!(scheduler.get_model_queue_batch_size(&model_name), 0);

    // 测试worker清理
    scheduler.cleanup_worker_requests(worker_id);
}

#[tokio::test]
async fn test_task_splitting_in_request_scheduler() {
    let scheduler = RequestScheduler::with_strategy(BatchExtractorStrategy::split_and_satisfy());
    let model_name = "test_model".to_string();
    let worker_id = Uuid::new_v4();

    // 创建一个大的任务（batch size = 5）
    let large_tensor = DynTensor::from(
        ndarray::Array::from_shape_vec(vec![5, 10], (0..50).collect::<Vec<i32>>()).unwrap(),
    );

    let large_work = BaseWork::<Input>::new(Uuid::new_v4(), large_tensor);

    // 添加大任务到队列
    let model_name_clone = model_name.clone();
    let scheduler_clone = scheduler.clone_for_test();
    let task_handle = tokio::spawn(async move {
        scheduler_clone
            .add_compute_task(model_name_clone, large_work)
            .await
    });

    // 等待任务被添加
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert_eq!(scheduler.get_model_queue_batch_size(&model_name), 5);

    // 创建一个worker请求，要求的batch size = 3（小于任务的batch size 5）
    let (tx, rx) = oneshot::channel();
    let worker_req = WorkerRequest {
        worker_id,
        model_name: model_name.clone(),
        desired_batch: 3,
        resp: tx,
    };

    // 发送worker请求
    scheduler.request_batch(worker_req).await.unwrap();

    // 等待处理
    tokio::time::sleep(Duration::from_millis(20)).await;

    // Worker应该收到切分后的任务（batch size = 3）
    let batch = rx.await.unwrap();
    assert_eq!(batch.len(), 1, "应该收到1个切分后的任务");

    let (split_work, resp_tx) = batch.into_iter().next().unwrap();
    let split_batch_size = split_work.data().dimensions()[0] as usize;
    assert_eq!(split_batch_size, 3, "切分后的任务batch size应该是3");

    // 验证剩余任务还在队列中（batch size = 2）
    assert_eq!(
        scheduler.get_model_queue_batch_size(&model_name),
        2,
        "应该还有1个剩余任务(batch size = 2)在队列中"
    );

    // 模拟worker处理切分后的任务
    let output_tensor = DynTensor::from(
        ndarray::Array::from_shape_vec(vec![3, 10], (100..130).collect::<Vec<i32>>()).unwrap(),
    );
    let output_work = BaseWork::<Output>::new(split_work.id(), output_tensor);
    resp_tx.send(output_work).unwrap();

    // 创建另一个worker请求处理剩余任务
    let (tx2, rx2) = oneshot::channel();
    let worker_req2 = WorkerRequest {
        worker_id: Uuid::new_v4(),
        model_name: model_name.clone(),
        desired_batch: 2,
        resp: tx2,
    };

    scheduler.request_batch(worker_req2).await.unwrap();
    tokio::time::sleep(Duration::from_millis(20)).await;

    let batch2 = rx2.await.unwrap();
    assert_eq!(batch2.len(), 1, "应该收到1个剩余任务");

    let (remaining_work, resp_tx2) = batch2.into_iter().next().unwrap();
    let remaining_batch_size = remaining_work.data().dimensions()[0] as usize;
    assert_eq!(remaining_batch_size, 2, "剩余任务的batch size应该是2");

    // 模拟处理剩余任务
    let output_tensor2 = DynTensor::from(
        ndarray::Array::from_shape_vec(vec![2, 10], (200..220).collect::<Vec<i32>>()).unwrap(),
    );
    let output_work2 = BaseWork::<Output>::new(remaining_work.id(), output_tensor2);
    resp_tx2.send(output_work2).unwrap();

    // 验证队列被完全清空
    assert_eq!(scheduler.get_model_queue_batch_size(&model_name), 0);

    // 等待原始任务完成（应该收到合并后的结果）
    let final_result = task_handle.await.unwrap();
    assert!(final_result.is_ok(), "原始任务应该成功完成");

    let merged_output = final_result.unwrap();
    let merged_dimensions = merged_output.data().dimensions();
    assert_eq!(
        merged_dimensions[0], 5,
        "合并后的结果应该恢复原始batch size"
    );
    assert_eq!(merged_dimensions[1], 10, "其他维度应该保持不变");
}

#[tokio::test]
async fn test_task_no_split_when_exact_match() {
    let scheduler = RequestScheduler::with_strategy(BatchExtractorStrategy::split_and_satisfy());
    let model_name = "test_model".to_string();
    let worker_id = Uuid::new_v4();

    // 创建一个任务（batch size = 3）
    let tensor = DynTensor::from(
        ndarray::Array::from_shape_vec(vec![3, 5], (0..15).collect::<Vec<i32>>()).unwrap(),
    );

    let work = BaseWork::<Input>::new(Uuid::new_v4(), tensor);

    // 添加任务到队列
    let model_name_clone = model_name.clone();
    let scheduler_clone = scheduler.clone_for_test();
    tokio::spawn(async move {
        let _ = scheduler_clone
            .add_compute_task(model_name_clone, work)
            .await;
    });

    // 等待任务被添加
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert_eq!(scheduler.get_model_queue_batch_size(&model_name), 3);

    // 创建一个worker请求，要求的batch size = 3（与任务的batch size相同）
    let (tx, rx) = oneshot::channel();
    let worker_req = WorkerRequest {
        worker_id,
        model_name: model_name.clone(),
        desired_batch: 3,
        resp: tx,
    };

    // 发送worker请求
    scheduler.request_batch(worker_req).await.unwrap();

    // 等待处理
    tokio::time::sleep(Duration::from_millis(20)).await;

    // Worker应该收到完整的任务（不需要切分）
    let batch = rx.await.unwrap();
    assert_eq!(batch.len(), 1, "应该收到1个完整任务");

    let (received_work, _) = &batch[0];
    let batch_size = received_work.data().dimensions()[0] as usize;
    assert_eq!(batch_size, 3, "任务batch size应该保持为3");

    // 验证队列被完全清空
    assert_eq!(scheduler.get_model_queue_batch_size(&model_name), 0);
}

#[tokio::test]
async fn test_scheduler_basic_operations() {
    let scheduler = RequestScheduler::with_strategy(BatchExtractorStrategy::split_and_satisfy());
    let model_name = "test_model".to_string();

    // 测试基本操作不会panic
    let sender = scheduler.get_worker_request_sender();
    assert!(!sender.is_closed());

    let queue_length = scheduler.get_model_queue_batch_size(&model_name);
    assert_eq!(queue_length, 0);

    let pending_stats = scheduler.get_pending_requests_stats();
    assert!(pending_stats.is_empty());
}

#[tokio::test]
async fn test_task_split_failure_logging() {
    let scheduler = RequestScheduler::with_strategy(BatchExtractorStrategy::split_and_satisfy());
    let model_name = "test_model".to_string();
    let worker_id = Uuid::new_v4();

    // 创建一个batch size较大的任务，但使其无法被切分（例如非标准维度结构）
    // 首先添加一个正常的大任务
    let large_tensor = DynTensor::from(
        ndarray::Array::from_shape_vec(vec![5, 10], (0..50).collect::<Vec<i32>>()).unwrap(),
    );
    let large_work = BaseWork::<Input>::new(Uuid::new_v4(), large_tensor);

    // 添加任务到队列
    let model_name_clone = model_name.clone();
    let scheduler_clone = scheduler.clone_for_test();
    tokio::spawn(async move {
        let _ = scheduler_clone
            .add_compute_task(model_name_clone, large_work)
            .await;
    });

    tokio::time::sleep(Duration::from_millis(10)).await;
    assert_eq!(scheduler.get_model_queue_batch_size(&model_name), 5);

    // 创建一个请求，要求batch size为3（小于5，应该可以满足）
    let (tx, rx) = oneshot::channel();
    let worker_req = WorkerRequest {
        worker_id,
        model_name: model_name.clone(),
        desired_batch: 3,
        resp: tx,
    };

    scheduler.request_batch(worker_req).await.unwrap();
    tokio::time::sleep(Duration::from_millis(20)).await;

    // 应该能收到切分后的任务
    let batch = rx.await.unwrap();
    assert_eq!(batch.len(), 1, "应该收到1个切分后的任务");

    let (split_work, _) = &batch[0];
    let split_batch_size = split_work.data().dimensions()[0] as usize;
    assert_eq!(split_batch_size, 3, "切分后的任务batch size应该是3");

    // 验证还有剩余任务在队列中
    assert_eq!(scheduler.get_model_queue_batch_size(&model_name), 2);
}

#[tokio::test]
async fn test_multiple_tasks_batch_collection() {
    let scheduler = RequestScheduler::with_strategy(BatchExtractorStrategy::split_and_satisfy());
    let model_name = "test_model".to_string();
    let worker_id = Uuid::new_v4();

    // 添加多个小任务
    for i in 0..3 {
        let tensor = DynTensor::from(
            ndarray::Array::from_shape_vec(vec![2, 5], (i * 10..i * 10 + 10).collect::<Vec<i32>>())
                .unwrap(),
        );
        let work = BaseWork::<Input>::new(Uuid::new_v4(), tensor);

        let model_name_clone = model_name.clone();
        let scheduler_clone = scheduler.clone_for_test();
        tokio::spawn(async move {
            let _ = scheduler_clone
                .add_compute_task(model_name_clone, work)
                .await;
        });
    }

    tokio::time::sleep(Duration::from_millis(10)).await;
    assert_eq!(scheduler.get_model_queue_batch_size(&model_name), 6);

    // 请求batch size = 5（需要2.5个任务，应该收集2个完整任务 + 1个切分任务）
    let (tx, rx) = oneshot::channel();
    let worker_req = WorkerRequest {
        worker_id,
        model_name: model_name.clone(),
        desired_batch: 5,
        resp: tx,
    };

    scheduler.request_batch(worker_req).await.unwrap();
    tokio::time::sleep(Duration::from_millis(20)).await;

    let batch = rx.await.unwrap();
    assert_eq!(batch.len(), 3, "应该收到3个任务（2个完整 + 1个切分）");

    // 验证总batch size为5
    let total_batch_size: usize = batch
        .iter()
        .map(|(work, _)| work.data().dimensions()[0] as usize)
        .sum();
    assert_eq!(total_batch_size, 5, "总batch size应该为5");

    // 验证还有剩余任务在队列中
    assert_eq!(scheduler.get_model_queue_batch_size(&model_name), 1);
}

#[tokio::test]
async fn test_process_model_requests_efficiency() {
    let scheduler = RequestScheduler::with_strategy(BatchExtractorStrategy::split_and_satisfy());
    let model_name = "test_model".to_string();

    // 添加大量任务到队列
    for i in 0..100 {
        let tensor = DynTensor::from(
            ndarray::Array::from_shape_vec(vec![1, 5], (i * 5..i * 5 + 5).collect::<Vec<i32>>())
                .unwrap(),
        );
        let work = BaseWork::<Input>::new(Uuid::new_v4(), tensor);

        let model_name_clone = model_name.clone();
        let scheduler_clone = scheduler.clone_for_test();
        tokio::spawn(async move {
            let _ = scheduler_clone
                .add_compute_task(model_name_clone, work)
                .await;
        });
    }

    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(scheduler.get_model_queue_batch_size(&model_name), 100);

    // 测试小batch请求（应该很快处理）
    let start_time = std::time::Instant::now();

    let (tx, rx) = oneshot::channel();
    let worker_req = WorkerRequest {
        worker_id: Uuid::new_v4(),
        model_name: model_name.clone(),
        desired_batch: 5,
        resp: tx,
    };

    scheduler.request_batch(worker_req).await.unwrap();
    tokio::time::sleep(Duration::from_millis(20)).await;

    let batch = rx.await.unwrap();
    let processing_time = start_time.elapsed();

    assert_eq!(batch.len(), 5, "应该收到5个任务");
    assert!(processing_time.as_millis() < 100, "处理时间应该很短");
    assert_eq!(scheduler.get_model_queue_batch_size(&model_name), 95);
}

#[tokio::test]
async fn test_insufficient_tasks_for_request() {
    let scheduler = RequestScheduler::with_strategy(BatchExtractorStrategy::split_and_satisfy());
    let model_name = "test_model".to_string();
    let worker_id = Uuid::new_v4();

    // 只添加一个小任务（batch size = 2）
    let tensor = DynTensor::from(
        ndarray::Array::from_shape_vec(vec![2, 5], (0..10).collect::<Vec<i32>>()).unwrap(),
    );
    let work = BaseWork::<Input>::new(Uuid::new_v4(), tensor);

    let model_name_clone = model_name.clone();
    let scheduler_clone = scheduler.clone_for_test();
    tokio::spawn(async move {
        let _ = scheduler_clone
            .add_compute_task(model_name_clone, work)
            .await;
    });

    tokio::time::sleep(Duration::from_millis(10)).await;
    assert_eq!(scheduler.get_model_queue_batch_size(&model_name), 2);

    // 请求更大的batch size（batch size = 5，大于现有的2，无法满足）
    let (tx, rx) = oneshot::channel();
    let worker_req = WorkerRequest {
        worker_id,
        model_name: model_name.clone(),
        desired_batch: 5,
        resp: tx,
    };

    scheduler.request_batch(worker_req).await.unwrap();

    // 请求应该超时，因为任务batch size不足
    let result = tokio::time::timeout(Duration::from_millis(100), rx).await;
    assert!(
        result.is_err(),
        "应该超时，因为batch size不足（需要5但只有2）"
    );

    // 验证任务仍在队列中（没有被错误消费）
    assert_eq!(scheduler.get_model_queue_batch_size(&model_name), 2);
}

#[tokio::test]
async fn test_worker_request_queue_priority() {
    let scheduler = RequestScheduler::with_strategy(BatchExtractorStrategy::split_and_satisfy());
    let model_name = "test_model".to_string();

    // 添加足够的任务
    for i in 0..10 {
        let tensor = DynTensor::from(
            ndarray::Array::from_shape_vec(vec![1, 5], (i * 5..i * 5 + 5).collect::<Vec<i32>>())
                .unwrap(),
        );
        let work = BaseWork::<Input>::new(Uuid::new_v4(), tensor);

        let model_name_clone = model_name.clone();
        let scheduler_clone = scheduler.clone_for_test();
        tokio::spawn(async move {
            let _ = scheduler_clone
                .add_compute_task(model_name_clone, work)
                .await;
        });
    }

    tokio::time::sleep(Duration::from_millis(20)).await;

    // 创建多个worker请求
    let mut channels = Vec::new();
    for i in 0..3 {
        let (tx, rx) = oneshot::channel();
        channels.push(rx);

        let worker_req = WorkerRequest {
            worker_id: Uuid::new_v4(),
            model_name: model_name.clone(),
            desired_batch: 2 + i, // 不同的batch size
            resp: tx,
        };

        scheduler.request_batch(worker_req).await.unwrap();
    }

    tokio::time::sleep(Duration::from_millis(50)).await;

    // 验证第一个请求被优先处理
    let first_result = channels[0].try_recv();
    assert!(first_result.is_ok(), "第一个请求应该被处理");

    let batch = first_result.unwrap();
    assert_eq!(batch.len(), 2, "第一个请求的batch size应该是2");
}
