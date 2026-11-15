//! HTTP处理器模块

use std::sync::{Arc, atomic::Ordering};

use axum::{
    extract::{Path, Request, State},
    http::{StatusCode, header},
    response::{Json, Response},
};
use compute_node::{
    compute_node::r#virtual::virtual_compute_node::WorkTask,
    util::{Input, Output},
};
use futures::StreamExt;
use serde_json::Value;

use crate::{
    daemon::{
        types::{AppState, RequestPermit, StatsResponse, TaskStats, TaskStatus},
        utils::{current_timestamp, update_task_status},
    },
    schedule_system::virtual_compute_node::system::heterogeneous::system::HeterogeneousSystem,
};

// Finalize exactly once, including when the caller drops its HTTP stream.
struct TaskCompletion {
    state: AppState,
    id: String,
    permit: Option<RequestPermit>,
}
impl TaskCompletion {
    fn finish(&mut self, status: TaskStatus, error: Option<String>) {
        let Some(permit) = self.permit.take() else {
            return;
        };
        let task = {
            let Some(mut task) = self.state.task_stats.get_mut(&self.id) else {
                return;
            };
            task.status = status;
            task.error = error;
            task.completed_at = Some(current_timestamp());
            task.clone()
        };
        if task.status == TaskStatus::Completed {
            self.state
                .task_totals
                .completed
                .fetch_add(1, Ordering::Relaxed);
            self.state.task_totals.completed_duration_ms.fetch_add(
                task.completed_at.unwrap().saturating_sub(task.received_at),
                Ordering::Relaxed,
            );
        } else {
            self.state
                .task_totals
                .failed
                .fetch_add(1, Ordering::Relaxed);
        }
        self.state.csv_logger.log_terminal(&task, Some(permit));
        let mut history = self.state.terminal_history.lock().unwrap();
        history.push_back(self.id.clone());
        while history.len() > 100_000 {
            if let Some(id) = history.pop_front() {
                self.state.task_stats.remove_if(&id, |_, task| {
                    matches!(task.status, TaskStatus::Completed | TaskStatus::Failed)
                });
            }
        }
    }
}
impl Drop for TaskCompletion {
    fn drop(&mut self) {
        self.finish(
            TaskStatus::Failed,
            Some("caller cancelled before completion".into()),
        );
    }
}

/// 健康检查端点
pub async fn health() -> Json<Value> {
    Json(serde_json::json!({
        "status": "ok",
        "service": "scheduler",
        "timestamp": current_timestamp()
    }))
}

/// 提交任务端点（Bincode版本）
pub async fn submit_task(
    State(app_state): State<AppState>,
    request: Request,
) -> Result<Response<axum::body::Body>, StatusCode> {
    if app_state
        .simulation_startup
        .as_ref()
        .is_some_and(|gate| !gate.ready.load(std::sync::atomic::Ordering::Acquire))
    {
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }
    let slot = app_state
        .request_slots
        .clone()
        .try_acquire_owned()
        .map_err(|_| StatusCode::TOO_MANY_REQUESTS)?;
    let mut byte_permit = app_state
        .request_bytes
        .clone()
        .try_acquire_many_owned(0)
        .map_err(|_| StatusCode::TOO_MANY_REQUESTS)?;
    let mut body = Vec::new();
    let mut stream = request.into_body().into_data_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|_| StatusCode::BAD_REQUEST)?;
        if body.len().saturating_add(chunk.len()) > 100 * 1024 * 1024 {
            return Err(StatusCode::PAYLOAD_TOO_LARGE);
        }
        let permit = app_state
            .request_bytes
            .clone()
            .try_acquire_many_owned(chunk.len() as u32)
            .map_err(|_| StatusCode::TOO_MANY_REQUESTS)?;
        byte_permit.merge(permit);
        body.extend_from_slice(&chunk);
    }
    let permit = Arc::new((slot, byte_permit));
    // 反序列化Bincode数据
    let config = bincode::config::standard();
    let task: WorkTask<Input> = match bincode::decode_from_slice(&body, config) {
        Ok((task, _)) => task,
        Err(e) => {
            tracing::error!("Bincode反序列化失败: {}", e);
            return Err(StatusCode::BAD_REQUEST);
        }
    };

    if !app_state.supported_models.contains(&task.model) {
        return Err(StatusCode::BAD_REQUEST);
    }
    let batch_size = task
        .work
        .data()
        .dimensions()
        .first()
        .copied()
        .and_then(|batch| usize::try_from(batch).ok())
        .unwrap_or(0);
    if batch_size == 0 || u32::try_from(batch_size).is_err() {
        return Err(StatusCode::BAD_REQUEST);
    }
    let task_id = task.work.id().to_string();
    let model = task.model.clone();

    tracing::trace!("收到Bincode任务: {} (模型: {})", task_id, model);

    // 记录任务接收时间
    match app_state.task_stats.entry(task_id.clone()) {
        dashmap::mapref::entry::Entry::Occupied(_) => return Err(StatusCode::CONFLICT),
        dashmap::mapref::entry::Entry::Vacant(entry) => {
            entry.insert(TaskStats {
                task_id: task_id.clone(),
                model: model.clone(),
                batch_size: batch_size as u64,
                received_at: current_timestamp(),
                completed_at: None,
                status: TaskStatus::Received,
                error: None,
            });
        }
    }
    app_state
        .task_totals
        .received
        .fetch_add(1, Ordering::Relaxed);
    let mut completion = TaskCompletion {
        state: app_state.clone(),
        id: task_id.clone(),
        permit: Some(permit.clone()),
    };

    // 记录任务到达到模型预测器（用于调度）
    app_state.rate_predictor.record_arrival(&model, batch_size);
    if let Some(gate) = &app_state.simulation_startup {
        gate.started.send_if_modified(|started| {
            if *started {
                false
            } else {
                *started = true;
                true
            }
        });
    }

    // 更新任务状态为处理中
    update_task_status(
        &app_state.task_stats,
        &task_id,
        TaskStatus::Processing,
        None,
    );

    // 异步处理任务
    let system = app_state.schedule_system.clone();
    let task_id_clone = task_id.clone();
    // Ownership stays with accepted work if the HTTP caller disconnects.
    // Otherwise cancelled tensors in a temporarily unavailable pool could
    // outlive the admission permit and bypass the global memory budget.
    let result = tokio::spawn(async move {
        let result = process_task_heterogeneous(system, task).await;
        drop(permit);
        result
    })
    .await
    .unwrap_or_else(|error| Err(error.into()));

    match result {
        Ok(res) => {
            tracing::trace!("Bincode任务处理完成: {}", task_id_clone);
            // 创建成功响应，直接返回Bincode序列化的WorkTask<Output>
            let config = bincode::config::standard();
            let serialized = match bincode::encode_to_vec(&res, config) {
                Ok(data) => data,
                Err(e) => {
                    tracing::error!("响应序列化失败: {}", e);
                    completion.finish(TaskStatus::Failed, Some(e.to_string()));
                    return Err(StatusCode::INTERNAL_SERVER_ERROR);
                }
            };

            completion.finish(TaskStatus::Completed, None);
            Ok(Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "application/octet-stream")
                .header("X-Task-ID", task_id)
                .header("X-Task-Status", "success")
                .body(axum::body::Body::from(serialized))
                .unwrap())
        }
        Err(e) => {
            tracing::error!("Bincode任务处理失败: {} - {}", task_id_clone, e);
            completion.finish(TaskStatus::Failed, Some(e.to_string()));

            // 返回错误响应
            Ok(Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "application/octet-stream")
                .header("X-Task-ID", task_id)
                .header("X-Task-Status", "failed")
                .header("X-Error-Message", format!("{:?}", e))
                .body(axum::body::Body::empty())
                .unwrap())
        }
    }
}

/// 获取任务详情端点
pub async fn get_task(
    State(app_state): State<AppState>,
    Path(task_id): Path<String>,
) -> Result<Json<TaskStats>, StatusCode> {
    match app_state.task_stats.get(&task_id) {
        Some(task_stats) => Ok(Json(task_stats.clone())),
        None => Err(StatusCode::NOT_FOUND),
    }
}

/// 获取统计信息端点
pub async fn get_stats(State(app_state): State<AppState>) -> Json<StatsResponse> {
    let total_tasks = app_state.task_totals.received.load(Ordering::Relaxed);
    let completed_tasks = app_state.task_totals.completed.load(Ordering::Relaxed);
    let failed_tasks = app_state.task_totals.failed.load(Ordering::Relaxed);
    let processing_tasks = app_state
        .task_stats
        .iter()
        .filter(|entry| matches!(entry.value().status, TaskStatus::Processing))
        .count();
    let avg_processing_time_ms = (completed_tasks > 0).then(|| {
        app_state
            .task_totals
            .completed_duration_ms
            .load(Ordering::Relaxed) as f64
            / completed_tasks as f64
    });

    Json(StatsResponse {
        total_tasks,
        completed_tasks,
        processing_tasks,
        failed_tasks,
        avg_processing_time_ms,
    })
}

/// 获取详细任务列表端点
pub async fn get_task_list(State(app_state): State<AppState>) -> Json<Vec<TaskStats>> {
    let mut tasks: Vec<TaskStats> = app_state
        .task_stats
        .iter()
        .map(|entry| entry.value().clone())
        .collect();

    // 按接收时间排序，最新的在前
    tasks.sort_by(|a, b| b.received_at.cmp(&a.received_at));

    Json(tasks)
}

/// 获取特定状态的任务列表端点
pub async fn get_tasks_by_status(
    State(app_state): State<AppState>,
    Path(status): Path<String>,
) -> Result<Json<Vec<TaskStats>>, StatusCode> {
    let target_status = match status.to_lowercase().as_str() {
        "received" => TaskStatus::Received,
        "processing" => TaskStatus::Processing,
        "completed" => TaskStatus::Completed,
        "failed" => TaskStatus::Failed,
        _ => return Err(StatusCode::BAD_REQUEST),
    };

    let tasks: Vec<TaskStats> = app_state
        .task_stats
        .iter()
        .filter(|entry| entry.value().status == target_status)
        .map(|entry| entry.value().clone())
        .collect();

    Ok(Json(tasks))
}

/// 清除已完成的任务端点
pub async fn clear_completed_tasks(State(app_state): State<AppState>) -> Json<Value> {
    let initial_count = app_state.task_stats.len();

    app_state
        .task_stats
        .retain(|_, task| !matches!(task.status, TaskStatus::Completed | TaskStatus::Failed));

    let removed_count = initial_count.saturating_sub(app_state.task_stats.len());

    tracing::info!("Cleared {} completed tasks", removed_count);

    Json(serde_json::json!({
        "status": "ok",
        "message": format!("Cleared {} completed tasks", removed_count),
        "removed_count": removed_count
    }))
}

/// 处理任务
async fn process_task_heterogeneous(
    system: Arc<HeterogeneousSystem>,
    task: WorkTask<Input>,
) -> Result<WorkTask<Output>, Box<dyn std::error::Error + Send + Sync>> {
    // 使用异构系统的compute方法处理任务
    match system.compute(task).await {
        Ok(result) => Ok(result),
        Err(err) => Err(err.into()),
    }
}

/// 获取模型预测统计信息端点
pub async fn get_prediction_stats(State(app_state): State<AppState>) -> Json<Value> {
    let predictions = app_state.rate_predictor.predict().await;
    let tracked_models = app_state.rate_predictor.get_tracked_models();

    let mut model_stats = serde_json::Map::new();
    for model in &tracked_models {
        let rate = predictions.get(model).copied().unwrap_or(0.0);
        let recent_count = app_state
            .rate_predictor
            .get_model_traffic_volume(model, 300); // 5分钟内的流量

        model_stats.insert(
            model.clone(),
            serde_json::json!({
                "predicted_rate_per_second": rate,
                "recent_arrivals_5min": recent_count,
                "last_update": current_timestamp()
            }),
        );
    }

    Json(serde_json::json!({
        "timestamp": current_timestamp(),
        "total_tracked_models": tracked_models.len(),
        "models": model_stats
    }))
}

/// Observe cached weights and eligible pools without refreshing routing state.
pub async fn get_routing_weights(State(state): State<AppState>) -> Json<Value> {
    let mut models = state.supported_models.iter().collect::<Vec<_>>();
    models.sort();
    let models = models
        .into_iter()
        .map(|model| {
            use crate::schedule_system::allocator::RankableShareTarget;
            let profiles = state.schedule_system.get_applied_bids(model)
                .into_iter()
                .map(|(pool, bids)| {
                    let profiles = bids.into_iter().map(|bid| {
                        #[cfg(feature = "nvidia")]
                        let profile = bid.data.downcast_ref::<compute_node::compute_device::vgpu::resource_spec::nv::mig::VDeviceScale>()
                            .map(|resource| resource.name());
                        #[cfg(not(feature = "nvidia"))]
                        let profile: Option<&str> = None;
                        serde_json::json!({"profile": profile, "target_id": bid.target_id(),
                            "predicted_seconds": ([1, 4, 8, 16].map(|batch| bid.evaluate_latency(model, batch)))})
                    }).collect::<Vec<_>>();
                    (pool, profiles)
                }).collect::<std::collections::HashMap<_, _>>();
            serde_json::json!({
                "model": model,
                "applied_profiles": profiles,
                "normalized_weights": state.schedule_system.get_normalized_weights(model),
                "eligible_pools": state.schedule_system.get_supporting_devices(model),
            })
        })
        .collect::<Vec<_>>();
    Json(serde_json::json!({"timestamp_ms": current_timestamp(), "models": models}))
}

/// 获取调度器统计信息端点
pub async fn scheduler_stats(State(app_state): State<AppState>) -> Json<Value> {
    let stats = app_state.scheduler_stats.get_stats();

    Json(serde_json::json!({
        "cycle_count": stats.cycle_count,
        "successful_schedules": stats.successful_schedules,
        "failed_schedules": stats.failed_schedules,
        "avg_schedule_time_ms": stats.avg_schedule_time_ms,
        "last_schedule_timestamp": stats.last_schedule_timestamp,
        "success_rate": if stats.cycle_count > 0 {
            stats.successful_schedules as f64 / stats.cycle_count as f64 * 100.0
        } else {
            0.0
        },
        "timestamp": current_timestamp()
    }))
}

/// 重置调度器统计信息端点
pub async fn reset_scheduler_stats(State(app_state): State<AppState>) -> Json<Value> {
    app_state.scheduler_stats.reset_stats();

    Json(serde_json::json!({
        "status": "ok",
        "message": "调度器统计信息已重置",
        "timestamp": current_timestamp()
    }))
}

#[cfg(test)]
mod boundary_tests {
    use axum::body::Body;
    use compute_node::compute_node::r#virtual::BaseWork;
    use tokio::sync::Semaphore;
    use uuid::Uuid;
    use worker::model::tensor::{DynTensor, TensorElementType};

    use super::*;

    #[tokio::test]
    async fn invalid_models_budgets_and_terminal_eviction_preserve_counters() {
        let directory = tempfile::tempdir().unwrap();
        let state = AppState {
            supported_models: Arc::new(["known".to_owned()].into_iter().collect()),
            request_slots: Arc::new(Semaphore::new(1)),
            request_bytes: Arc::new(Semaphore::new(1024)),
            task_totals: Arc::default(),
            terminal_history: Arc::default(),
            simulation_startup: None,
            task_stats: Arc::default(),
            rate_predictor: Default::default(),
            schedule_system: Arc::new(HeterogeneousSystem::new(
                Default::default(),
                Arc::new(|| Box::pin(async { Default::default() })),
                1..=32,
            )),
            csv_logger: crate::daemon::csv_logger::CsvTaskLogger::new(directory.path())
                .await
                .unwrap(),
            scheduler_stats: crate::daemon::stats::SchedulerStatsManager::new(directory.path())
                .await
                .unwrap(),
        };
        let body_for = |model: &str| {
            let task = WorkTask {
                model: model.into(),
                work: BaseWork::new(
                    Uuid::new_v4(),
                    DynTensor::zeros(vec![1, 0], TensorElementType::Float),
                ),
            };
            Request::builder()
                .body(Body::from(
                    bincode::encode_to_vec(task, bincode::config::standard()).unwrap(),
                ))
                .unwrap()
        };
        assert_eq!(
            submit_task(State(state.clone()), body_for("unknown"))
                .await
                .unwrap_err(),
            StatusCode::BAD_REQUEST
        );
        assert!(state.rate_predictor.get_tracked_models().is_empty());
        assert!(state.task_stats.is_empty());
        let slot = state.request_slots.clone().try_acquire_owned().unwrap();
        assert_eq!(
            submit_task(State(state.clone()), body_for("known"))
                .await
                .unwrap_err(),
            StatusCode::TOO_MANY_REQUESTS
        );
        drop(slot);
        assert_eq!(
            submit_task(
                State(state.clone()),
                Request::builder().body(Body::from(vec![0; 1025])).unwrap()
            )
            .await
            .unwrap_err(),
            StatusCode::TOO_MANY_REQUESTS
        );
        state
            .terminal_history
            .lock()
            .unwrap()
            .extend((0..100_000).map(|i| format!("old-{i}")));
        state.task_stats.insert(
            "old-0".into(),
            TaskStats {
                task_id: "old-0".into(),
                model: "known".into(),
                batch_size: 1,
                received_at: 0,
                completed_at: Some(1),
                status: TaskStatus::Failed,
                error: None,
            },
        );
        // A configured model reaches the system and returns a terminal failure
        // in this no-device setup; rejection has not poisoned admission.
        let response = submit_task(State(state.clone()), body_for("known"))
            .await
            .unwrap();
        assert_eq!(response.headers()["X-Task-Status"], "failed");
        state.csv_logger.flush().await.unwrap();
        assert_eq!(state.terminal_history.lock().unwrap().len(), 100_000);
        assert!(!state.task_stats.contains_key("old-0"));
        assert_eq!(state.request_slots.available_permits(), 1);
        assert_eq!(state.request_bytes.available_permits(), 1024);
        clear_completed_tasks(State(state.clone())).await;
        assert!(state.task_stats.is_empty());
        let stats = get_stats(State(state.clone())).await.0;
        assert_eq!(
            (stats.total_tasks, stats.completed_tasks, stats.failed_tasks),
            (1, 0, 1)
        );
        let log = std::fs::read_to_string(directory.path().join("task_terminal_log.csv")).unwrap();
        assert!(log.contains("task_id,status,error"));
        assert!(log.contains(",failed,"));
        let id = "cancelled-task".to_owned();
        state.task_stats.insert(
            id.clone(),
            TaskStats {
                task_id: id.clone(),
                model: "known".into(),
                batch_size: 1,
                received_at: current_timestamp(),
                completed_at: None,
                status: TaskStatus::Processing,
                error: None,
            },
        );
        state.task_totals.received.fetch_add(1, Ordering::Relaxed);
        let active_work = Arc::new((
            state.request_slots.clone().try_acquire_owned().unwrap(),
            state
                .request_bytes
                .clone()
                .try_acquire_many_owned(1)
                .unwrap(),
        ));
        drop(TaskCompletion {
            state: state.clone(),
            id,
            permit: Some(active_work.clone()),
        });
        state.csv_logger.flush().await.unwrap();
        assert_eq!(state.request_slots.available_permits(), 0);
        drop(active_work);
        assert_eq!(state.request_slots.available_permits(), 1);
        clear_completed_tasks(State(state.clone())).await;
        let stats = get_stats(State(state)).await.0;
        assert_eq!((stats.total_tasks, stats.failed_tasks), (2, 2));
        let log = std::fs::read_to_string(directory.path().join("task_terminal_log.csv")).unwrap();
        assert!(log.contains("cancelled-task,failed,caller cancelled before completion"));
    }
}
