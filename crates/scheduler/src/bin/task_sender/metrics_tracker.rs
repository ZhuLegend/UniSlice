//! 指标追踪器模块
//!
//! 负责追踪样本处理速率和基本计数

use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
    time::{Duration, Instant},
};

use tokio::{
    io::{AsyncWriteExt, BufWriter},
    sync::{Mutex, RwLock},
};
use tracing::{debug, warn};

use crate::types::{LocalTaskStatus, TaskLifecycle};

/// 任务延迟记录，包含延迟和对应的样本数量用于加权计算
#[derive(Debug, Clone)]
struct LatencyRecord {
    latency: Duration,
    batch_size: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tracker() -> MetricsTracker {
        MetricsTracker::new(crate::config::MetricsSettings {
            rate_window_size: 5,
            latency_window_size: 100,
        })
    }
    #[tokio::test]
    async fn terminal_log_keeps_planned_arrivals_failures_and_exactly_one_record() {
        let folder = tempfile::tempdir().unwrap();
        let path = folder.path().join("terminal.jsonl");
        let mut tracker = tracker();
        tracker.enable_terminal_log(&path).await.unwrap();
        tracker
            .record_task_sent_at("ok", "m", 7, Instant::now() - Duration::from_secs(1))
            .await;
        tracker.mark_task_completed("ok").await;
        tracker.mark_task_completed("ok").await;
        tracker.record_task_sent("failed", "m", 3).await;
        tracker
            .mark_task_failed_reason("failed", "application_deadline")
            .await;
        tracker.mark_task_failed_reason("failed", "timeout").await;
        tracker.flush_terminal_log().await.unwrap();
        let rows = std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["request_id"], "ok");
        assert_eq!(rows[0]["batch_size"], 7);
        assert_eq!(rows[0]["success"], true);
        assert!(rows[0]["end_to_end_ms"].as_f64().unwrap() >= 1000.0);
        assert!(
            rows[0]["completed_at_ms"].as_f64().unwrap()
                > rows[0]["planned_due_ms"].as_f64().unwrap()
        );
        assert_eq!(rows[1]["success"], false);
        assert_eq!(rows[1]["failure_kind"], "application_deadline");
    }
    #[tokio::test]
    async fn all_offered_requests_and_sender_wait_count_toward_slo() {
        let tracker = tracker();
        tracker
            .record_task_sent_at("slow", "m", 1, Instant::now() - Duration::from_secs(1))
            .await;
        tracker.mark_task_completed("slow").await;
        tracker.mark_task_completed("slow").await;
        tracker.record_task_sent("overflow", "m", 3).await;
        tracker
            .mark_task_failed_reason("overflow", "sender_overflow")
            .await;
        tracker.mark_task_failed("overflow").await;
        assert_eq!(tracker.get_sample_counts().await, (2, 1, 0, 1, 4, 1, 0, 3));
        let snapshot = tracker.transport_snapshot().await;
        assert_eq!(snapshot["models"]["m"]["offered"], 2);
        assert_eq!(snapshot["models"]["m"]["failed"], 1);
        assert_eq!(snapshot["models"]["m"]["slo_400_ms_hits"], 0);
        assert_eq!(snapshot["failed_by_reason"]["sender_overflow"], 1);
    }
    #[tokio::test]
    async fn expiration_keeps_pending_and_all_aggregate_counts() {
        let mut tracker = tracker();
        tracker.task_expire_duration = Duration::ZERO;
        tracker.record_task_sent("pending", "m", 1).await;
        tracker.record_task_sent("failed", "m", 1).await;
        tracker.mark_task_failed("failed").await;
        tracker.cleanup_expired_tasks().await;
        assert_eq!(tracker.task_lifecycles.read().await.len(), 1);
        assert!(tracker.retired_tasks.read().await.is_empty());
        assert_eq!(tracker.get_sample_counts().await, (2, 0, 1, 1, 2, 0, 1, 1));
        tracker.mark_task_completed("pending").await;
        assert_eq!(tracker.get_sample_counts().await.2, 0);
    }
}

/// 指标追踪器
#[derive(Debug)]
pub struct MetricsTracker {
    terminal_log: Mutex<Option<(BufWriter<tokio::fs::File>, Option<String>)>>,
    clock_origin: Instant,
    clock_origin_unix_ms: f64,
    transport: Arc<RwLock<TransportCounts>>,
    /// 任务生命周期记录
    task_lifecycles: Arc<RwLock<HashMap<String, TaskLifecycle>>>,
    /// Terminal IDs in completion order, for incremental expiration.
    retired_tasks: Arc<RwLock<VecDeque<(Instant, String)>>>,
    /// 样本计数
    sample_counts: Arc<RwLock<SampleCounts>>,
    /// Recent terminal details; aggregate counters are never expired.
    task_expire_duration: Duration,
    /// 速率计算窗口大小
    rate_window_size: Duration,
    /// 每个模型的延迟历史记录（用于计算移动平均）
    model_latencies: Arc<RwLock<HashMap<String, VecDeque<Duration>>>>,
    /// 延迟移动平均窗口大小
    latency_window_size: usize,
    /// 每个模型的延迟记录（用于加权计算移动平均）
    model_latency_records: Arc<RwLock<HashMap<String, VecDeque<LatencyRecord>>>>,
}

#[derive(Debug, Default, serde::Serialize)]
pub struct TransportCounts {
    pub http_started_tasks: u64,
    pub max_start_lag_ms: f64,
    pub failed_by_reason: HashMap<String, u64>,
    pub models: HashMap<String, ModelCompletionCounts>,
}
#[derive(Debug, Default, serde::Serialize)]
pub struct ModelCompletionCounts {
    pub offered: u64,
    pub completed: u64,
    pub failed: u64,
    pub slo_250_ms_hits: u64,
    pub slo_400_ms_hits: u64,
}

/// 样本计数器 - 基于样本数量而非任务数量
#[derive(Debug, Default, Clone)]
struct SampleCounts {
    /// 已发送的任务数量
    sent_tasks: u64,
    /// 已完成的任务数量
    completed_tasks: u64,
    /// 已失败的任务数量
    failed_tasks: u64,
    /// 已发送的样本总数
    sent_samples: u64,
    /// 已完成的样本总数
    completed_samples: u64,
    /// 已失败的样本总数
    failed_samples: u64,
}

/// 指标追踪器
impl MetricsTracker {
    /// 创建新的指标追踪器
    pub fn new(config: crate::config::MetricsSettings) -> Self {
        Self {
            terminal_log: Default::default(),
            clock_origin: Instant::now(),
            clock_origin_unix_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64()
                * 1000.0,
            transport: Default::default(),
            task_lifecycles: Arc::new(RwLock::new(HashMap::new())),
            retired_tasks: Default::default(),
            sample_counts: Arc::new(RwLock::new(SampleCounts::default())),
            task_expire_duration: Duration::from_secs(config.rate_window_size.max(60)),
            rate_window_size: Duration::from_secs(config.rate_window_size),
            model_latencies: Arc::new(RwLock::new(HashMap::new())),
            model_latency_records: Arc::new(RwLock::new(HashMap::new())),
            latency_window_size: config.latency_window_size,
        }
    }

    pub async fn enable_terminal_log(&mut self, path: &std::path::Path) -> std::io::Result<()> {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            tokio::fs::create_dir_all(parent).await?;
        }
        let file = tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .await?;
        *self.terminal_log.get_mut() = Some((BufWriter::with_capacity(64 * 1024, file), None));
        Ok(())
    }

    pub(crate) fn unix_ms(&self, time: Instant) -> f64 {
        self.clock_origin_unix_ms
            + if time >= self.clock_origin {
                time.duration_since(self.clock_origin).as_secs_f64() * 1000.0
            } else {
                -self.clock_origin.duration_since(time).as_secs_f64() * 1000.0
            }
    }

    async fn log_terminal(&self, lifecycle: &TaskLifecycle, failure_kind: Option<&str>) {
        let mut log = self.terminal_log.lock().await;
        let Some((writer, error)) = log.as_mut() else {
            return;
        };
        let completed = lifecycle.completed_at.unwrap();
        let row = serde_json::json!({"request_id": lifecycle._task_id,
            "model": lifecycle.model, "batch_size": lifecycle.batch_size,
            "planned_due_ms": self.unix_ms(lifecycle.sent_at),
            "completed_at_ms": self.unix_ms(completed),
            "end_to_end_ms": completed.duration_since(lifecycle.sent_at).as_secs_f64() * 1000.0,
            "success": failure_kind.is_none(), "failure_kind": failure_kind});
        if let Err(failure) = writer.write_all(format!("{row}\n").as_bytes()).await {
            tracing::error!(%failure, "could not persist client terminal record");
            *error = Some(failure.to_string());
        }
    }

    pub async fn flush_terminal_log(&self) -> std::io::Result<()> {
        let mut log = self.terminal_log.lock().await;
        if let Some((writer, error)) = log.as_mut() {
            writer.flush().await?;
            if let Some(error) = error {
                return Err(std::io::Error::other(error.clone()));
            }
        }
        Ok(())
    }

    /// 记录任务发送
    pub async fn record_task_sent(&self, task_id: &str, model: &str, batch_size: u32) {
        self.record_task_sent_at(task_id, model, batch_size, Instant::now())
            .await;
    }

    pub async fn record_task_sent_at(
        &self,
        task_id: &str,
        model: &str,
        batch_size: u32,
        due: Instant,
    ) {
        let lifecycle = TaskLifecycle {
            _task_id: task_id.to_string(),
            model: model.to_string(),
            sent_at: due,
            completed_at: None,
            status: LocalTaskStatus::Sent,
            batch_size, // 添加batch_size到生命周期记录
        };

        {
            let mut lifecycles = self.task_lifecycles.write().await;
            lifecycles.insert(task_id.to_string(), lifecycle);
        }
        self.transport
            .write()
            .await
            .models
            .entry(model.to_owned())
            .or_default()
            .offered += 1;

        {
            let mut counts = self.sample_counts.write().await;
            counts.sent_tasks += 1;
            counts.sent_samples += batch_size as u64;
        }

        debug!(
            "记录任务发送: {} (batch_size: {} samples)",
            task_id, batch_size
        );
    }

    /// 标记任务完成
    pub async fn mark_task_completed(&self, task_id: &str) {
        let (model_name, latency, batch_size, terminal) = {
            let mut lifecycles = self.task_lifecycles.write().await;
            let completed_at = Instant::now();
            if let Some(lifecycle) = lifecycles.get_mut(task_id) {
                if lifecycle.status != LocalTaskStatus::Sent {
                    return;
                }
                lifecycle.completed_at = Some(completed_at);
                lifecycle.status = LocalTaskStatus::Completed;
                let latency = completed_at.duration_since(lifecycle.sent_at);
                self.retired_tasks
                    .write()
                    .await
                    .push_back((completed_at, task_id.to_owned()));
                (
                    lifecycle.model.clone(),
                    Some(latency),
                    lifecycle.batch_size,
                    lifecycle.clone(),
                )
            } else {
                warn!("A unexpeted task:{task_id:?} found");
                return;
            }
        };

        self.log_terminal(&terminal, None).await;

        // 记录延迟（包含batch_size用于加权计算）
        if let Some(latency) = latency {
            let mut transport = self.transport.write().await;
            let model = transport.models.entry(model_name.clone()).or_default();
            model.completed += 1;
            model.slo_250_ms_hits += u64::from(latency <= Duration::from_millis(250));
            model.slo_400_ms_hits += u64::from(latency <= Duration::from_millis(400));
            drop(transport);
            self.record_weighted_latency(&model_name, latency, batch_size)
                .await;
        }

        {
            let mut counts = self.sample_counts.write().await;
            counts.completed_tasks += 1;
            counts.completed_samples += batch_size as u64;
        }

        debug!(
            "标记任务完成: {} (batch_size: {} samples)",
            task_id, batch_size
        );
    }

    /// 标记任务失败
    pub async fn mark_task_failed(&self, task_id: &str) {
        self.mark_task_failed_reason(task_id, "request_error").await;
    }

    pub async fn mark_task_failed_reason(&self, task_id: &str, reason: &str) -> u64 {
        let (batch_size, model, terminal) = {
            let mut lifecycles = self.task_lifecycles.write().await;
            if let Some(lifecycle) = lifecycles.get_mut(task_id) {
                if lifecycle.status != LocalTaskStatus::Sent {
                    return 0;
                }
                lifecycle.status = LocalTaskStatus::Failed;
                let completed_at = Instant::now();
                lifecycle.completed_at = Some(completed_at);
                self.retired_tasks
                    .write()
                    .await
                    .push_back((completed_at, task_id.to_owned()));
                (
                    lifecycle.batch_size,
                    lifecycle.model.clone(),
                    lifecycle.clone(),
                )
            } else {
                return 0;
            }
        };
        self.log_terminal(&terminal, Some(reason)).await;
        let reason_count = {
            let mut transport = self.transport.write().await;
            transport.models.entry(model).or_default().failed += 1;
            let count = transport
                .failed_by_reason
                .entry(reason.to_owned())
                .or_default();
            *count += 1;
            *count
        };

        {
            let mut counts = self.sample_counts.write().await;
            counts.failed_tasks += 1;
            counts.failed_samples += batch_size as u64;
        }

        debug!(
            "标记任务失败: {} (batch_size: {} samples)",
            task_id, batch_size
        );
        reason_count
    }

    /// 记录加权延迟（基于batch_size）
    async fn record_weighted_latency(&self, model: &str, latency: Duration, batch_size: u32) {
        // 记录原始延迟（用于简单统计）
        {
            let mut model_latencies = self.model_latencies.write().await;
            let latencies = model_latencies
                .entry(model.to_string())
                .or_insert_with(VecDeque::new);

            latencies.push_back(latency);

            // 保持窗口大小
            while latencies.len() > self.latency_window_size {
                latencies.pop_front();
            }
        }

        // 记录加权延迟记录（用于样本级别的统计）
        {
            let mut model_latency_records = self.model_latency_records.write().await;
            let records = model_latency_records
                .entry(model.to_string())
                .or_insert_with(VecDeque::new);

            records.push_back(LatencyRecord {
                latency,
                batch_size,
            });

            // 保持窗口大小
            while records.len() > self.latency_window_size {
                records.pop_front();
            }
        }
    }

    /// 获取样本计数 (sent_tasks, completed_tasks, pending_tasks, failed_tasks, sent_samples, completed_samples, pending_samples, failed_samples)
    pub async fn get_sample_counts(&self) -> (u64, u64, u64, u64, u64, u64, u64, u64) {
        let counts = self.sample_counts.read().await;
        let pending_tasks = counts
            .sent_tasks
            .saturating_sub(counts.completed_tasks + counts.failed_tasks);
        let pending_samples = counts
            .sent_samples
            .saturating_sub(counts.completed_samples + counts.failed_samples);

        (
            counts.sent_tasks,
            counts.completed_tasks,
            pending_tasks,
            counts.failed_tasks,
            counts.sent_samples,
            counts.completed_samples,
            pending_samples,
            counts.failed_samples,
        )
    }

    pub async fn transport_snapshot(&self) -> serde_json::Value {
        serde_json::to_value(&*self.transport.read().await).unwrap()
    }
    pub async fn record_http_start(&self, lag: Duration) {
        let mut counts = self.transport.write().await;
        counts.http_started_tasks += 1;
        counts.max_start_lag_ms = counts.max_start_lag_ms.max(lag.as_secs_f64() * 1000.0);
    }

    /// 获取每个模型的样本处理速率
    pub async fn get_model_sample_rates(&self) -> HashMap<String, f64> {
        let now = Instant::now();
        let window_start = now - self.rate_window_size;
        let lifecycles = self.task_lifecycles.read().await;

        let mut model_sample_counts: HashMap<String, u64> = HashMap::new();

        // 统计窗口期内每个模型处理的样本数
        for lifecycle in lifecycles.values() {
            if lifecycle.sent_at >= window_start {
                *model_sample_counts
                    .entry(lifecycle.model.clone())
                    .or_insert(0) += lifecycle.batch_size as u64;
            }
        }

        // 计算每个模型的样本处理速率 (samples/sec)
        let window_duration_secs = self.rate_window_size.as_secs_f64();
        model_sample_counts
            .into_iter()
            .map(|(model, count)| (model, count as f64 / window_duration_secs))
            .collect()
    }

    /// 获取每个模型的任务发送速率（用于调试）
    pub async fn get_model_task_rates(&self) -> HashMap<String, f64> {
        let now = Instant::now();
        let window_start = now - self.rate_window_size;
        let lifecycles = self.task_lifecycles.read().await;

        let mut model_task_counts: HashMap<String, u32> = HashMap::new();

        // 统计窗口期内每个模型发送的任务数
        for lifecycle in lifecycles.values() {
            if lifecycle.sent_at >= window_start {
                *model_task_counts
                    .entry(lifecycle.model.clone())
                    .or_insert(0) += 1;
            }
        }

        // 计算每个模型的任务发送速率 (tasks/sec)
        let window_duration_secs = self.rate_window_size.as_secs_f64();
        model_task_counts
            .into_iter()
            .map(|(model, count)| (model, count as f64 / window_duration_secs))
            .collect()
    }

    /// 获取每个模型的移动平均延迟（毫秒）- 基于样本数量加权
    pub async fn get_model_average_latencies(&self) -> HashMap<String, f64> {
        let model_latency_records = self.model_latency_records.read().await;
        let mut result = HashMap::new();

        for (model, records) in model_latency_records.iter() {
            if !records.is_empty() {
                let (total_weighted_ms, total_samples) =
                    records
                        .iter()
                        .fold((0.0, 0u64), |(acc_ms, acc_samples), record| {
                            let latency_ms = record.latency.as_millis() as f64;
                            let weighted_ms = latency_ms * record.batch_size as f64;
                            (acc_ms + weighted_ms, acc_samples + record.batch_size as u64)
                        });

                if total_samples > 0 {
                    let weighted_avg_ms = total_weighted_ms / total_samples as f64;
                    result.insert(model.clone(), weighted_avg_ms);
                }
            }
        }

        // 如果没有加权记录，回退到简单平均
        if result.is_empty() {
            let model_latencies = self.model_latencies.read().await;
            for (model, latencies) in model_latencies.iter() {
                if !latencies.is_empty() {
                    let total_ms: f64 = latencies.iter().map(|d| d.as_millis() as f64).sum();
                    let avg_ms = total_ms / latencies.len() as f64;
                    result.insert(model.clone(), avg_ms);
                }
            }
        }

        result
    }

    /// 清理过期任务
    pub async fn cleanup_expired_tasks(&self) {
        let now = Instant::now();
        let expire_threshold = now - self.task_expire_duration;

        let mut lifecycles = self.task_lifecycles.write().await;
        let mut retired = self.retired_tasks.write().await;
        let mut removed_count = 0;
        while retired
            .front()
            .is_some_and(|(time, _)| *time <= expire_threshold)
        {
            let (_, id) = retired.pop_front().unwrap();
            removed_count += usize::from(lifecycles.remove(&id).is_some());
        }
        // Pending requests remain owned until success, explicit failure or cancellation.
        if removed_count > 0 {
            debug!("清理了 {} 个过期任务", removed_count);
        }
    }
}
