//! 泊松分布发送器模块
//!
//! 负责按照泊松分布发送任务到scheduler daemon
//! 基于样本处理速率和batch size计算任务发送间隔

use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use compute_node::compute_node::r#virtual::{BaseWork, virtual_compute_node::WorkTask};
use rand::Rng;
use tokio::{
    sync::{RwLock, Semaphore, watch},
    time::{sleep, timeout},
};
use tracing::{debug, info, warn};
use uuid::Uuid;
use worker::model::tensor::{DynTensor, Tensor};

use crate::{
    config::{ModelConfig, RateChangeStrategy, SenderConfig},
    http_client::HttpClient,
    metrics_tracker::MetricsTracker,
};

/// Explicit replay namespace in the high 64 bits, zero-based ordinal in the low 64.
/// The pair is preserved exactly; no random source or UUID version bits are applied.
pub(crate) fn replay_request_id(seed: u64, ordinal: u64) -> Uuid {
    Uuid::from_u128((u128::from(seed) << 64) | u128::from(ordinal))
}

/// 泊松分布发送器
#[derive(Debug, Clone)]
pub struct PoissonSender {
    in_flight: Arc<Semaphore>,
    cancelled: watch::Sender<bool>,
    failures_logged: Arc<AtomicU64>,
    /// 配置
    config: SenderConfig,
    /// HTTP客户端
    http_client: Arc<HttpClient>,
    /// 指标追踪器
    metrics_tracker: Arc<MetricsTracker>,
    /// 当前样本速率配置 (samples/sec)
    current_sample_rates: Arc<RwLock<HashMap<String, f64>>>,
    /// 目标样本速率配置 (samples/sec)
    target_sample_rates: Arc<RwLock<HashMap<String, f64>>>,
    /// 是否仅发送batch size
    send_batch_size_only: bool,
}

impl PoissonSender {
    /// 创建新的泊松分布发送器
    pub fn new(
        config: SenderConfig,
        http_client: Arc<HttpClient>,
        metrics_tracker: Arc<MetricsTracker>,
        send_batch_size_only: bool,
        max_in_flight: usize,
    ) -> Self {
        Self {
            in_flight: Arc::new(Semaphore::new(max_in_flight)),
            cancelled: watch::channel(false).0,
            failures_logged: Default::default(),
            config,
            http_client,
            metrics_tracker,
            current_sample_rates: Arc::new(RwLock::new(HashMap::new())),
            target_sample_rates: Arc::new(RwLock::new(HashMap::new())),
            send_batch_size_only,
        }
    }

    /// 更新模型的目标样本速率
    pub async fn update_model_rate(&self, model: &str, rate_range: &[f64]) -> f64 {
        if rate_range.len() < 2 || rate_range[0] > rate_range[1] {
            warn!("模型{}的样本速率范围无效: {:?}", model, rate_range);
            return 0.0;
        }

        // 根据策略计算新的目标样本速率
        let new_sample_rate = match self.config.rate_change.strategy {
            RateChangeStrategy::Sudden | RateChangeStrategy::Smooth => {
                let mut rng = rand::rng();
                rng.random_range(rate_range[0]..=rate_range[1])
            }
            RateChangeStrategy::RandomWalk => {
                let current_rate = {
                    let rates = self.target_sample_rates.read().await;
                    rates
                        .get(model)
                        .copied()
                        .unwrap_or((rate_range[0] + rate_range[1]) / 2.0)
                };

                let mut rng = rand::rng();
                let max_delta =
                    (rate_range[1] - rate_range[0]) * self.config.rate_change.walk_step_percent;
                let change = rng.random_range(-max_delta..=max_delta);
                (current_rate + change).clamp(rate_range[0], rate_range[1])
            }
        };

        // 更新目标样本速率
        {
            let mut target_rates = self.target_sample_rates.write().await;
            target_rates.insert(model.to_string(), new_sample_rate);
        }

        // 根据策略，更新用于发送的"当前样本速率"
        match self.config.rate_change.strategy {
            RateChangeStrategy::Smooth => {
                let mut current_rates = self.current_sample_rates.write().await;
                if !current_rates.contains_key(model) {
                    current_rates.insert(model.to_string(), new_sample_rate);
                }
            }
            RateChangeStrategy::Sudden | RateChangeStrategy::RandomWalk => {
                let mut current_rates = self.current_sample_rates.write().await;
                current_rates.insert(model.to_string(), new_sample_rate);
            }
        }

        debug!(
            "模型{}更新目标样本速率: {:.2} samples/sec",
            model, new_sample_rate
        );
        new_sample_rate
    }

    /// 运行一个发送周期
    pub async fn run_cycle(&self, models: Vec<ModelConfig>, duration: Duration) {
        let start_time = Instant::now();
        let end_time = start_time + duration;

        info!("开始泊松发送周期，持续时间: {:?}", duration);

        // 为每个模型启动发送任务
        let mut handles = Vec::new();

        for model_config in models {
            let sender = self.clone();
            let model_name = model_config.name.clone();

            let handle = tokio::spawn(async move {
                sender.send_for_model(model_config, end_time).await;
            });

            handles.push((model_name, handle));
        }

        // 启动速率调整任务
        let rate_adjustment_handle = {
            let sender = self.clone();
            let models_clone = {
                let current_rates = self.current_sample_rates.read().await;
                current_rates.keys().cloned().collect::<Vec<_>>()
            };

            tokio::spawn(async move {
                sender.adjust_rates_smoothly(models_clone, duration).await;
            })
        };

        // 等待所有任务完成或周期结束
        let timeout_duration = duration + Duration::from_secs(10); // 给一些额外时间
        let results = timeout(timeout_duration, async {
            // 等待所有模型发送任务完成
            for (model_name, handle) in handles {
                if let Err(e) = handle.await {
                    crate::error_display::print_error(&format!(
                        "模型{}的发送任务异常结束: {}",
                        model_name, e
                    ));
                }
            }

            // 等待速率调整任务完成
            if let Err(e) = rate_adjustment_handle.await {
                crate::error_display::print_error(&format!("速率调整任务异常结束: {}", e));
            }
        })
        .await;

        match results {
            Ok(_) => {
                info!("发送周期正常完成");
            }
            Err(_) => {
                warn!("发送周期超时，强制结束");
            }
        }

        let actual_duration = start_time.elapsed();
        info!("发送周期结束，实际持续时间: {:?}", actual_duration);
    }

    /// 为特定模型发送任务
    async fn send_for_model(&self, model_config: ModelConfig, end_time: Instant) {
        let model_name = &model_config.name;
        info!("开始为模型{}发送任务", model_name);

        while Instant::now() < end_time {
            // 获取当前样本速率（样本/秒）
            let current_sample_rate = {
                let current_rates = self.current_sample_rates.read().await;
                current_rates.get(model_name).copied().unwrap_or(0.0)
            };

            if current_sample_rate <= 0.0 {
                // 如果速率为0，等待一小段时间后继续检查
                sleep(Duration::from_millis(100)).await;
                continue;
            }

            // 生成当前任务的batch size
            let batch_size = if model_config.batch_size_range.len() >= 2 {
                let mut rng = rand::rng();
                rng.random_range(
                    model_config.batch_size_range[0]..=model_config.batch_size_range[1],
                ) as usize
            } else {
                model_config.batch_size_range.first().copied().unwrap_or(1) as usize
            };

            // 计算基于batch size的任务发送间隔
            // 如果样本速率是 S samples/sec，batch size是 B，那么任务发送速率应该是 S/B tasks/sec
            let task_rate = current_sample_rate / batch_size as f64;
            let interval = self.calculate_poisson_interval(task_rate);

            debug!(
                "模型{}: 样本速率={:.2} samples/sec, batch_size={}, 任务速率={:.2} tasks/sec, 间隔={:?}",
                model_name, current_sample_rate, batch_size, task_rate, interval
            );

            // 等待指定间隔
            sleep(interval.min(end_time.saturating_duration_since(Instant::now()))).await;

            // 检查是否还在周期内
            if Instant::now() >= end_time {
                break;
            }

            // 生成并发送任务（使用预先计算的batch size）
            if let Err(e) = self
                .generate_and_send_task_with_batch_size(&model_config, batch_size)
                .await
            {
                crate::error_display::print_error(&format!("发送任务失败: {}", e));
                // 发送失败后等待一小段时间再继续
                sleep(Duration::from_millis(500)).await;
            }
        }

        info!("模型{}的发送任务完成", model_name);
    }

    /// 计算泊松分布的间隔时间
    /// task_rate: 任务发送速率 (tasks/sec) = sample_rate / batch_size
    fn calculate_poisson_interval(&self, task_rate: f64) -> Duration {
        if task_rate <= 0.0 {
            return Duration::from_secs(3600); // 如果速率为0，返回很长的间隔
        }

        // 使用指数分布生成泊松过程的间隔时间
        // 间隔时间 = -ln(U) / λ，其中U是[0,1)的均匀分布，λ是任务发送速率
        let u: f64 = {
            let mut rng = rand::rng();
            rng.random_range(f64::EPSILON..1.0) // 避免ln(0)
        };
        let interval_secs = -u.ln() / task_rate;

        // 限制间隔时间在合理范围内
        let interval_secs = interval_secs.clamp(0.001, 300.0); // 最小1ms，最大5分钟

        Duration::from_secs_f64(interval_secs)
    }

    /// 生成并发送任务（使用指定的batch size）
    pub(crate) async fn generate_and_send_task_with_batch_size(
        &self,
        model_config: &ModelConfig,
        batch_size: usize,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.generate_and_send_task_at(model_config, batch_size, Instant::now(), None)
            .await
    }

    pub(crate) async fn generate_and_send_task_at(
        &self,
        model_config: &ModelConfig,
        batch_size: usize,
        due: Instant,
        task_id: Option<Uuid>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // 生成任务ID
        let task_id = task_id.unwrap_or_else(Uuid::new_v4);

        // 生成 [batch_size, 3, 224, 224] 的张量，使用全1填充
        let dyn_tensor: DynTensor = if self.send_batch_size_only {
            Tensor::<f32>::ones(vec![batch_size, 0, 0, 0]).into_dyn()
        } else {
            Tensor::<f32>::random(vec![batch_size, 3, 224, 224]).into_dyn()
        };

        // 创建任务
        let work = BaseWork::new(task_id, dyn_tensor);
        let task = WorkTask {
            model: model_config.name.clone(),
            work,
        };

        debug!(
            "生成任务: {} (模型: {}, batch_size: {} samples)",
            task_id, model_config.name, batch_size
        );

        // 记录任务发送（传递batch_size）
        self.metrics_tracker
            .record_task_sent_at(
                &task_id.to_string(),
                &model_config.name,
                batch_size as u32,
                due,
            )
            .await;

        let Ok(permit) = self.in_flight.clone().try_acquire_owned() else {
            self.metrics_tracker
                .mark_task_failed_reason(&task_id.to_string(), "sender_overflow")
                .await;
            return Ok(());
        };

        tokio::spawn({
            let metrics_tracker = self.metrics_tracker.clone();
            let http_client = self.http_client.clone();
            let task_id_str = task_id.to_string();
            let mut cancelled = self.cancelled.subscribe();
            let failures_logged = self.failures_logged.clone();
            async move {
                let _permit = permit;
                metrics_tracker.record_http_start(due.elapsed()).await;
                // 使用Bincode发送任务
                let result = tokio::select! {
                    result = http_client.submit_task(task) => result,
                    _ = async { let _ = cancelled.wait_for(|value| *value).await; } => {
                        metrics_tracker.mark_task_failed_reason(&task_id_str, "drain_cancelled").await;
                        return;
                    }
                };
                match result {
                    Ok(result_task) => {
                        debug!(
                            "任务发送并处理成功: {} (模型: {})",
                            task_id, result_task.model
                        );
                        // 任务已完成
                        metrics_tracker.mark_task_completed(&task_id_str).await;
                        debug!("任务完成: {}", task_id);
                    }
                    Err(e) => {
                        // 任务失败或网络错误
                        let reason = e.failure_reason();
                        let reason_count = metrics_tracker
                            .mark_task_failed_reason(&task_id_str, reason)
                            .await;
                        let count = failures_logged.fetch_add(1, Ordering::Relaxed) + 1;
                        if reason_count <= 3 || count <= 10 || count.is_power_of_two() {
                            crate::error_display::print_error(&format!(
                                "任务发送失败 (第 {count} 次, {reason} #{reason_count}): {e}"
                            ));
                        }
                    }
                }
            }
        });

        Ok(())
    }

    pub fn cancel_outstanding(&self) {
        self.cancelled.send_replace(true);
    }

    /// 获取目标样本速率配置
    pub async fn get_target_sample_rates(&self) -> HashMap<String, f64> {
        let target_rates = self.target_sample_rates.read().await;
        target_rates.clone()
    }

    /// 平滑调整发送速率
    async fn adjust_rates_smoothly(&self, models: Vec<String>, duration: Duration) {
        if self.config.rate_change.strategy != RateChangeStrategy::Smooth {
            return; // 如果不是平滑策略，直接返回
        }

        let steps = self.config.rate_change.transition_steps as usize;
        let step_duration = duration / steps as u32;

        debug!(
            "开始平滑速率调整，步数: {}, 每步间隔: {:?}",
            steps, step_duration
        );

        for step in 1..=steps {
            let progress = step as f64 / steps as f64;

            for model in &models {
                let (current_opt, target_opt) = {
                    let current_rates = self.current_sample_rates.read().await;
                    let target_rates = self.target_sample_rates.read().await;
                    (
                        current_rates.get(model).copied(),
                        target_rates.get(model).copied(),
                    )
                };

                if let (Some(current), Some(target)) = (current_opt, target_opt) {
                    let new_rate = current + (target - current) * progress;

                    {
                        let mut current_rates = self.current_sample_rates.write().await;
                        current_rates.insert(model.clone(), new_rate);
                    }

                    debug!(
                        "步骤{}/{}: 模型{}样本速率调整为{:.2} samples/sec",
                        step, steps, model, new_rate
                    );
                }
            }

            sleep(step_duration).await;
        }

        info!("样本速率平滑调整完成");
    }
}

#[cfg(test)]
mod tests {
    use super::replay_request_id;

    #[test]
    fn replay_request_ids_preserve_the_explicit_namespace_and_original_ordinal() {
        for seed in [0, 20260923, u64::MAX] {
            for ordinal in [0, 1, 10, u64::MAX] {
                let id = replay_request_id(seed, ordinal);
                assert_eq!(id, replay_request_id(seed, ordinal));
                assert_eq!(id.as_u128() >> 64, u128::from(seed));
                assert_eq!(id.as_u128() as u64, ordinal);
                assert_ne!(id, replay_request_id(seed ^ 1, ordinal));
                assert_ne!(id, replay_request_id(seed, ordinal ^ 1));
            }
        }
    }
}
