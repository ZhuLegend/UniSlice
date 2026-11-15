//! 实时显示组件模块
//!
//! 使用 indicatif 库在终端实时显示各模型的指标
//! 所有显示都基于样本数量而非任务数量

use std::{collections::HashMap, sync::Arc, time::Duration};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use tokio::time::interval;
use tokio_util::sync::CancellationToken;
use tracing::debug;

use crate::{metrics_tracker::MetricsTracker, poisson_sender::PoissonSender};

/// 实时显示器
pub struct LiveDisplay {
    /// 各模型的进度条
    model_progress_bars: HashMap<String, ProgressBar>,
    /// 总体统计进度条
    pub summary_bar: ProgressBar,
    /// 取消令牌
    cancel_token: CancellationToken,
    /// 后台任务句柄
    _handle: tokio::task::JoinHandle<()>,
}

impl LiveDisplay {
    /// 创建新的实时显示器
    pub fn new(
        models: Vec<String>,
        metrics_tracker: Arc<MetricsTracker>,
        poisson_sender: Arc<PoissonSender>,
    ) -> Self {
        let multi_progress = MultiProgress::new();
        let mut model_progress_bars = HashMap::new();

        // 为每个模型创建进度条
        for model in &models {
            let pb = multi_progress.add(ProgressBar::new_spinner());
            pb.set_style(
                ProgressStyle::default_spinner()
                    .template("{spinner:.green} {prefix:.bold.dim} │ {msg}")
                    .unwrap(),
            );
            pb.set_prefix(format!("模型[{}]", model));
            pb.set_message("初始化中...");
            pb.enable_steady_tick(Duration::from_millis(120));

            model_progress_bars.insert(model.clone(), pb);
        }

        // 创建总体统计进度条
        let summary_bar = multi_progress.add(ProgressBar::new_spinner());
        summary_bar.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.blue} {prefix:.bold} │ {msg}")
                .unwrap(),
        );
        summary_bar.set_prefix("总体统计");
        summary_bar.enable_steady_tick(Duration::from_millis(100));

        let cancel_token = CancellationToken::new();

        // 启动后台更新任务
        let _handle = tokio::spawn({
            let metrics_tracker = metrics_tracker.clone();
            let poisson_sender = poisson_sender.clone();
            let model_progress_bars = model_progress_bars.clone();
            let summary_bar = summary_bar.clone();
            let cancel_token = cancel_token.clone();
            let models = models.clone();

            async move {
                Self::update_loop(
                    metrics_tracker,
                    poisson_sender,
                    model_progress_bars,
                    summary_bar,
                    models,
                    cancel_token,
                )
                .await;
            }
        });

        Self {
            model_progress_bars,
            summary_bar,
            cancel_token,
            _handle,
        }
    }

    /// 更新循环
    async fn update_loop(
        metrics_tracker: Arc<MetricsTracker>,
        poisson_sender: Arc<PoissonSender>,
        model_progress_bars: HashMap<String, ProgressBar>,
        summary_bar: ProgressBar,
        models: Vec<String>,
        cancel_token: CancellationToken,
    ) {
        let mut interval = interval(Duration::from_millis(500)); // 每500ms更新一次

        loop {
            tokio::select! {
                _ = cancel_token.cancelled() => {
                    debug!("实时显示更新循环已取消");
                    break;
                }
                _ = interval.tick() => {
                    Self::update_displays(
                        &metrics_tracker,
                        &poisson_sender,
                        &model_progress_bars,
                        &summary_bar,
                        &models,
                    ).await;
                }
            }
        }
    }

    /// 更新所有显示
    async fn update_displays(
        metrics_tracker: &MetricsTracker,
        poisson_sender: &PoissonSender,
        model_progress_bars: &HashMap<String, ProgressBar>,
        summary_bar: &ProgressBar,
        models: &[String],
    ) {
        // 获取指标数据
        let model_sample_rates = metrics_tracker.get_model_sample_rates().await;
        let model_task_rates = metrics_tracker.get_model_task_rates().await;
        let model_latencies = metrics_tracker.get_model_average_latencies().await;
        let target_sample_rates = poisson_sender.get_target_sample_rates().await;
        let (
            sent_tasks,
            completed_tasks,
            pending_tasks,
            failed_tasks,
            sent_samples,
            completed_samples,
            pending_samples,
            failed_samples,
        ) = metrics_tracker.get_sample_counts().await;

        // 更新每个模型的进度条
        for model in models {
            if let Some(pb) = model_progress_bars.get(model) {
                let target_sample_rate = target_sample_rates.get(model).copied().unwrap_or(0.0);
                let current_sample_rate = model_sample_rates.get(model).copied().unwrap_or(0.0);
                let current_task_rate = model_task_rates.get(model).copied().unwrap_or(0.0);
                let avg_latency = model_latencies.get(model).copied().unwrap_or(0.0);

                // 构建完整的状态消息
                let status_msg = if avg_latency > 0.0 {
                    format!(
                        "目标: {:.1} samples/s │ 当前: {:.1} samples/s ({:.2} tasks/s) │ 延迟: {:.1}ms",
                        target_sample_rate, current_sample_rate, current_task_rate, avg_latency
                    )
                } else {
                    format!(
                        "目标: {:.1} samples/s │ 当前: {:.1} samples/s ({:.2} tasks/s) │ 延迟: --",
                        target_sample_rate, current_sample_rate, current_task_rate
                    )
                };
                pb.set_message(status_msg);
                pb.set_position((current_sample_rate * 10.0) as u64); // 将样本速率放大10倍作为位置显示
            }
        }

        // 更新总体统计
        let total_sample_rate: f64 = model_sample_rates.values().sum();
        let total_task_rate: f64 = model_task_rates.values().sum();
        let status_msg = format!(
            "任务: {}/{}/{}/{} │ 样本: {}/{}/{}/{} │ 样本速率: {:.1}/s │ 任务速率: {:.2}/s",
            completed_tasks,
            pending_tasks,
            failed_tasks,
            sent_tasks,
            completed_samples,
            pending_samples,
            failed_samples,
            sent_samples,
            total_sample_rate,
            total_task_rate
        );
        summary_bar.set_message(status_msg);
        summary_bar.set_position(completed_samples);
    }

    /// 停止实时显示
    pub fn stop(&self) {
        self.cancel_token.cancel();

        // 清理所有进度条
        for pb in self.model_progress_bars.values() {
            pb.finish_and_clear();
        }
        self.summary_bar.finish_and_clear();
    }

    /// 打印最终统计信息
    pub async fn print_final_stats(&self, metrics_tracker: &MetricsTracker) {
        let (
            sent_tasks,
            completed_tasks,
            pending_tasks,
            failed_tasks,
            sent_samples,
            completed_samples,
            pending_samples,
            failed_samples,
        ) = metrics_tracker.get_sample_counts().await;
        let model_sample_rates = metrics_tracker.get_model_sample_rates().await;
        let model_task_rates = metrics_tracker.get_model_task_rates().await;
        let model_latencies = metrics_tracker.get_model_average_latencies().await;

        println!("\n=== 最终统计 ===");
        println!("任务统计:");
        println!("  已发送任务: {}", sent_tasks);
        println!("  已完成任务: {}", completed_tasks);
        println!("  待处理任务: {}", pending_tasks);
        println!("  失败任务: {}", failed_tasks);

        println!("\n样本统计:");
        println!("  已发送样本: {}", sent_samples);
        println!("  已完成样本: {}", completed_samples);
        println!("  待处理样本: {}", pending_samples);
        println!("  失败样本: {}", failed_samples);

        if completed_tasks > 0 {
            let task_success_rate = (completed_tasks as f64 / sent_tasks as f64) * 100.0;
            println!("  任务成功率: {:.1}%", task_success_rate);
        }

        if completed_samples > 0 {
            let sample_success_rate = (completed_samples as f64 / sent_samples as f64) * 100.0;
            println!("  样本成功率: {:.1}%", sample_success_rate);
        }

        println!("\n模型统计:");
        for (model, sample_rate) in model_sample_rates {
            let task_rate = model_task_rates.get(&model).copied().unwrap_or(0.0);
            let avg_latency = model_latencies.get(&model).copied().unwrap_or(0.0);
            println!(
                "  {}: {:.1} samples/s ({:.2} tasks/s), 平均延迟: {:.1}ms",
                model, sample_rate, task_rate, avg_latency
            );
        }
        println!("==================\n");
    }
}

impl Drop for LiveDisplay {
    fn drop(&mut self) {
        self.stop();
    }
}
