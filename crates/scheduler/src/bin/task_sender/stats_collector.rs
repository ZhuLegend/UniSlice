//! 状态收集器模块
//!
//! 负责收集sender和daemon的统计信息，并以JSON格式记录到文件
//! 所有统计都基于样本数量而非任务数量

use std::{path::PathBuf, sync::Arc};

use chrono::Utc;
use tokio::{fs::OpenOptions, io::AsyncWriteExt};
use tracing::debug;

use crate::{
    http_client::HttpClient,
    metrics_tracker::MetricsTracker,
    types::{DaemonStats, SampleCounts, SenderMetrics, SenderStats},
};

/// 状态收集器
pub struct StatsCollector {
    /// 指标追踪器
    metrics_tracker: Arc<MetricsTracker>,
    /// HTTP客户端，用于查询daemon状态
    http_client: Arc<HttpClient>,
    /// 输出文件路径
    output_file: PathBuf,
}

impl StatsCollector {
    /// 创建新的状态收集器
    pub fn new(
        metrics_tracker: Arc<MetricsTracker>,
        http_client: Arc<HttpClient>,
        log_dir: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // 确保日志目录存在
        std::fs::create_dir_all(log_dir)?;

        let output_file = PathBuf::from(log_dir).join("task_sender_stats.json");

        Ok(Self {
            metrics_tracker,
            http_client,
            output_file,
        })
    }

    /// 收集并保存统计信息
    pub async fn collect_and_save(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        debug!("开始收集统计信息");

        // 收集发送器指标
        let sender_metrics = self.collect_sender_metrics().await;

        // 收集daemon统计信息
        let daemon_stats = self.collect_daemon_stats().await;

        // 组装完整的统计信息
        let stats = SenderStats {
            timestamp: Utc::now(),
            sender_metrics,
            daemon_stats,
        };

        // 保存到文件
        self.save_stats(&stats).await?;

        debug!("统计信息收集完成");
        Ok(())
    }

    /// 收集发送器指标
    async fn collect_sender_metrics(&self) -> SenderMetrics {
        // 获取每个模型的样本处理速率和任务发送速率（基于统计窗口）
        let model_sample_rates = self.metrics_tracker.get_model_sample_rates().await;
        let model_task_rates = self.metrics_tracker.get_model_task_rates().await;
        let total_sample_rate = model_sample_rates.values().sum();
        let total_task_rate = model_task_rates.values().sum();

        // 获取每个模型的移动平均延迟
        let model_average_latencies = self.metrics_tracker.get_model_average_latencies().await;

        // 获取样本计数
        let (
            sent_tasks,
            completed_tasks,
            pending_tasks,
            failed_tasks,
            sent_samples,
            completed_samples,
            pending_samples,
            failed_samples,
        ) = self.metrics_tracker.get_sample_counts().await;

        let sample_counts = SampleCounts {
            sent_tasks,
            completed_tasks,
            pending_tasks,
            failed_tasks,
            sent_samples,
            completed_samples,
            pending_samples,
            failed_samples,
        };

        SenderMetrics {
            model_sample_rates,
            model_task_rates,
            model_average_latencies,
            total_sample_rate,
            total_task_rate,
            sample_counts,
        }
    }

    /// 收集daemon统计信息
    async fn collect_daemon_stats(&self) -> DaemonStats {
        // 获取基本统计信息
        let stats = match self.http_client.get_daemon_stats().await {
            Ok(data) => {
                debug!("成功获取daemon基本统计信息");
                Some(data)
            }
            Err(e) => {
                tracing::warn!("获取daemon基本统计信息失败: {}", e);
                None
            }
        };

        // 获取预测统计信息
        let predictions = match self.http_client.get_prediction_stats().await {
            Ok(data) => {
                debug!("成功获取daemon预测统计信息");
                Some(data)
            }
            Err(e) => {
                tracing::warn!("获取daemon预测统计信息失败: {}", e);
                None
            }
        };

        // 获取调度器统计信息
        let scheduler_stats = match self.http_client.get_scheduler_stats().await {
            Ok(data) => {
                debug!("成功获取daemon调度器统计信息");
                Some(data)
            }
            Err(e) => {
                tracing::warn!("获取daemon调度器统计信息失败: {}", e);
                None
            }
        };

        DaemonStats {
            stats,
            predictions,
            scheduler_stats,
        }
    }

    /// 保存统计信息到文件
    async fn save_stats(&self, stats: &SenderStats) -> Result<(), Box<dyn std::error::Error>> {
        // 序列化为JSON
        let json_line = serde_json::to_string(stats)?;

        // 追加到文件（JSONL格式）
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.output_file)
            .await?;

        file.write_all(json_line.as_bytes()).await?;
        file.write_all(b"\n").await?;
        file.flush().await?;

        debug!("统计信息已保存到: {:?}", self.output_file);
        Ok(())
    }
}
