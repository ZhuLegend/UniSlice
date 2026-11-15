//! 核心数据类型定义

use std::{collections::HashMap, time::Instant};

use chrono::{DateTime, Utc};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};

/// 命令行参数
#[derive(Parser)]
#[command(name = "task-sender")]
#[command(about = "向scheduler daemon发送任务的工具")]
pub struct Args {
    /// 配置文件路径
    #[arg(short, long, default_value = "task_sender_config.toml")]
    pub config: String,

    /// Daemon URL (覆盖配置文件中的设置)
    #[arg(long)]
    pub daemon_url: Option<String>,

    /// 日志输出目录 (覆盖配置文件中的设置)
    #[arg(long)]
    pub log_dir: Option<String>,

    /// 详细输出
    #[arg(short, long)]
    pub verbose: bool,

    /// 发送端是否仅发送batch size即\[batch size, 0, 0, 0\]给服务端
    #[arg(long)]
    pub send_batch_size_only: bool,

    /// Stop sending after this duration and write a final drained snapshot.
    #[arg(long)]
    pub duration_secs: Option<u64>,
    /// JSON array of [offset_seconds, model_name, batch_size]. Requires --duration-secs.
    #[arg(long, requires = "duration_secs")]
    pub replay: Option<std::path::PathBuf>,
    /// Explicit namespace for replay request UUIDs; low 64 bits contain the original ordinal.
    #[arg(long, requires = "replay")]
    pub replay_request_id_seed: Option<u64>,
    /// Maximum time to wait for outstanding requests in finite runs.
    #[arg(long, default_value_t = 300)]
    pub drain_timeout_secs: u64,
    /// Per-HTTP-request timeout, including queueing and reallocation.
    #[arg(long, default_value_t = 300)]
    pub request_timeout_secs: u64,

    /// HTTP/2 uses persistent multiplexed data connections.
    #[arg(long, value_enum, default_value_t = HttpProtocol::Http1)]
    pub http_protocol: HttpProtocol,
    #[arg(long, default_value_t = 4)]
    pub http_connections: usize,
    /// Maximum admitted request futures; overflow is recorded as failure.
    #[arg(long, default_value_t = 262144)]
    pub max_in_flight: usize,

    /// Optional buffered per-request terminal JSONL, including failures and planned arrivals.
    #[arg(long)]
    pub terminal_log: Option<std::path::PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum HttpProtocol {
    Http1,
    Http2,
}

/// 任务生命周期状态
#[derive(Debug, Clone)]
pub struct TaskLifecycle {
    pub _task_id: String,
    pub model: String,
    pub sent_at: Instant,
    pub completed_at: Option<Instant>,
    pub status: LocalTaskStatus,
    pub batch_size: u32,
}

/// 本地任务状态枚举
#[derive(Debug, Clone, PartialEq)]
pub enum LocalTaskStatus {
    Sent,
    Completed,
    Failed,
}

/// 发送器统计信息
#[derive(Debug, Serialize, Deserialize)]
pub struct SenderStats {
    pub timestamp: DateTime<Utc>,
    pub sender_metrics: SenderMetrics,
    pub daemon_stats: DaemonStats,
}

/// 发送器指标
#[derive(Debug, Serialize, Deserialize)]
pub struct SenderMetrics {
    /// 每个模型的样本处理速率 (samples/sec) - 基于统计窗口计算
    pub model_sample_rates: HashMap<String, f64>,
    /// 每个模型的任务发送速率 (tasks/sec) - 基于统计窗口计算
    pub model_task_rates: HashMap<String, f64>,
    /// 每个模型的移动平均延迟 (毫秒) - 基于样本数量加权
    pub model_average_latencies: HashMap<String, f64>,
    /// 总样本处理速率
    pub total_sample_rate: f64,
    /// 总任务发送速率
    pub total_task_rate: f64,
    /// 样本计数
    pub sample_counts: SampleCounts,
}

/// 样本计数 - 基于样本数量而非任务数量
#[derive(Debug, Serialize, Deserialize)]
pub struct SampleCounts {
    /// 已发送任务数
    pub sent_tasks: u64,
    /// 已完成任务数
    pub completed_tasks: u64,
    /// 待处理任务数
    pub pending_tasks: u64,
    /// 失败任务数
    pub failed_tasks: u64,
    /// 已发送样本数
    pub sent_samples: u64,
    /// 已完成样本数
    pub completed_samples: u64,
    /// 待处理样本数
    pub pending_samples: u64,
    /// 失败样本数
    pub failed_samples: u64,
}

/// Daemon统计信息 (从daemon API获取)
#[derive(Debug, Serialize, Deserialize)]
pub struct DaemonStats {
    /// 基本统计
    pub stats: Option<serde_json::Value>,
    /// 预测统计
    pub predictions: Option<serde_json::Value>,
    /// 调度器统计
    pub scheduler_stats: Option<serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::Args;

    #[test]
    fn explicit_request_namespace_requires_replay_and_defaults_to_random_ids() {
        assert!(
            Args::try_parse_from(["task_sender"])
                .unwrap()
                .replay_request_id_seed
                .is_none()
        );
        assert!(Args::try_parse_from(["task_sender", "--replay-request-id-seed", "7"]).is_err());
        let args = Args::try_parse_from([
            "task_sender",
            "--duration-secs",
            "30",
            "--replay",
            "trace.json",
            "--replay-request-id-seed",
            "7",
        ])
        .unwrap();
        assert_eq!(args.replay_request_id_seed, Some(7));
    }
}
