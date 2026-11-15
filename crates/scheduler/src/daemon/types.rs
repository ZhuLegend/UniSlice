//! 调度器守护进程的通用数据结构和类型定义

use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::{
        Arc, Mutex, RwLock,
        atomic::{AtomicU64, AtomicUsize},
    },
    time::SystemTime,
};

#[cfg(feature = "huawei")]
use compute_node::compute_device::vgpu::resource_spec::hw::HuaweiMigResource;
#[cfg(feature = "nvidia")]
use compute_node::compute_device::vgpu::resource_spec::nv::mig::NvidiaMigResource;
use compute_node::{
    compute_node::r#virtual::virtual_compute_node::{
        WorkTask, components::request_scheduler::batch_size_extractor::BatchExtractorStrategy,
    },
    util::Output,
};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};

use crate::{
    daemon::{csv_logger::CsvTaskLogger, stats::SchedulerStatsManager},
    schedule_system::virtual_compute_node::system::heterogeneous::system::HeterogeneousSystem,
};

/// 延迟数据类型定义
/// model name -> (resource name -> (batch size -> latency in seconds))
pub type LatencyData = latency_profile::LatencyData;

/// 统一的设备配置枚举
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "config")]
pub enum DeviceConfig {
    /// Huawei MIG设备配置
    #[cfg(feature = "huawei")]
    Huawei {
        /// 可选的共享目标列表
        share_targets: Vec<String>,
        /// 资源模版
        resource_template: HuaweiMigResource,
    },
    /// Nvidia MIG设备配置
    #[cfg(feature = "nvidia")]
    Nvidia {
        /// 可选的共享目标列表
        share_targets: Vec<String>,
        /// 资源模版
        resource_template: NvidiaMigResource,
    },
    /// Mock Nvidia MIG设备配置
    #[cfg(all(feature = "mock", feature = "nvidia"))]
    MockNvidia {
        /// 可选的共享目标列表
        share_targets: Vec<String>,
        /// 资源模版
        resource_template: NvidiaMigResource,
    },
    /// Mock Huawei MIG设备配置
    #[cfg(all(feature = "mock", feature = "huawei"))]
    MockHuawei {
        /// 可选的共享目标列表
        share_targets: Vec<String>,
        /// 资源模版
        resource_template: HuaweiMigResource,
    },
}

/// 远程节点配置
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RemoteNodeConfig {
    /// 远程节点IP地址
    pub ip: String,
    /// 远程节点端口号
    pub port: u16,
    /// 设备组合模版名称
    pub template_name: String,
    /// Response budget for lifecycle control RPCs.
    #[serde(default = "default_rpc_timeout")]
    pub control_timeout_ms: u64,
    /// Response budget for inference RPCs.
    #[serde(default = "default_rpc_timeout")]
    pub inference_timeout_ms: u64,
    /// Expected virtual-device configuration identity, set from shared simulation JSON.
    #[serde(default)]
    pub expected_simulation_identity: Option<String>,
}

/// 配置文件结构
#[derive(Debug, Deserialize, Serialize)]
pub struct Config {
    /// 批量提取策略
    pub batch_extractor_strategy: BatchExtractorStrategy,
    /// 设备组合配置列表
    pub device_combinations: HashMap<String, DeviceConfig>,
    /// 远程节点配置列表
    pub remote_nodes: HashMap<String, RemoteNodeConfig>,
    /// 批次大小范围
    pub batch_size_range: (u32, u32),
    /// 初始样本到达率（samples/s；一个请求可包含多个样本）
    pub init_arrival_rate: HashMap<String, f32>,
}

/// 任务处理统计
#[derive(Debug, Clone, Serialize)]
pub struct TaskStats {
    /// 任务ID
    pub task_id: String,
    /// 模型名称
    pub model: String,
    /// 批量大小
    pub batch_size: u64,
    /// 接收时间戳 (Unix时间戳，毫秒)
    pub received_at: u64,
    /// 完成时间戳 (Unix时间戳，毫秒)
    pub completed_at: Option<u64>,
    /// 处理状态
    pub status: TaskStatus,
    /// 错误信息（如果有）
    pub error: Option<String>,
}

/// 任务状态枚举
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum TaskStatus {
    /// 已接收任务
    Received,
    /// 正在处理任务
    Processing,
    /// 已完成任务
    Completed,
    /// 任务失败
    Failed,
}

type ArrivalHistory = HashMap<String, VecDeque<(SystemTime, usize)>>;

/// 模型请求速率预测器
#[derive(Clone, Debug)]
pub struct ModelRequestRatePredictor {
    /// 任务到达历史，包含时间戳和批次大小
    pub arrivals: Arc<RwLock<ArrivalHistory>>,
}

/// Simulation startup barrier: preserve exact configured demand until load begins.
pub struct SimulationStartup {
    /// True only after the entire initial apply succeeds.
    pub ready: std::sync::atomic::AtomicBool,
    /// Latched by the first accepted request.
    pub started: tokio::sync::watch::Sender<bool>,
    /// Exact demand used by initial allocation before requests begin.
    pub initial_rates: HashMap<String, f32>,
}

/// Lifetime budget shared by accepted work and its persistent terminal record.
pub type RequestPermit = Arc<(
    tokio::sync::OwnedSemaphorePermit,
    tokio::sync::OwnedSemaphorePermit,
)>;

/// Cumulative counters survive eviction of detailed task records.
#[derive(Default)]
pub struct TaskTotals {
    /// Accepted tasks.
    pub received: AtomicUsize,
    /// Successfully completed tasks.
    pub completed: AtomicUsize,
    /// Failed or cancelled tasks.
    pub failed: AtomicUsize,
    /// Sum of successful request durations in milliseconds.
    pub completed_duration_ms: AtomicU64,
}

/// 应用状态
#[derive(Clone)]
pub struct AppState {
    /// Global accepted-request and encoded-input byte budgets.
    pub request_slots: Arc<tokio::sync::Semaphore>,
    /// Encoded-input bytes reserved while reading and serving a request.
    pub request_bytes: Arc<tokio::sync::Semaphore>,
    /// Cumulative statistics, independent of recent task details.
    pub task_totals: Arc<TaskTotals>,
    /// At most 100,000 terminal task IDs are retained for HTTP inspection.
    pub terminal_history: Arc<Mutex<VecDeque<String>>>,
    /// Models admitted before recording arrival demand.
    pub supported_models: Arc<HashSet<String>>,
    /// Optional startup barrier for shared-config simulations.
    pub simulation_startup: Option<Arc<SimulationStartup>>,
    /// 任务统计信息
    pub task_stats: Arc<DashMap<String, TaskStats>>,
    /// 异构调度系统
    pub schedule_system: Arc<HeterogeneousSystem>,
    /// 模型请求速率预测器
    pub rate_predictor: ModelRequestRatePredictor,
    /// CSV任务记录器
    pub csv_logger: CsvTaskLogger,
    /// 调度器统计管理器
    pub scheduler_stats: SchedulerStatsManager,
}

/// 任务提交响应
#[derive(Serialize)]
pub struct TaskResponse {
    /// 任务ID
    pub task_id: String,
    /// 提交状态
    pub status: String,
    /// 任务内容（可选）
    pub task: Option<WorkTask<Output>>,
}

/// 统计信息响应
#[derive(Serialize)]
pub struct StatsResponse {
    /// 总任务数
    pub total_tasks: usize,
    /// 已完成任务数
    pub completed_tasks: usize,
    /// 正在处理任务数
    pub processing_tasks: usize,
    /// 失败任务数
    pub failed_tasks: usize,
    /// 平均处理时间（毫秒）
    pub avg_processing_time_ms: Option<f64>,
}

fn default_rpc_timeout() -> u64 {
    30_000
}
