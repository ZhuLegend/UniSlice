//! 任务追踪模块
//!
//! 提供统一的任务追踪接口，用于记录virtual_compute_node中任务的生命周期事件。
//! 所有追踪事件都以结构化JSON格式输出到专用的日志文件中。

use std::time::{SystemTime, UNIX_EPOCH};

use tracing::info;
use uuid::Uuid;

use super::worker_registry::WorkerId;

/// 任务追踪目标名称常量
pub(in crate::compute_node::r#virtual::virtual_compute_node) const TASK_TRACKER_TARGET: &str =
    "virtual_compute_node::task_tracker";

/// 任务追踪事件类型
#[derive(Debug, Clone)]
pub enum TaskTrackingEvent {
    /// 任务到达事件 - 当任务首次进入RequestScheduler时触发
    TaskArrived {
        /// 任务的唯一标识符
        task_id: Uuid,
        /// 模型标识符
        model_id: String,
        /// 批次大小
        batch_size: usize,
    },
    /// 任务分配事件 - 当任务被分配给特定worker时触发
    TaskAssigned {
        /// 任务的唯一标识符
        task_id: Uuid,
        /// Worker的唯一标识符
        worker_id: WorkerId,
        /// 模型标识符
        model_id: String,
        /// 批次大小
        batch_size: usize,
    },
    /// 计算失败时间 - 但worker计算失败时触发
    ComputeFailed {
        /// 任务的唯一标识符
        task_id: Uuid,
        /// Worker的唯一标识符
        worker_id: WorkerId,
        /// 模型标识符
        model_id: String,
        /// 错误信息
        error_message: String,
    },
    /// 计算完成事件 - 当worker完成任务计算时触发
    ComputeCompleted {
        /// 任务的唯一标识符
        task_id: Uuid,
        /// Worker的唯一标识符
        worker_id: WorkerId,
        /// 模型标识符
        model_id: String,
        /// 批次大小
        batch_size: usize,
        /// 处理持续时间（毫秒）
        processing_duration_ms: u64,
    },
}

/// 任务追踪器
///
/// 负责记录virtual_compute_node中的任务生命周期事件，
/// 所有事件都会以结构化的方式记录到专用的追踪日志中。
pub struct TaskTracker;

impl TaskTracker {
    /// 创建新的任务追踪器
    pub fn new() -> Self {
        Self
    }

    /// 记录任务到达事件
    ///
    /// 当任务首次进入RequestScheduler::add_compute_task时调用
    pub fn track_task_arrived(&self, task_id: Uuid, model_id: &str, batch_size: usize) {
        self.track_task_arrived_at(task_id, model_id, batch_size, get_timestamp_ms());
    }

    /// Record the actual enqueue time even if log emission runs on another thread.
    pub fn track_task_arrived_at(
        &self,
        task_id: Uuid,
        model_id: &str,
        batch_size: usize,
        timestamp_ms: u64,
    ) {
        info!(
            target: TASK_TRACKER_TARGET,
            event = "task_arrived",
            task_id = %task_id,
            model_id = %model_id,
            batch_size = batch_size,
            timestamp_ms = timestamp_ms,
            "Task arrived"
        );
    }

    /// 记录任务分配事件
    ///
    /// 当任务被分配给特定worker时调用
    pub fn track_task_assigned(
        &self,
        task_id: Uuid,
        worker_id: WorkerId,
        model_id: &str,
        batch_size: usize,
    ) {
        let timestamp_ms = get_timestamp_ms();
        info!(
            target: TASK_TRACKER_TARGET,
            event = "task_assigned",
            task_id = %task_id,
            worker_id = %worker_id,
            model_id = %model_id,
            batch_size = batch_size,
            timestamp_ms = timestamp_ms,
            "Task assigned"
        );
    }

    /// 记录计算失败事件
    ///
    /// 当worker计算失败时调用
    pub fn track_compute_failed(
        &self,
        task_id: Uuid,
        worker_id: WorkerId,
        model_id: &str,
        error_message: String,
    ) {
        let timestamp_ms = get_timestamp_ms();
        info!(
            target: TASK_TRACKER_TARGET,
            event = "compute_failed",
            task_id = %task_id,
            worker_id = %worker_id,
            model_id = %model_id,
            error_message = %error_message,
            timestamp_ms = timestamp_ms,
            "Compute failed"
        );
    }

    /// 记录计算完成事件
    ///
    /// 当worker完成任务计算时调用
    pub fn track_compute_completed(
        &self,
        task_id: Uuid,
        worker_id: WorkerId,
        model_id: &str,
        batch_size: usize,
        processing_duration_ms: u64,
    ) {
        let timestamp_ms = get_timestamp_ms();
        info!(
            target: TASK_TRACKER_TARGET,
            event = "compute_completed",
            task_id = %task_id,
            worker_id = %worker_id,
            model_id = %model_id,
            batch_size = batch_size,
            processing_duration_ms = processing_duration_ms,
            timestamp_ms = timestamp_ms,
            "Compute completed"
        );
    }

    /// 批量记录任务分配事件
    ///
    /// 当多个任务被同时分配给同一个worker时调用
    pub fn track_batch_tasks_assigned(
        &self,
        task_ids: &[Uuid],
        worker_id: WorkerId,
        model_id: &str,
        total_batch_size: usize,
    ) {
        let timestamp_ms = get_timestamp_ms();
        for &task_id in task_ids {
            info!(
                target: TASK_TRACKER_TARGET,
                event = "task_assigned",
                task_id = %task_id,
                worker_id = %worker_id,
                model_id = %model_id,
                batch_size = total_batch_size,
                timestamp_ms = timestamp_ms,
                "Task assigned (batch)"
            );
        }
    }

    /// 批量记录计算失败事件
    ///
    /// 当worker计算多个任务失败时调用
    pub fn track_batch_compute_failed(
        &self,
        task_ids: &[Uuid],
        worker_id: WorkerId,
        model_id: &str,
        error_message: String,
    ) {
        let timestamp_ms = get_timestamp_ms();
        for &task_id in task_ids {
            info!(
                target: TASK_TRACKER_TARGET,
                event = "compute_failed",
                task_id = %task_id,
                worker_id = %worker_id,
                model_id = %model_id,
                error_message = %error_message,
                timestamp_ms = timestamp_ms,
                "Compute failed (batch)"
            );
        }
    }

    /// 批量记录计算完成事件
    ///
    /// 当worker完成批量任务计算时调用
    pub fn track_batch_compute_completed(
        &self,
        task_ids: &[Uuid],
        worker_id: WorkerId,
        model_id: &str,
        total_batch_size: usize,
        processing_duration_ms: u64,
    ) {
        let timestamp_ms = get_timestamp_ms();
        for &task_id in task_ids {
            info!(
                target: TASK_TRACKER_TARGET,
                event = "compute_completed",
                task_id = %task_id,
                worker_id = %worker_id,
                model_id = %model_id,
                batch_size = total_batch_size,
                processing_duration_ms = processing_duration_ms,
                timestamp_ms = timestamp_ms,
                "Compute completed (batch)"
            );
        }
    }
}

impl Default for TaskTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// 获取当前毫秒级时间戳
fn get_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// 全局任务追踪器实例
///
/// 使用懒加载的方式创建全局追踪器，避免重复创建开销
pub fn global_task_tracker() -> &'static TaskTracker {
    use std::sync::OnceLock;
    static GLOBAL_TRACKER: OnceLock<TaskTracker> = OnceLock::new();
    GLOBAL_TRACKER.get_or_init(TaskTracker::new)
}
