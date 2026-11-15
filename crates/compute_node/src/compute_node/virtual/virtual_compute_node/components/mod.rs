//! VirtualComputeNode组件模块
//!
//! 这个模块包含了VirtualComputeNode的所有子组件，每个组件都有明确的职责：
//! - WorkerManager: 负责Worker生命周期管理和指标收集
//! - RequestScheduler: 负责请求调度、批处理分发和任务队列管理
//! - ModelServiceManager: 负责模型处理器管理和模型到达率跟踪
//! - WorkerHandler: 负责单个Worker的任务执行和自适应批大小控制

/// 模型服务管理器模块
pub mod model_service_manager;

/// 请求调度器模块
pub mod request_scheduler;
/// Worker状态通知器模块
pub mod state_notifier;
/// 任务追踪器模块
pub mod task_tracker;
/// 测试工具模块
#[cfg(test)]
pub mod tests;
mod worker_handler;
/// Worker管理器模块
pub mod worker_manager;
/// Worker注册表模块
pub mod worker_registry;

// 导出组件模块的公共接口
pub(super) use model_service_manager::ModelServiceManager;
pub(super) use request_scheduler::RequestScheduler;
pub(super) use state_notifier::{StateListenerHandle, WorkerStateNotifier};
pub(super) use task_tracker::global_task_tracker;
pub(super) use worker_handler::WORKER_PERF_TARGET;
pub(super) use worker_manager::BatchSize;
pub(super) use worker_registry::{NodeId, WorkerId, WorkerRegistry};
