//! 工具函数模块

use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use dashmap::DashMap;

use crate::daemon::types::{TaskStats, TaskStatus};

/// 获取当前时间戳（毫秒）
pub fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// 更新任务状态
pub fn update_task_status(
    stats: &Arc<DashMap<String, TaskStats>>,
    task_id: &str,
    status: TaskStatus,
    error: Option<String>,
) {
    if let Some(mut task_stats) = stats.get_mut(task_id) {
        task_stats.status = status;
        task_stats.error = error;

        if matches!(
            task_stats.status,
            TaskStatus::Completed | TaskStatus::Failed
        ) {
            task_stats.completed_at = Some(current_timestamp());
        }
    }
}

/// 优雅关闭信号
pub async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            tracing::info!("收到 Ctrl+C 信号");
        },
        _ = terminate => {
            tracing::info!("收到终止信号");
        },
    }

    tracing::info!("开始优雅关闭...");
}

/// 解析监听地址
pub fn parse_listen_address(host: &str, port: u16) -> Result<std::net::SocketAddr, String> {
    format!("{}:{}", host, port)
        .parse()
        .map_err(|e| format!("无效的监听地址: {}", e))
}

/// 生成唯一的任务ID
pub fn generate_task_id() -> String {
    uuid::Uuid::new_v4().to_string()
}
