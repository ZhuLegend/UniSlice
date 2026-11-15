use std::{sync::Arc, time::Duration};

use dashmap::DashMap;
use tokio::{sync::watch, time};

use super::worker_registry::{WorkerId, WorkerStatus};

/// 状态变更事件
#[derive(Debug, Clone)]
pub struct WorkerStateChangeEvent {
    /// 当前状态
    pub current_status: Option<WorkerStatus>,
    /// 事件发生时间戳
    pub timestamp: std::time::Instant,
}

/// 状态监听器句柄
pub struct StateListenerHandle {
    receiver: watch::Receiver<WorkerStateChangeEvent>,
}

impl StateListenerHandle {
    /// 等待特定worker达到指定状态
    pub async fn wait_for_worker_status(
        &mut self,
        target_status: WorkerStatus,
        timeout: Option<Duration>,
    ) -> Result<(), StateWaitError> {
        if let Some(timeout) = timeout {
            time::timeout(
                timeout,
                self.receiver
                    .wait_for(|event| event.current_status == Some(target_status)),
            )
            .await
            .map_err(|_| StateWaitError::Timeout)?
        } else {
            self.receiver
                .wait_for(|event| event.current_status == Some(target_status))
                .await
        }
        .map_err(|_| StateWaitError::ChannelClosed)
        .map(|_| ())
    }

    /// 等待状态变更事件
    pub async fn wait_for_status_change(
        &mut self,
        timeout: Option<Duration>,
    ) -> Result<WorkerStateChangeEvent, StateWaitError> {
        if let Some(timeout) = timeout {
            time::timeout(timeout, self.receiver.changed())
                .await
                .map_err(|_| StateWaitError::Timeout)?
                .map_err(|_| StateWaitError::ChannelClosed)?;
        } else {
            self.receiver
                .changed()
                .await
                .map_err(|_| StateWaitError::ChannelClosed)?;
        }

        Ok(self.receiver.borrow_and_update().clone())
    }
}

/// 状态等待错误
#[derive(Debug, thiserror::Error)]
pub enum StateWaitError {
    /// 等待超时
    #[error("等待超时")]
    Timeout,
    /// 通道已关闭
    #[error("通道已关闭")]
    ChannelClosed,
}

/// Worker状态通知器
#[derive(Debug, Clone)]
pub struct WorkerStateNotifier {
    /// 广播发送器
    senders: Arc<DashMap<WorkerId, watch::Sender<WorkerStateChangeEvent>>>,
}

impl Default for WorkerStateNotifier {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkerStateNotifier {
    /// 创建新的状态通知器
    pub fn new() -> Self {
        Self {
            senders: Arc::new(DashMap::new()),
        }
    }

    /// 通知worker状态变更
    pub fn notify_state_change(
        &self,
        worker_id: WorkerId,
        new_status: WorkerStatus,
    ) -> Result<(), NotifyError> {
        // 获取或创建状态发送器
        let sender = self.senders.entry(worker_id).or_insert_with(|| {
            let (tx, _) = watch::channel(WorkerStateChangeEvent {
                current_status: Some(new_status),
                timestamp: std::time::Instant::now(),
            });
            tx
        });

        // Commit even without receivers, and change/check the value atomically.
        // The registry calls this while committing its state, so notifications
        // cannot overtake each other on separately scheduled Tokio tasks.
        sender.send_if_modified(|event| {
            if event.current_status == Some(new_status) {
                return false;
            }
            *event = WorkerStateChangeEvent {
                current_status: Some(new_status),
                timestamp: std::time::Instant::now(),
            };
            true
        });

        Ok(())
    }

    /// 创建状态监听器句柄
    pub fn create_listener(&self, worker_id: WorkerId) -> StateListenerHandle {
        let sender = self.senders.entry(worker_id).or_insert_with(|| {
            let (tx, _) = watch::channel(WorkerStateChangeEvent {
                current_status: None,
                timestamp: std::time::Instant::now(),
            });
            tx
        });
        StateListenerHandle {
            receiver: sender.subscribe(),
        }
    }
}

/// 通知错误
#[derive(Debug, thiserror::Error)]
pub enum NotifyError {
    /// 通道已关闭
    #[error("通道已关闭")]
    ChannelClosed,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn committed_state_is_retained_for_a_late_listener() {
        let notifier = WorkerStateNotifier::new();
        let id = uuid::Uuid::new_v4();
        notifier
            .notify_state_change(id, WorkerStatus::Running)
            .unwrap();
        notifier
            .notify_state_change(id, WorkerStatus::Paused)
            .unwrap();
        notifier
            .create_listener(id)
            .wait_for_worker_status(WorkerStatus::Paused, Some(Duration::ZERO))
            .await
            .unwrap();
    }
}
