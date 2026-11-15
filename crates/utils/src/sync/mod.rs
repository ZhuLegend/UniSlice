//! 并发模块

use std::future::Future;

use tokio::{select, task::JoinHandle};
use tokio_util::{
    sync::{CancellationToken, WaitForCancellationFuture},
    task::TaskTracker,
};

/// 异步任务
///
/// 当由同一个任务clone出去的其余任务共享取消状态和操作
#[derive(Clone, Debug)]
pub struct Task {
    tracker: TaskTracker,
    token: CancellationToken,
}

impl Task {
    /// 创建一个支持传递取消状态的任务
    pub fn new() -> Self {
        Self {
            tracker: TaskTracker::new(),
            token: CancellationToken::new(),
        }
    }

    /// 创建一个子任务，当父任务取消时会自动取消所有子任务
    pub fn sub_task(&self) -> Self {
        Self {
            tracker: self.tracker.clone(),
            token: self.token.child_token(),
        }
    }

    /// 创建一个新的任务
    ///
    /// 通过监听传入的[Task]的取消状态来取消任务
    ///
    /// **由于接受的参数是[Future]，在任务取消时，[Future]不再被轮询，其拥有的所有资源都会被立刻调用析构函数并释放**
    /// 由于[tokio::task::spawn]创建的任务不会自动取消，同时[JoinHandle]也是Future的一种，如果你传入的是[JoinHandle]，请自行管理其取消状态
    ///
    /// 如果任务创建时传入的[Task]已经被取消，那么任务会立即被取消，不会创建
    pub fn spawn<R1: Future + Send + 'static, F1>(&self, task: F1) -> JoinHandle<Option<R1::Output>>
    where
        R1::Output: Send + 'static,
        F1: FnOnce(Self) -> R1 + Send + 'static,
    {
        let new_self = self.clone();

        let handle = self.tracker.spawn(async move {
            let token = new_self.token.clone();

            select! {
                res = task(new_self) => {
                    Some(res)
                },
                _ = token.cancelled() => {
                    None
                },
            }
        });

        handle
    }

    /// Track a task that observes cancellation at its own safe boundaries.
    /// Unlike `spawn`, cancellation does not drop its in-flight future.
    pub fn spawn_cooperative<R: Future + Send + 'static, F>(&self, task: F) -> JoinHandle<R::Output>
    where
        R::Output: Send + 'static,
        F: FnOnce(Self) -> R + Send + 'static,
    {
        self.tracker.spawn(task(self.clone()))
    }

    /// 创建阻塞任务
    pub fn spawn_blocking<R1: Send + 'static, F1: FnOnce() -> R1 + Send + 'static>(
        &self,
        task: F1,
    ) -> JoinHandle<Option<R1>> {
        let new_self = self.clone();

        let handle = self.tracker.spawn_blocking(move || {
            let token = new_self.token.clone();

            if token.is_cancelled() {
                return None;
            }

            Some(task())
        });

        handle
    }

    /// 取消任务及其所有子任务
    pub fn cancel(&self) {
        self.token.cancel();
        self.tracker.close();
    }

    /// 判断任务是否被取消
    pub fn is_cancelled(&self) -> bool {
        self.token.is_cancelled()
    }

    /// 等待取消指令
    pub fn cancelled(&self) -> WaitForCancellationFuture<'_> {
        self.token.cancelled()
    }

    /// 等待共享任务树中的所有任务结束，包括父任务和兄弟任务
    pub async fn wait(&self) {
        self.tracker.close();

        self.tracker.wait().await;
    }
}

impl Default for Task {
    fn default() -> Self {
        Self::new()
    }
}
