mod task {
    use std::sync::{Arc, Mutex};

    use tokio::time::Duration;

    use crate::sync::Task;

    #[tokio::test]
    async fn parent_wait_tracks_descendants_and_child_cancel_is_local() {
        let parent = Task::new();
        let child = parent.sub_task();
        child.cancel();
        assert!(!parent.is_cancelled());
        let child = parent.sub_task();
        let (release, released) = tokio::sync::oneshot::channel();
        child.spawn(|_| async { released.await.unwrap() });
        let waiter = tokio::spawn({
            let parent = parent.clone();
            async move { parent.wait().await }
        });
        tokio::task::yield_now().await;
        assert!(!waiter.is_finished());
        release.send(()).unwrap();
        waiter.await.unwrap();
        parent.cancel();
        assert!(child.is_cancelled());
    }

    /// 测试创建一个等待十秒的任务，在两秒后取消任务
    #[tokio::test]
    async fn test_task() {
        let task = Task::new();

        let handle = task.spawn(|_| async {
            tokio::time::sleep(Duration::from_secs(10)).await;
            true
        });

        tokio::time::sleep(Duration::from_secs(2)).await;

        task.cancel();

        assert!(handle.await.unwrap().is_none());
        task.wait().await;
    }

    /// 创建两个任务，任务A先创建子任务B，子任务B等待五秒后返回true，任务A创建任务B后等待两秒取消子任务并等待五秒，任务A返回子任务B的结果
    #[tokio::test]
    async fn test_sub_task() {
        let task = Task::new();
        let sub_task_result: Arc<Mutex<Option<bool>>> = Arc::new(Mutex::new(None));
        let sub_task_result_clone = sub_task_result.clone();

        let handle = task.spawn(|task| async move {
            task.sub_task().spawn(|_| async move {
                tokio::time::sleep(Duration::from_secs(5)).await;
                sub_task_result_clone.lock().unwrap().replace(true);
                true
            });

            tokio::time::sleep(Duration::from_secs(2)).await;

            task.cancel();

            true
        });

        // 父任务A正常执行
        assert_eq!(handle.await.unwrap(), Some(true));
        // 子任务B被取消
        assert_eq!(sub_task_result.lock().unwrap().as_ref(), None);
        task.wait().await;
    }

    /// 测试取消父任务时子任务级联取消
    #[tokio::test]
    async fn test_cascade_cancel() {
        let task = Task::new();
        let sub_task_result: Arc<Mutex<Option<bool>>> = Arc::new(Mutex::new(None));
        let sub_task_result_clone = sub_task_result.clone();

        let handle = task.spawn(|task| async move {
            task.sub_task().spawn(|_| async move {
                tokio::time::sleep(Duration::from_secs(5)).await;
                sub_task_result_clone.lock().unwrap().replace(true);
                true
            });

            tokio::time::sleep(Duration::from_secs(2)).await;

            true
        });
        task.cancel();

        // 父任务A被取消
        assert_eq!(handle.await.unwrap(), None);
        // 子任务B被取消
        assert_eq!(sub_task_result.lock().unwrap().as_ref(), None);
        task.wait().await;
    }
}
