use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};

use comm::CommChannel;
use compute_node::{
    ComputeNode,
    util::bin_compute_node::{
        BinComputeNode, BinComputeNodeRequest, BinComputeNodeResponse, BinComputeNodeWrapper,
        connect_bin_compute_node,
    },
};
use tokio::{
    net::{TcpListener, TcpStream},
    sync::{Notify, oneshot},
    time::timeout,
};

#[derive(Default)]
struct Observations {
    started: Notify,
    finish: Notify,
    completed: AtomicUsize,
    cleanups: AtomicUsize,
    stall_cleanup: AtomicBool,
}
struct Node {
    observations: Arc<Observations>,
    fail_cleanup: bool,
}

impl ComputeNode for Node {
    type Command = ();
    type State = ();
    type Status = ();
    type InputData = u32;
    type OutputData = u32;
    async fn handle(&self, _: ()) -> Result<(), String> {
        Ok(())
    }
    async fn state(&self) -> Result<(), String> {
        Ok(())
    }
    async fn status(&self) -> Result<(), String> {
        Ok(())
    }
    async fn compute(&self, value: u32) -> Result<u32, String> {
        self.observations.started.notify_one();
        self.observations.finish.notified().await;
        self.observations.completed.fetch_add(1, Ordering::SeqCst);
        Ok(value)
    }
}
impl BinComputeNode for Node {
    async fn shutdown(&self) -> Result<(), String> {
        self.observations.cleanups.fetch_add(1, Ordering::SeqCst);
        if self.observations.stall_cleanup.load(Ordering::SeqCst) {
            self.observations.finish.notified().await;
        }
        if self.fail_cleanup {
            Err("cleanup failed".into())
        } else {
            Ok(())
        }
    }
}

#[tokio::test]
async fn shutdown_drains_dispatched_work_and_closes_idle_connections() {
    let observations = Arc::new(Observations::default());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (stop, stopped) = oneshot::channel();
    let wrapper = BinComputeNodeWrapper::builder()
        .wrap(Node {
            observations: observations.clone(),
            fail_cleanup: false,
        })
        .build();
    let mut server = tokio::spawn(wrapper.serve_until(listener, async {
        let _ = stopped.await;
    }));
    // One negotiated idle connection and one connection stalled during negotiation.
    let mut idle = connect_bin_compute_node::<(), (), (), u32, u32>(addr)
        .await
        .unwrap();
    let _negotiating = TcpStream::connect(addr).await.unwrap();
    let mut client = connect_bin_compute_node::<(), (), (), u32, u32>(addr)
        .await
        .unwrap();
    client
        .send(&BinComputeNodeRequest::Compute(42))
        .await
        .unwrap();
    timeout(Duration::from_secs(2), observations.started.notified())
        .await
        .unwrap();
    stop.send(()).unwrap();
    assert!(
        timeout(Duration::from_millis(30), &mut server)
            .await
            .is_err()
    );
    assert_eq!(observations.cleanups.load(Ordering::SeqCst), 0);
    observations.finish.notify_one();
    assert!(matches!(
        timeout(Duration::from_secs(2), client.recv())
            .await
            .unwrap()
            .unwrap(),
        Some(BinComputeNodeResponse::Compute(42))
    ));
    timeout(Duration::from_secs(2), server)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(observations.completed.load(Ordering::SeqCst), 1);
    assert_eq!(observations.cleanups.load(Ordering::SeqCst), 1);
    assert!(matches!(
        timeout(Duration::from_secs(2), idle.recv()).await.unwrap(),
        Ok(None) | Err(_)
    ));
    assert!(TcpStream::connect(addr).await.is_err());
}

#[tokio::test]
async fn shutdown_returns_cleanup_error_and_still_closes_listener() {
    let observations = Arc::new(Observations::default());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let wrapper = BinComputeNodeWrapper::builder()
        .wrap(Node {
            observations: observations.clone(),
            fail_cleanup: true,
        })
        .build();
    assert_eq!(
        wrapper.serve_until(listener, async {}).await.unwrap_err(),
        "cleanup failed"
    );
    assert_eq!(observations.cleanups.load(Ordering::SeqCst), 1);
    assert!(TcpStream::connect(addr).await.is_err());
}

#[tokio::test]
async fn shutdown_times_out_without_unloading_a_running_request() {
    let observations = Arc::new(Observations::default());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (stop, stopped) = oneshot::channel();
    let wrapper = BinComputeNodeWrapper::builder()
        .wrap(Node {
            observations: observations.clone(),
            fail_cleanup: false,
        })
        .shutdown_timeout(Duration::from_millis(30))
        .build();
    let server = tokio::spawn(wrapper.serve_until(listener, async {
        let _ = stopped.await;
    }));
    let mut client = connect_bin_compute_node::<(), (), (), u32, u32>(addr)
        .await
        .unwrap();
    client
        .send(&BinComputeNodeRequest::Compute(42))
        .await
        .unwrap();
    timeout(Duration::from_secs(2), observations.started.notified())
        .await
        .unwrap();
    stop.send(()).unwrap();
    let error = timeout(Duration::from_secs(2), server)
        .await
        .unwrap()
        .unwrap()
        .unwrap_err();
    assert!(error.contains("request drain timed out"));
    assert_eq!(observations.cleanups.load(Ordering::SeqCst), 0);
    assert!(TcpStream::connect(addr).await.is_err());
    // The binary exits on this error. Finish the fake operation here so the
    // test runtime also retires the deliberately blocked connection task.
    observations.finish.notify_one();
    assert!(matches!(
        timeout(Duration::from_secs(2), client.recv())
            .await
            .unwrap()
            .unwrap(),
        Some(BinComputeNodeResponse::Compute(42))
    ));
}

#[tokio::test]
async fn shutdown_reports_a_stalled_cleanup() {
    let observations = Arc::new(Observations::default());
    observations.stall_cleanup.store(true, Ordering::SeqCst);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let wrapper = BinComputeNodeWrapper::builder()
        .wrap(Node {
            observations: observations.clone(),
            fail_cleanup: false,
        })
        .shutdown_timeout(Duration::from_millis(30))
        .build();
    let error = timeout(
        Duration::from_secs(2),
        wrapper.serve_until(listener, async {}),
    )
    .await
    .unwrap()
    .unwrap_err();
    assert!(error.contains("node cleanup timed out"));
    assert_eq!(observations.cleanups.load(Ordering::SeqCst), 1);
}
