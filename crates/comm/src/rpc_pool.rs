//! Reuse framed RPC connections without serializing independent requests.
use std::{
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use bincode::{Decode, Encode};
use tokio::{io::BufStream, time::timeout};

use crate::{
    CommChannel, TypedConnection,
    connection::IoStream,
    service::{ChunkLayer, ConnectionService, chunk::ChunkService},
};

type Channel<Req, Resp> =
    TypedConnection<ChunkService<ConnectionService<BufStream<IoStream>>>, Req, Resp>;

/// An idle connection cache. Each in-flight RPC exclusively owns its connection.
/// Errors, cancellation and timeouts drop that connection. Requests are never
/// replayed automatically: a timed-out command may already have taken effect.
pub struct RpcPool<Req: Encode + Send, Resp: Decode<()> + Send> {
    address: String,
    idle: Mutex<Vec<Channel<Req, Resp>>>,
    opened: AtomicU64,
}
impl<Req: Encode + Send + Sync, Resp: Decode<()> + Send + 'static> RpcPool<Req, Resp> {
    /// Connect lazily to this address; idle connections are bounded to 32.
    pub fn new(address: String) -> Self {
        Self {
            address,
            idle: Mutex::new(Vec::new()),
            opened: AtomicU64::new(0),
        }
    }
    /// Exchange one request and response with separate phase deadlines.
    pub async fn request(
        &self,
        request: &Req,
        connect_budget: Duration,
        send_budget: Duration,
        response_budget: Duration,
    ) -> Result<Resp, String> {
        let started = std::time::Instant::now();
        // The mutex is never held across I/O. Parallel requests can open separate
        // connections instead of waiting behind a long inference/control request.
        let idle = self.idle.lock().unwrap().pop();
        let reused = idle.is_some();
        let mut channel = match idle {
            Some(channel) => channel,
            None => {
                let channel = timeout(connect_budget, TypedConnection::connect(&self.address))
                    .await
                    .map_err(|_| "Timeout connecting to RPC peer".to_string())?
                    .map_err(|e| format!("RPC connection failed: {e}"))?
                    .layer(ChunkLayer::default());
                let opened = self.opened.fetch_add(1, Ordering::Relaxed) + 1;
                tracing::info!(target: "rpc_timing", event = "connection_opened", address = %self.address, opened);
                channel
            }
        };
        let connect_ms = started.elapsed().as_secs_f64() * 1000.0;
        timeout(send_budget, channel.send(request))
            .await
            .map_err(|_| "Timeout sending RPC request".to_string())?
            .map_err(|e| format!("RPC send failed: {e}"))?;
        let sent_ms = started.elapsed().as_secs_f64() * 1000.0;
        let response = timeout(response_budget, channel.recv())
            .await
            .map_err(|_| "Timeout waiting for RPC response".to_string())?
            .map_err(|e| format!("RPC receive failed: {e}"))?
            .ok_or_else(|| "RPC peer closed connection".to_string())?;
        // Only a completely received response makes the stream safe to reuse.
        let mut idle = self.idle.lock().unwrap();
        if idle.len() < 32 {
            idle.push(channel);
        }
        tracing::debug!(target: "rpc_timing", address = %self.address, reused, connect_ms, sent_ms, total_ms = started.elapsed().as_secs_f64() * 1000.0);
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use utils::sync::Task;

    use super::*;
    use crate::{Listener, SimpleChannelCreator};
    struct Echo;
    impl Listener for Echo {
        type Request = (u64, u64);
        type Response = u64;
        async fn on_recv_frame(&self, (value, delay_ms): Self::Request) -> u64 {
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            value
        }
    }
    async fn setup() -> (RpcPool<(u64, u64), u64>, Task, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let pool = RpcPool::new(listener.local_addr().unwrap().to_string());
        let task = Task::new();
        let serving = tokio::spawn(Echo.listen(
            task.clone(),
            listener,
            SimpleChannelCreator::<Channel<u64, (u64, u64)>>::new(1024, 1),
        ));
        (pool, task, serving)
    }
    async fn request(
        pool: &RpcPool<(u64, u64), u64>,
        value: u64,
        delay: u64,
    ) -> Result<u64, String> {
        pool.request(
            &(value, delay),
            Duration::from_secs(2),
            Duration::from_secs(2),
            Duration::from_secs(2),
        )
        .await
    }
    #[tokio::test]
    async fn reuse_and_concurrency() {
        let (pool, task, serving) = setup().await;
        assert_eq!(request(&pool, 1, 0).await.unwrap(), 1);
        assert_eq!(request(&pool, 2, 0).await.unwrap(), 2);
        assert_eq!(pool.opened.load(Ordering::Relaxed), 1);
        let (a, b) = tokio::join!(request(&pool, 3, 100), request(&pool, 4, 100));
        assert_eq!((a.unwrap(), b.unwrap()), (3, 4));
        assert_eq!(
            pool.opened.load(Ordering::Relaxed),
            2,
            "concurrent work must not share one busy stream"
        );
        task.cancel();
        serving.await.unwrap();
    }
    #[tokio::test]
    async fn cancelled_or_timed_out_rpc_never_reuses_a_stale_response() {
        let (pool, task, serving) = setup().await;
        assert!(
            tokio::time::timeout(Duration::from_millis(20), request(&pool, 1, 150))
                .await
                .is_err()
        );
        assert_eq!(request(&pool, 2, 0).await.unwrap(), 2);
        assert_eq!(pool.opened.load(Ordering::Relaxed), 2);
        assert!(
            pool.request(
                &(3, 150),
                Duration::from_secs(2),
                Duration::from_secs(2),
                Duration::from_millis(20)
            )
            .await
            .is_err()
        );
        assert_eq!(request(&pool, 4, 0).await.unwrap(), 4);
        assert_eq!(pool.opened.load(Ordering::Relaxed), 3);
        task.cancel();
        serving.await.unwrap();
    }
    #[tokio::test]
    async fn frames_larger_than_initial_io_buffers_round_trip() {
        struct BytesEcho;
        impl Listener for BytesEcho {
            type Request = Vec<u8>;
            type Response = Vec<u8>;
            async fn on_recv_frame(&self, frame: Vec<u8>) -> Vec<u8> {
                frame
            }
        }
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let pool: RpcPool<Vec<u8>, Vec<u8>> =
            RpcPool::new(listener.local_addr().unwrap().to_string());
        let task = Task::new();
        let serving = tokio::spawn(BytesEcho.listen(
            task.clone(),
            listener,
            SimpleChannelCreator::<Channel<Vec<u8>, Vec<u8>>>::new(2 * 1024 * 1024, 10),
        ));
        let bytes = vec![37; 1024 * 1024];
        let budget = Duration::from_secs(5);
        assert_eq!(
            pool.request(&bytes, budget, budget, budget).await.unwrap(),
            bytes
        );
        task.cancel();
        serving.await.unwrap();
    }
}
