use std::hint::black_box;

use bincode::{Decode, Encode};
use comm::{
    CommChannel, Listener, SimpleChannelCreator, TypedConnection, connection::ConnectionBuilder,
    service::ChunkLayer,
};
use divan::{AllocProfiler, Bencher, counter::BytesCount};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use utils::sync::Task;

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    divan::main();
}

// --- Test Setup ---

const ITERATIONS: usize = 1000;
const SIZES: [usize; 4] = [64, 512, 1024, 4096];
const LARGE_SIZES: [usize; 3] = [1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024]; // 1MB, 4MB, 16MB

#[derive(Serialize, Deserialize, Encode, Decode, Debug, PartialEq, Eq, Clone)]
struct PingPong {
    data: Vec<u8>,
}

#[derive(Clone)]
struct BenchListener;

impl Listener for BenchListener {
    type Request = PingPong;
    type Response = PingPong;

    async fn on_recv_frame(&self, frame: Self::Request) -> Self::Response {
        // For throughput test, we just acknowledge. For ping-pong, we echo.
        PingPong { data: frame.data }
    }
}

async fn setup_server(packet_size: usize) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();

    tokio::spawn(async move {
        let task = Task::new();
        let channel_creator: SimpleChannelCreator<TypedConnection<_, PingPong, PingPong>> =
            SimpleChannelCreator::new(packet_size * 2, 10);
        BenchListener.listen(task, listener, channel_creator).await;
    });

    port
}

// --- Benchmarks ---

#[divan::bench(args = SIZES)]
fn ipc_ping_pong(bencher: Bencher, size: usize) {
    bencher
        .counter(size)
        .with_inputs(|| {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();

            let port = runtime.block_on(setup_server(size));
            let packet = PingPong {
                data: vec![0; size],
            };

            let client = runtime.block_on(async {
                TypedConnection::connect(("127.0.0.1", port))
                    .await
                    .unwrap()
                    .layer(ChunkLayer::default())
            });

            (runtime, client, packet)
        })
        .bench_local_values(move |(runtime, mut client, packet)| {
            runtime.block_on(async {
                for _ in 0..ITERATIONS {
                    client.send(&packet).await.unwrap();
                    let _: PingPong = client.recv().await.unwrap().unwrap();
                }
            });
        });
}

#[divan::bench(args = SIZES)]
fn tcp_ping_pong(bencher: Bencher, size: usize) {
    bencher
        .counter(size)
        .with_inputs(|| {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();

            let port = runtime.block_on(setup_server(size));
            let packet = PingPong {
                data: vec![0; size],
            };

            let client = runtime.block_on(async {
                TypedConnection::connect_over_tcp(("127.0.0.1", port))
                    .await
                    .unwrap()
                    .layer(ChunkLayer::default())
            });

            (runtime, client, packet)
        })
        .bench_local_values(move |(runtime, mut client, packet)| {
            runtime.block_on(async {
                for _ in 0..ITERATIONS {
                    client.send(&packet).await.unwrap();
                    let _: PingPong = client.recv().await.unwrap().unwrap();
                }
            });
        });
}

#[divan::bench(args = LARGE_SIZES)]
fn ipc_throughput(bencher: Bencher, size: usize) {
    bencher
        .counter(BytesCount::new(size))
        .with_inputs(|| {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();

            let port = runtime.block_on(setup_server(size));
            let packet = PingPong {
                data: vec![0; size],
            };

            let client = runtime.block_on(async {
                TypedConnection::connect(("127.0.0.1", port))
                    .await
                    .unwrap()
                    .layer(ChunkLayer::new(1024 * 1024 * 4))
            });

            (runtime, client, packet)
        })
        .bench_local_values(|(runtime, mut client, packet)| {
            runtime.block_on(async {
                client.send(&packet).await.unwrap();
                let received: PingPong = client.recv().await.unwrap().unwrap();
                assert_eq!(received.data.len(), size);
            });
        });
}

#[divan::bench(args = LARGE_SIZES)]
fn tcp_throughput(bencher: Bencher, size: usize) {
    bencher
        .counter(BytesCount::new(size))
        .with_inputs(|| {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();

            let port = runtime.block_on(setup_server(size));
            let packet = PingPong {
                data: vec![0; size],
            };

            let client = runtime.block_on(async {
                TypedConnection::connect_over_tcp(("127.0.0.1", port))
                    .await
                    .unwrap()
                    .layer(ChunkLayer::new(1024 * 1024 * 4))
            });

            (runtime, client, packet)
        })
        .bench_local_values(|(runtime, mut client, packet)| {
            runtime.block_on(async {
                client.send(&packet).await.unwrap();
                let received: PingPong = client.recv().await.unwrap().unwrap();
                assert_eq!(received.data.len(), size);
            });
        });
}

// --- Connection Overhead Benchmarks ---

#[divan::bench]
fn connect_negotiated_tcp(bencher: Bencher) {
    bencher
        .with_inputs(|| {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            // Packet size doesn't matter for connection benchmark
            let port = runtime.block_on(setup_server(1024));
            (runtime, port)
        })
        .bench_local_values(|(runtime, port)| {
            runtime.block_on(async {
                // Benchmark the connection with negotiation
                let _conn = black_box(
                    TypedConnection::<_, PingPong, PingPong>::connect(("127.0.0.1", port))
                        .await
                        .unwrap(),
                );
            });
        });
}

#[divan::bench]
fn connect_forced_tcp(bencher: Bencher) {
    bencher
        .with_inputs(|| {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            // Packet size doesn't matter for connection benchmark
            let port = runtime.block_on(setup_server(1024));
            (runtime, port)
        })
        .bench_local_values(|(runtime, port)| {
            runtime.block_on(async {
                // Benchmark the connection with TCP forced
                let _conn = black_box(
                    TypedConnection::<_, PingPong, PingPong>::connect_over_tcp(("127.0.0.1", port))
                        .await
                        .unwrap(),
                );
            });
        });
}
