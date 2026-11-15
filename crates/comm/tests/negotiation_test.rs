use std::future::Future;

use bincode::{Decode, Encode};
use comm::{CommChannel, Listener, SimpleChannelCreator, TypedConnection};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use utils::sync::Task;

#[derive(Serialize, Deserialize, Encode, Decode, Debug, PartialEq, Eq, Clone)]
struct TestRequest {
    content: String,
}

#[derive(Serialize, Deserialize, Encode, Decode, Debug, PartialEq, Eq, Clone)]
struct TestResponse {
    content: String,
}

#[derive(Clone)]
struct EchoListener;

impl Listener for EchoListener {
    type Request = TestRequest;
    type Response = TestResponse;

    fn on_recv_frame(&self, frame: Self::Request) -> impl Future<Output = Self::Response> + Send {
        async move {
            TestResponse {
                content: frame.content,
            }
        }
    }
}

use comm::service::ChunkLayer;
use tracing::info;

async fn setup_server() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    info!("Test server listening on port {}", port);

    tokio::spawn(async move {
        let task = Task::new();
        let channel_creator: SimpleChannelCreator<TypedConnection<_, TestResponse, TestRequest>> =
            SimpleChannelCreator::new(1024, 10);
        EchoListener.listen(task, listener, channel_creator).await;
    });

    port
}

#[tokio::test]
async fn test_negotiation_local_connection_upgrades_to_ipc() {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_test_writer()
        .try_init();
    let port = setup_server().await;

    info!("Client (local) connecting to port {}", port);
    let mut client: TypedConnection<_, TestRequest, TestResponse> =
        TypedConnection::connect(("127.0.0.1", port))
            .await
            .unwrap()
            .layer(ChunkLayer::default());
    info!("Client (local) connected.");

    let request = TestRequest {
        content: "hello_local".to_string(),
    };
    info!("Client (local) sending request...");
    client.send(&request).await.unwrap();
    info!("Client (local) request sent.");

    info!("Client (local) receiving response...");
    let response = client.recv().await.unwrap().unwrap();
    info!("Client (local) response received.");

    assert_eq!(response.content, request.content);
}

#[tokio::test]
async fn test_negotiation_remote_connection_uses_tcp() {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_test_writer()
        .try_init();
    let port = setup_server().await;

    info!("Client (remote) connecting to port {}", port);
    let mut client: TypedConnection<_, TestRequest, TestResponse> =
        TypedConnection::connect_over_tcp(("127.0.0.1", port))
            .await
            .unwrap()
            .layer(ChunkLayer::default());
    info!("Client (remote) connected.");

    let request = TestRequest {
        content: "hello_remote".to_string(),
    };
    info!("Client (remote) sending request...");
    client.send(&request).await.unwrap();
    info!("Client (remote) request sent.");

    info!("Client (remote) receiving response...");
    let response = client.recv().await.unwrap().unwrap();
    info!("Client (remote) response received.");

    assert_eq!(response.content, request.content);
}
