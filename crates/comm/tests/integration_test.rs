use std::{net::SocketAddr, time::Duration};

use comm::{
    connection::{Connection, ConnectionBuilder, IoStream},
    service::{Backoff, ConnectionService, RetryLayer, RetryPolicy},
};
use tokio::{io::BufStream, net::TcpListener};

async fn start_echo_server() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (socket, addr) = listener.accept().await.unwrap();
        let mut conn: Connection<ConnectionService<IoStream>> =
            ConnectionBuilder::new_io(IoStream::handle_tcp_stream(socket, addr).await.unwrap())
                .build();
        while let Ok(Some(data)) = conn.recv().await {
            conn.send(data).await.unwrap();
        }
    });
    addr
}

#[tokio::test]
async fn test_simple_comm() {
    let server_addr = start_echo_server().await;
    let mut client: Connection<ConnectionService<BufStream<IoStream>>> =
        ConnectionBuilder::connect(server_addr)
            .await
            .unwrap()
            .build();

    client.send("hello".into()).await.unwrap();
    let response = client.recv().await.unwrap().unwrap();
    assert_eq!(response, "hello");
}

#[tokio::test]
async fn test_connection_with_retry_layer() {
    let server_addr = start_echo_server().await;
    let policy = RetryPolicy::new(3, Backoff::Fixed(Duration::from_millis(10)), false);
    let mut client: Connection<_> = ConnectionBuilder::connect(server_addr)
        .await
        .unwrap()
        .layer(RetryLayer::new(policy))
        .build();

    // Just test a simple request-response to ensure the layered connection works.
    client.send("hello with retry".into()).await.unwrap();
    let response = client.recv().await.unwrap().unwrap();
    assert_eq!(response, "hello with retry");
}
