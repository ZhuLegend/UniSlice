use std::net::{Ipv4Addr, SocketAddr};

#[cfg(feature = "torch")]
use bincode::serde::Compat;
use comm::{
    CommChannel, CommError,
    connection::{ConnectionBuilder, IoStream, TypedConnection},
    get_unused_port,
};
#[cfg(feature = "torch")]
use model::tensor::DynTensor;
#[cfg(feature = "torch")]
use tch::Tensor;
use tokio::net::TcpListener;

/// 辅助函数，用于启动一个TCP服务器并返回其监听地址
async fn start_test_server() -> Result<(TcpListener, SocketAddr), CommError> {
    let addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), get_unused_port());
    let listener = TcpListener::bind(addr).await.unwrap();
    let server_addr = listener.local_addr().unwrap();
    Ok((listener, server_addr))
}

#[tokio::test]
/// 测试发送字符串
async fn test_str() -> Result<(), CommError> {
    let (listener, addr) = start_test_server().await?;

    let client_task = async move {
        let mut client: TypedConnection<_, &str, String> = TypedConnection::connect(addr).await?;
        client.send(&"Hello, world!").await?;
        Ok::<(), CommError>(())
    };

    let server_task = async move {
        let (server_io, addr) = listener.accept().await.unwrap();
        let mut server: TypedConnection<_, &str, String> =
            ConnectionBuilder::new_io(IoStream::handle_tcp_stream(server_io, addr).await.unwrap())
                .build();
        let received = server.recv().await?.unwrap();
        assert_eq!(received, "Hello, world!");
        Ok::<(), CommError>(())
    };

    let (client_res, server_res) = tokio::join!(client_task, server_task);
    client_res?;
    server_res?;
    Ok(())
}

#[cfg(feature = "torch")]
#[tokio::test]
/// 测试发送DynTensor
async fn test_token_repr() -> Result<(), CommError> {
    let (listener, addr) = start_test_server().await?;
    let test_tensor = Tensor::ones([2, 3], (tch::Kind::Float, tch::Device::Cpu));
    let tensor_repr = DynTensor::from(test_tensor.copy());

    let client_task = async move {
        let mut client: TypedConnection<_, DynTensor, ()> = TypedConnection::connect(addr).await?;
        client.send(&tensor_repr).await?;
        Ok::<(), CommError>(())
    };

    let server_task = async move {
        let (server_io, addr) = listener.accept().await.unwrap();
        let mut server: TypedConnection<_, (), DynTensor> =
            ConnectionBuilder::new_io(IoStream::handle_tcp_stream(server_io, addr).await.unwrap())
                .build();
        let received: DynTensor = server.recv().await?.unwrap();
        assert_eq!(Tensor::from(received), test_tensor);
        Ok::<(), CommError>(())
    };

    let (client_res, server_res) = tokio::join!(client_task, server_task);
    client_res?;
    server_res?;
    Ok(())
}

#[cfg(feature = "torch")]
#[tokio::test]
#[ignore] // This test consumes a lot of memory and time, run it manually.
/// 测试发送包含大量数据的DynTensor
async fn test_large_token_repr() -> Result<(), CommError> {
    let (listener, addr) = start_test_server().await?;
    // 4bytes * 256 * 1024 * 1024 = 1GB
    let test_tensor = Tensor::ones([256, 1024, 1024], (tch::Kind::Float, tch::Device::Cpu));
    let tensor_repr = Compat(DynTensor::from(test_tensor.copy()));

    let client_task = async move {
        let mut client: TypedConnection<_, _, ()> = ConnectionBuilder::connect(addr)
            .await?
            .buffered_with_capacity(1024 * 1024 * 1024 + 100, 1024 * 1024 * 1024 + 100)
            .build();
        client.send(&tensor_repr).await?;
        Ok::<(), CommError>(())
    };

    let server_task = async move {
        let (server_io, addr) = listener.accept().await.unwrap();
        let mut server: TypedConnection<_, (), _> =
            ConnectionBuilder::new_io(IoStream::handle_tcp_stream(server_io, addr).await.unwrap())
                .buffered_with_capacity(1024 * 1024 * 1024 + 100, 1024 * 1024 * 1024 + 100)
                .build();
        let received: Compat<DynTensor> = server.recv().await?.unwrap();
        assert_eq!(Tensor::from(received.0), test_tensor);
        Ok::<(), CommError>(())
    };

    let (client_res, server_res) = tokio::join!(client_task, server_task);
    client_res?;
    server_res?;
    Ok(())
}

#[cfg(feature = "torch")]
#[tokio::test]
/// 测试发送多帧DynTensor、按次解析
async fn test_multi_token_repr() -> Result<(), CommError> {
    let (listener, addr) = start_test_server().await?;
    let test_tensor1 = Tensor::ones([2, 3], (tch::Kind::Float, tch::Device::Cpu));
    let tensor_repr1 = DynTensor::from(test_tensor1.copy());
    let test_tensor2 = Tensor::ones([3, 2], (tch::Kind::Float, tch::Device::Cpu));
    let tensor_repr2 = DynTensor::from(test_tensor2.copy());

    let client_task = async move {
        let mut client: TypedConnection<_, DynTensor, ()> = TypedConnection::connect(addr).await?;
        client.send(&tensor_repr1).await?;
        client.send(&tensor_repr2).await?;
        Ok::<(), CommError>(())
    };

    let server_task = async move {
        let (server_io, addr) = listener.accept().await.unwrap();
        let mut server: TypedConnection<_, (), DynTensor> =
            ConnectionBuilder::new_io(IoStream::handle_tcp_stream(server_io, addr).await.unwrap())
                .build();
        let received1: DynTensor = server.recv().await?.unwrap();
        let received2: DynTensor = server.recv().await?.unwrap();
        assert_eq!(Tensor::from(received1), test_tensor1);
        assert_eq!(Tensor::from(received2), test_tensor2);
        Ok::<(), CommError>(())
    };

    let (client_res, server_res) = tokio::join!(client_task, server_task);
    client_res?;
    server_res?;
    Ok(())
}
