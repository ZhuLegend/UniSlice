//! 通信实用模块

#![warn(missing_docs)]

use std::future::Future;

use bincode::{Decode, Encode};
use snafu::Snafu;
use tokio::net::TcpSocket;

pub mod connection;
mod listener;
mod negotiation;
pub mod rpc_pool;
pub mod service;
mod transport;
mod util;

#[cfg(test)]
mod test;

pub use connection::TypedConnection;
pub use listener::{ChannelCreator, Listener, SimpleChannelCreator};

/// 通信错误
#[derive(Debug, Snafu)]
pub enum CommError {
    /// IO错误
    #[snafu(display("IO error: {}", source))]
    Io {
        /// 原始[std::io::Error]错误
        source: std::io::Error,
    },
    /// Bincode编码错误
    #[snafu(display("Bincode encode error: {}", source))]
    Encode {
        /// 原始[bincode::error::EncodeError]错误
        source: bincode::error::EncodeError,
    },
    /// Bincode解码错误
    #[snafu(display("Bincode decode error: {}", source))]
    Decode {
        /// 原始[bincode::error::DecodeError]错误
        source: bincode::error::DecodeError,
    },
    /// 超时错误
    #[snafu(display("Timeout"))]
    Timeout,
    /// 连接关闭
    #[snafu(display("Connection closed"))]
    ConnectionClosed,
    /// 任务连接错误
    #[snafu(display("Task join error: {}", source))]
    Join {
        /// 原始[tokio::task::JoinError]错误
        source: tokio::task::JoinError,
    },
}

/// 通信通道
pub trait CommChannel: Send {
    /// 发送的数据类型
    type Enc: Encode + Send;
    /// 接收的数据类型
    type De: Decode<()> + Send;

    /// 发送指定类型的数据包
    ///
    /// 如果连接已经关闭，返回`Ok(())`
    fn send(&mut self, item: &Self::Enc) -> impl Future<Output = Result<(), CommError>> + Send;

    /// 接收指定类型的数据包数据包
    ///
    /// 如果连接已经关闭，返回`Ok(None)`
    fn recv(&mut self) -> impl Future<Output = Result<Option<Self::De>, CommError>> + Send;
}

/// 获取一个空闲的端口号
pub fn get_unused_port() -> u16 {
    let socket = TcpSocket::new_v4().unwrap();
    let _ = socket.bind("127.0.0.1:0".parse().unwrap());
    let addr = socket.local_addr().unwrap();
    drop(socket);
    addr.port()
}
