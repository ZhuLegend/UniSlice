use std::{
    net::SocketAddr,
    pin::Pin,
    task::{Context, Poll},
};

use interprocess::local_socket::{
    GenericFilePath, ListenerOptions, ToFsName,
    tokio::prelude::LocalSocketStream,
    traits::tokio::{Listener, Stream},
};
use snafu::ResultExt;
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, ReadBuf},
    net::{TcpStream, ToSocketAddrs},
};
use tracing::{trace, warn};

use crate::{CommError, negotiation, negotiation::NegotiationMessage, util::is_local_addr};

/// A unified stream that can be either a TCP stream or a local IPC stream.
/// It implements `AsyncRead` and `AsyncWrite` to be transparently used by the rest of the communication stack.
#[derive(Debug)]
pub enum IoStream {
    /// A stream over the network.
    Tcp(TcpStream),
    /// A stream over a local socket (Unix Domain Socket or Windows Named Pipe).
    Local(LocalSocketStream),
}

impl AsyncRead for IoStream {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        match self.get_mut() {
            IoStream::Tcp(s) => Pin::new(s).poll_read(cx, buf),
            IoStream::Local(s) => Pin::new(s).poll_read(cx, buf),
        }
    }
}

impl AsyncWrite for IoStream {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<Result<usize, std::io::Error>> {
        match self.get_mut() {
            IoStream::Tcp(s) => Pin::new(s).poll_write(cx, buf),
            IoStream::Local(s) => Pin::new(s).poll_write(cx, buf),
        }
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), std::io::Error>> {
        match self.get_mut() {
            IoStream::Tcp(s) => Pin::new(s).poll_flush(cx),
            IoStream::Local(s) => Pin::new(s).poll_flush(cx),
        }
    }

    fn poll_shutdown(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<(), std::io::Error>> {
        match self.get_mut() {
            IoStream::Tcp(s) => Pin::new(s).poll_shutdown(cx),
            IoStream::Local(s) => Pin::new(s).poll_shutdown(cx),
        }
    }
}

impl IoStream {
    /// Connects to a remote address, negotiates the transport, and creates a `TypedConnection`.
    ///
    /// It first connects via TCP. Then, it waits for the server to either offer an upgrade
    /// to a local IPC transport (if on the same machine) or to continue with TCP.
    pub async fn connect<A: ToSocketAddrs>(addr: A) -> Result<Self, CommError> {
        let mut tcp_stream = TcpStream::connect(addr).await.context(crate::IoSnafu {})?;

        // Wait for the server's negotiation message.
        match negotiation::read_message(&mut tcp_stream).await? {
            NegotiationMessage::UpgradeToIpc { name } => {
                trace!("Received offer to upgrade to IPC with name: {}", name);
                let name = name
                    .as_str()
                    .to_fs_name::<interprocess::local_socket::GenericFilePath>()
                    .unwrap();
                match LocalSocketStream::connect(name).await {
                    Ok(ipc_stream) => {
                        trace!(
                            "Successfully connected to IPC socket. Sending acceptance to server."
                        );
                        // Send acceptance message to server
                        negotiation::send_message(
                            &mut tcp_stream,
                            &NegotiationMessage::AcceptIpcUpgrade,
                        )
                        .await?;
                        // Gracefully shut down the TCP stream to signal completion.
                        use tokio::io::AsyncWriteExt;
                        let _ = tcp_stream.shutdown().await;
                        Ok(IoStream::Local(ipc_stream))
                    }
                    Err(e) => {
                        trace!("IPC connection failed, rejecting upgrade: {:?}", e);
                        // Send rejection message to server
                        negotiation::send_message(
                            &mut tcp_stream,
                            &NegotiationMessage::RejectIpcUpgrade,
                        )
                        .await?;
                        // Continue with TCP connection
                        trace!("Falling back to TCP connection.");
                        Ok(IoStream::Tcp(tcp_stream))
                    }
                }
            }
            NegotiationMessage::ContinueOnTcp => {
                trace!("Continuing with TCP connection.");
                Ok(IoStream::Tcp(tcp_stream))
            }
            _ => {
                warn!("Unexpected negotiation message, falling back to TCP.");
                Ok(IoStream::Tcp(tcp_stream))
            }
        }
    }

    /// Connects to a remote address using TCP only, bypassing any negotiation.
    pub async fn connect_tcp<A: ToSocketAddrs>(addr: A) -> Result<Self, CommError> {
        let mut tcp_stream = TcpStream::connect(addr).await.context(crate::IoSnafu {})?;
        trace!("Forcing TCP connection, negotiation skipped.");
        // For TCP-only connections, we need to send a [NegotiationMessage::RejectIpcUpgrade] message to the server
        // to indicate we're not waiting for negotiation
        let msg = NegotiationMessage::RejectIpcUpgrade;
        let _ = negotiation::read_message(&mut tcp_stream).await?;
        negotiation::send_message(&mut tcp_stream, &msg).await?;
        Ok(IoStream::Tcp(tcp_stream))
    }

    /// Handles an incoming TCP stream by negotiating the transport.
    ///
    /// If the connection is from a local address, it attempts to upgrade to a local IPC transport.
    /// If the upgrade is accepted by the client, it switches to IPC. Otherwise, it continues with TCP.
    /// For remote connections, it continues with TCP without attempting an upgrade.
    pub async fn handle_tcp_stream(
        mut tcp_stream: TcpStream,
        addr: SocketAddr,
    ) -> Result<Self, CommError> {
        if is_local_addr(addr.ip()) {
            // Local connection, attempt to upgrade to IPC.
            let ipc_name_str = format!("hsgs-{}.sock", uuid::Uuid::new_v4());
            let ipc_name = ipc_name_str
                .as_str()
                .to_fs_name::<GenericFilePath>()
                .unwrap();
            let ipc_option = ListenerOptions::new().name(ipc_name);

            let ipc_listener = match ipc_option.create_tokio() {
                Ok(listener) => listener,
                Err(e) => return Err(CommError::Io { source: e }),
            };

            let msg = NegotiationMessage::UpgradeToIpc {
                name: ipc_name_str.clone(),
            };
            negotiation::send_message(&mut tcp_stream, &msg).await?;

            // Wait for client's response
            match negotiation::read_message(&mut tcp_stream).await? {
                NegotiationMessage::AcceptIpcUpgrade => {
                    trace!(%addr, "Client accepted IPC upgrade, waiting for IPC connection.");
                    // Wait for the client to connect to the IPC socket.
                    match interprocess::local_socket::tokio::Listener::accept(&ipc_listener).await {
                        Ok(ipc_stream) => {
                            trace!(%addr, "Successfully upgraded connection to local IPC.");
                            // Spawn a background task to hold the TCP stream until the client closes it.
                            tokio::spawn(async move {
                                let mut buf = [0];
                                // Wait for the client to shutdown its end of the TCP stream.
                                let _ = tcp_stream.read(&mut buf).await;
                                // The tcp_stream is dropped here, closing the connection.
                            });
                            Ok(IoStream::Local(ipc_stream))
                        }
                        Err(e) => {
                            warn!("IPC upgrade failed, falling back to TCP: {:?}", e);
                            // Fallback to TCP if IPC connection fails
                            Ok(IoStream::Tcp(tcp_stream))
                        }
                    }
                }
                NegotiationMessage::RejectIpcUpgrade => {
                    trace!(%addr, "Client rejected IPC upgrade, continuing with TCP.");
                    // Client rejected upgrade, continue with TCP
                    Ok(IoStream::Tcp(tcp_stream))
                }
                _ => {
                    warn!(%addr, "Unexpected negotiation message, falling back to TCP.");
                    // Unexpected message, fallback to TCP
                    Ok(IoStream::Tcp(tcp_stream))
                }
            }
        } else {
            // Remote connection, continue with TCP.
            negotiation::send_message(&mut tcp_stream, &NegotiationMessage::ContinueOnTcp).await?;
            trace!(%addr, "Continuing with TCP for remote connection.");
            // For remote connections, we don't need to wait for client negotiation response
            Ok(IoStream::Tcp(tcp_stream))
        }
    }
}
