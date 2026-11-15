use std::{future::Future, net::SocketAddr, ops::Deref, sync::Arc};

use tokio::{io::BufStream, net::TcpListener, select};
use tracing::{error, trace, warn};
use utils::sync::Task;

use crate::{
    CommChannel, CommError,
    connection::{Connection, ConnectionBuilder},
    service::{
        ConnectionService,
        chunk::{ChunkLayer, ChunkService},
    },
    transport::IoStream,
};

/// A trait for creating a communication channel of a specific type from a generic I/O stream.
pub trait ChannelCreator<Req, Resp>: Send + Sync + 'static {
    /// The specific type of channel this creator produces.
    type Channel: CommChannel<De = Req, Enc = Resp>;

    /// Creates a new communication channel from the given I/O stream and remote address.
    fn create_channel(&self, stream: IoStream, addr: SocketAddr) -> Self::Channel;
}

/// A trait for services that listen for incoming connections and handle received frames.
pub trait Listener: Sized + Send + Sync + 'static {
    /// The type of request messages this listener expects.
    type Request: Send;

    /// The type of response messages this listener will send.
    type Response: Send;

    /// Handles an error that occurs during connection establishment.
    fn on_connect_error(&self, e: std::io::Error) -> impl Future<Output = ()> + Send {
        async move {
            error!("connection error: {e:?}");
        }
    }

    /// Handles an error that occurs within the communication channel.
    fn on_channel_error(&self, e: CommError) -> impl Future<Output = ()> + Send {
        async move {
            error!("channel error: {e:?}");
        }
    }

    /// Listens for incoming connections on the given TCP listener and handles them.
    ///
    /// This method implements a negotiation protocol:
    /// 1. It accepts a new TCP connection.
    /// 2. It checks if the connection is from the local machine.
    /// 3. If local, it attempts to upgrade the connection to a more performant local IPC socket.
    /// 4. If remote or upgrade fails, it continues with the existing TCP connection.
    /// 5. Once the final transport is determined, it creates a channel and handles its lifecycle.
    fn listen<Creator>(
        self,
        task: Task,
        tcp_listener: TcpListener,
        channel_creator: Creator,
    ) -> impl Future<Output = ()> + Send
    where
        Creator: ChannelCreator<Self::Request, Self::Response>,
    {
        listen_impl(self, task, tcp_listener, channel_creator, false)
    }

    /// Stop accepting frames on cancellation, finish already dispatched requests,
    /// and track connections until their final response has been sent.
    /// The caller must cancel `task`, then await `task.wait()` before cleanup.
    fn listen_gracefully<Creator>(
        self,
        task: Task,
        tcp_listener: TcpListener,
        channel_creator: Creator,
    ) -> impl Future<Output = ()> + Send
    where
        Creator: ChannelCreator<Self::Request, Self::Response>,
    {
        listen_impl(self, task, tcp_listener, channel_creator, true)
    }

    /// Handles a received data frame and returns a response.
    fn on_recv_frame(&self, frame: Self::Request) -> impl Future<Output = Self::Response> + Send;
}

async fn listen_impl<L: Listener, Creator: ChannelCreator<L::Request, L::Response>>(
    listener: L,
    task: Task,
    tcp_listener: TcpListener,
    channel_creator: Creator,
    graceful: bool,
) {
    let listener = Arc::new(listener);
    let creator = Arc::new(channel_creator);

    while !task.is_cancelled() {
        let (tcp_stream, addr) = select! {
            result = tcp_listener.accept() => {
                match result {
                    Ok(result) => result,
                    Err(e) => {
                        let listener = listener.clone();
                        task.spawn(move |_| async move { listener.on_connect_error(e).await });
                        continue;
                    }
                }
            },
            _ = task.cancelled() => {
                trace!("Task cancelled, stopping listener.");
                return;
            }
        };

        let sub_task = task.sub_task();
        let listener = listener.clone();
        let creator = creator.clone();

        // Spawn a new task to handle the negotiation and lifecycle of each connection.
        let handle_connection = move |connection_task: Task| async move {
            let final_stream = match if graceful {
                select! {
                    biased;
                    _ = connection_task.cancelled() => return,
                    result = IoStream::handle_tcp_stream(tcp_stream, addr) => result,
                }
            } else {
                IoStream::handle_tcp_stream(tcp_stream, addr).await
            } {
                Ok(stream) => stream,
                Err(e) => {
                    warn!("Transport negotiation failed: {:?}", e);
                    return;
                }
            };

            let mut channel = creator.create_channel(final_stream, addr);

            trace!(%addr, "New connection established and channel created.");
            loop {
                let frame = if graceful {
                    select! {
                        biased;
                        _ = connection_task.cancelled() => break,
                        result = channel.recv() => result,
                    }
                } else {
                    channel.recv().await
                };
                match frame {
                    Ok(Some(recv_frame)) => {
                        trace!(%addr, "Received frame.");
                        let send_frame = listener.on_recv_frame(recv_frame).await;
                        if let Err(e) = channel.send(&send_frame).await {
                            trace!(%addr, "Terminating connection due to send failure: {:?}", e);
                            listener.on_channel_error(e).await;
                            break;
                        }
                    }
                    Ok(None) => {
                        trace!(%addr, "Channel closed gracefully.");
                        break;
                    }
                    Err(e) => {
                        trace!(%addr, "Terminating connection due to receive failure: {:?}", e);
                        listener.on_channel_error(e).await;
                        break;
                    }
                }
            }
        };
        if graceful {
            sub_task.spawn_cooperative(handle_connection);
        } else {
            sub_task.spawn(handle_connection);
        }
    }
}

impl<L, T> Listener for T
where
    L: Listener,
    T: Deref<Target = L> + 'static + Send + Sync,
{
    type Request = L::Request;
    type Response = L::Response;

    fn on_recv_frame(&self, frame: Self::Request) -> impl Future<Output = Self::Response> + Send {
        L::on_recv_frame(self, frame)
    }
}

/// A simple channel creator implementation.
pub struct SimpleChannelCreator<C> {
    max_frame_size: usize,
    buffer_capacity: usize,
    _marker: std::marker::PhantomData<fn() -> C>,
}

impl<C> SimpleChannelCreator<C> {
    /// Creates a new `SimpleChannelCreator`.
    pub fn new(max_frame_size: usize, buffer_capacity: usize) -> Self {
        Self {
            max_frame_size,
            buffer_capacity,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<Req, Resp, C> ChannelCreator<Req, Resp> for SimpleChannelCreator<C>
where
    C: CommChannel<De = Req, Enc = Resp>
        + From<Connection<ChunkService<ConnectionService<BufStream<IoStream>>>>>
        + 'static,
{
    type Channel = C;

    fn create_channel(&self, stream: IoStream, _addr: SocketAddr) -> Self::Channel {
        // Framed grows to fit received frames. Reserving max_frame_size * 10
        // per direction eagerly costs ~735 MiB for a 38 MiB tensor frame.
        // Bound only the initial I/O buffers, not the protocol frame limit.
        let service = ConnectionService::new(BufStream::with_capacity(
            self.buffer_capacity
                .saturating_mul(self.max_frame_size)
                .min(64 * 1024),
            self.buffer_capacity
                .saturating_mul(self.max_frame_size)
                .min(64 * 1024),
            stream,
        ))
        .with_max_frame_length(self.max_frame_size);
        ConnectionBuilder::new(service)
            .build::<Connection<_>>()
            .layer(ChunkLayer::default())
            .into()
    }
}
