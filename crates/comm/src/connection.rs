//! This module provides the `Connection` type, which is a customizable service for sending and receiving data over a network connection.

use bincode::{Decode, Encode};
use bytes::Bytes;
use snafu::ResultExt;
use tokio::{
    io::{AsyncRead, AsyncWrite, BufStream},
    net::ToSocketAddrs,
};

pub use crate::transport::IoStream;
use crate::{
    CommChannel, CommError, DecodeSnafu, EncodeSnafu, JoinSnafu,
    service::{ConnectionService, Layer, Service},
};

/// A builder for creating a `Connection` with a specific service.
#[derive(Debug)]
pub struct ConnectionBuilder<S: Service> {
    service: S,
}

impl<S: Service> ConnectionBuilder<S> {
    /// Creates a new `ConnectionBuilder` with the given service.
    pub fn new(service: S) -> Self {
        Self { service }
    }

    /// Wrap the current service with a layer, producing a new `ConnectionBuilder`.
    pub fn layer<L: Layer>(self, layer: L) -> ConnectionBuilder<L::Service<S>> {
        ConnectionBuilder {
            service: layer.layer(self.service),
        }
    }

    /// Builds the connection, converting it into a specific type.
    pub fn build<Conn: From<ConnectionBuilder<S>>>(self) -> Conn {
        Conn::from(self)
    }
}

impl<Io> ConnectionBuilder<ConnectionService<Io>>
where
    Io: AsyncRead + AsyncWrite + Send + Unpin + 'static,
{
    /// Creates a new `ConnectionBuilder` with the given IO type.
    pub fn new_io(io: Io) -> Self {
        Self::new(ConnectionService::new(io))
    }

    /// set the maximum frame length for the codec.
    pub fn with_max_frame_length(mut self, length: usize) -> Self {
        self.service = self.service.with_max_frame_length(length);
        self
    }

    /// Wrap the IO with a `BufStream`.
    pub fn buffered(self) -> ConnectionBuilder<ConnectionService<BufStream<Io>>> {
        ConnectionBuilder::new(self.service.buffered())
    }

    /// Wrap the IO with a `BufStream` with the specified capacity.
    pub fn buffered_with_capacity(
        self,
        read: usize,
        write: usize,
    ) -> ConnectionBuilder<ConnectionService<BufStream<Io>>> {
        ConnectionBuilder::new(self.service.buffered_with_capacity(read, write))
    }
}

impl ConnectionBuilder<ConnectionService<BufStream<IoStream>>> {
    /// Connect to a remote address and create a `ConnectionBuilder`.
    pub async fn connect<A: ToSocketAddrs>(addr: A) -> Result<Self, CommError> {
        let stream = IoStream::connect(addr).await?;
        Ok(Self::new(ConnectionService::new(BufStream::new(stream))))
    }
}

/// A connection that can be customized with layers.
#[derive(Debug)]
pub struct Connection<S: Service> {
    service: S,
}

impl<S: Service> From<ConnectionBuilder<S>> for Connection<S> {
    fn from(builder: ConnectionBuilder<S>) -> Self {
        Connection {
            service: builder.service,
        }
    }
}

impl<S: Service> Connection<S> {
    /// Send a `Bytes` item through the connection.
    pub async fn send(&mut self, item: Bytes) -> Result<(), CommError> {
        self.service.send(item).await
    }

    /// Receive a `Bytes` item from the connection.
    pub async fn recv(&mut self) -> Result<Option<Bytes>, CommError> {
        self.service.recv().await
    }

    /// Wrap the connection with a layer.
    pub fn layer<L: Layer>(self, layer: L) -> Connection<L::Service<S>> {
        Connection {
            service: layer.layer(self.service),
        }
    }
}

/// A connection with a specific encoding and decoding type.
#[derive(Debug)]
pub struct TypedConnection<S: Service, Enc: Encode + Send, De: Decode<()> + Send> {
    conn: Connection<S>,
    _encode: std::marker::PhantomData<Enc>,
    _decode: std::marker::PhantomData<De>,
}

impl<S: Service, Enc: Encode + Send, De: Decode<()> + Send> From<ConnectionBuilder<S>>
    for TypedConnection<S, Enc, De>
{
    fn from(builder: ConnectionBuilder<S>) -> Self {
        TypedConnection {
            conn: builder.build(),
            _encode: std::marker::PhantomData,
            _decode: std::marker::PhantomData,
        }
    }
}

impl<Enc: Encode + Send, De: Decode<()> + Send>
    TypedConnection<ConnectionService<BufStream<IoStream>>, Enc, De>
{
    /// Connect to a remote address and create a `TypedConnection`.
    /// This function handles the transport negotiation automatically.
    pub async fn connect<A: ToSocketAddrs>(addr: A) -> Result<Self, CommError> {
        let stream = IoStream::connect(addr).await?;
        let service = ConnectionService::new(BufStream::new(stream));
        Ok(ConnectionBuilder::new(service).build())
    }

    /// Connects to a remote address using TCP only, bypassing transport negotiation.
    pub async fn connect_over_tcp<A: ToSocketAddrs>(addr: A) -> Result<Self, CommError> {
        let stream = IoStream::connect_tcp(addr).await?;
        let service = ConnectionService::new(BufStream::new(stream));
        Ok(ConnectionBuilder::new(service).build())
    }
}

impl<S: Service, Enc: Encode + Send, De: Decode<()> + Send + 'static> TypedConnection<S, Enc, De> {
    /// Wrap the connection with a layer.
    pub fn layer<L: Layer>(self, layer: L) -> TypedConnection<L::Service<S>, Enc, De> {
        TypedConnection {
            conn: self.conn.layer(layer),
            _encode: std::marker::PhantomData,
            _decode: std::marker::PhantomData,
        }
    }
}

impl<S: Service, Enc: Encode + Send + Sync, De: Decode<()> + Send + 'static> CommChannel
    for TypedConnection<S, Enc, De>
{
    type Enc = Enc;
    type De = De;

    async fn send(&mut self, item: &Self::Enc) -> Result<(), CommError> {
        let bytes =
            bincode::encode_to_vec(item, bincode::config::standard()).context(EncodeSnafu {})?;
        self.conn.send(Bytes::from(bytes)).await
    }

    async fn recv(&mut self) -> Result<Option<Self::De>, CommError> {
        match self.conn.recv().await? {
            Some(bytes) => {
                let (item, _): (Self::De, _) = tokio::task::spawn_blocking(move || {
                    bincode::decode_from_slice(bytes.as_ref(), bincode::config::standard())
                })
                .await
                .context(JoinSnafu {})?
                .context(DecodeSnafu {})?;
                Ok(Some(item))
            }
            None => Ok(None),
        }
    }
}

impl<S: Service, Enc: Encode + Send, De: Decode<()> + Send> From<Connection<S>>
    for TypedConnection<S, Enc, De>
{
    fn from(conn: Connection<S>) -> Self {
        TypedConnection {
            conn,
            _encode: std::marker::PhantomData,
            _decode: std::marker::PhantomData,
        }
    }
}
