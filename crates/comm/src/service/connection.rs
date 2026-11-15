use bytes::Bytes;
use futures::{SinkExt, StreamExt};
use snafu::ResultExt;
use tokio::io::{AsyncRead, AsyncWrite, BufStream};
use tokio_util::codec::{Framed, LengthDelimitedCodec};
use tracing::trace;

use crate::{CommError, IoSnafu, service::Service};

/// The base service that directly interacts with the `Framed` transport.
pub struct ConnectionService<T: AsyncRead + AsyncWrite + Unpin + Send> {
    transport: Framed<T, LengthDelimitedCodec>,
}

impl<T: AsyncRead + AsyncWrite + Unpin + Send> ConnectionService<T> {
    /// Creates a new `ConnectionService` with the given IO type.
    pub fn new(io: T) -> Self {
        Self {
            transport: Framed::new(io, LengthDelimitedCodec::new()),
        }
    }

    /// set the maximum frame length for the codec.
    pub fn with_max_frame_length(mut self, max_frame_length: usize) -> Self {
        self.transport
            .codec_mut()
            .set_max_frame_length(max_frame_length);
        self
    }

    /// Wrap the IO with a `BufStream`.
    pub fn buffered(self) -> ConnectionService<BufStream<T>> {
        let codec = self.transport.codec().clone();
        let io = self.transport.into_inner();

        ConnectionService {
            transport: Framed::new(BufStream::new(io), codec),
        }
    }

    /// Wrap the IO with a `BufStream` with the specified capacity.
    pub fn buffered_with_capacity(
        self,
        read: usize,
        write: usize,
    ) -> ConnectionService<BufStream<T>> {
        let codec = self.transport.codec().clone();
        let io = self.transport.into_inner();

        ConnectionService {
            transport: Framed::new(BufStream::with_capacity(read, write, io), codec),
        }
    }
}

impl<T: AsyncRead + AsyncWrite + Unpin + Send> Service for ConnectionService<T> {
    async fn send(&mut self, item: Bytes) -> Result<(), CommError> {
        trace!(
            "sending {} bytes, max bytes: {}",
            item.len(),
            self.transport.codec().max_frame_length()
        );
        self.transport.send(item).await.context(IoSnafu {})
    }

    async fn recv(&mut self) -> Result<Option<Bytes>, CommError> {
        match self.transport.next().await {
            Some(Ok(bytes)) => {
                trace!("receiving {} bytes", bytes.len());
                Ok(Some(bytes.freeze()))
            }
            Some(Err(e)) => Err(CommError::Io { source: e }),
            None => Ok(None),
        }
    }
}
