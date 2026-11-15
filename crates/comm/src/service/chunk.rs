//! Chunking layer for splitting large messages into smaller pieces.

use std::marker::PhantomData;

use bincode::{Decode, Encode};
use bytes::{Bytes, BytesMut};
use snafu::ResultExt;
use tokio::task;
use tracing::trace;

use crate::{
    CommError, DecodeSnafu, EncodeSnafu, JoinSnafu,
    service::{Layer, Service},
};

/// The message chunk used for splitting large messages.
#[derive(Debug, Clone, Encode, Decode)]
pub(crate) enum Chunk {
    /// A chunk of data.
    Data(#[bincode(with_serde)] Bytes),
    /// The final chunk of data.
    End(#[bincode(with_serde)] Bytes),
}

const DEFAULT_CHUNK_SIZE: usize = 1024 * 1024; // 1MB

/// A layer that chunks large `Bytes` messages into smaller, manageable pieces.
///
/// This layer is essential for sending data that exceeds the capacity of the underlying
/// transport or when dealing with stream-based protocols where message boundaries are
/// not preserved. It wraps an inner `Service` and intercepts `send` and `recv` calls
/// to handle the chunking and reassembly process.
///
/// # Behavior
///
/// - **Sending**: When `send` is called with a large `Bytes` object, `ChunkLayer`
///   splits it into multiple smaller chunks. Each chunk is encoded into a `Chunk`
///   enum (`Chunk::Data` or `Chunk::End`) and sent individually to the inner service.
///
/// - **Receiving**: On the receiving end, this layer collects incoming `Chunk::Data`
///   parts and buffers them. When a `Chunk::End` chunk is received, it appends the
///   final data and returns the fully reassembled `Bytes` object.
///
/// # Ordering Guarantees
///
/// **Important**: The `ChunkLayer` itself does **not** guarantee the order of chunks.
/// It sends chunks sequentially, but their arrival order depends entirely on the
/// guarantees provided by the inner `Service`. For the data to be reassembled
/// correctly, the wrapped service **must** ensure that the chunks are delivered in the
/// same order they were sent. This is typically true for services built on reliable,
/// ordered protocols like TCP.
#[derive(Debug, Clone)]
pub struct ChunkLayer {
    chunk_size: usize,
}

impl Default for ChunkLayer {
    fn default() -> Self {
        Self {
            chunk_size: DEFAULT_CHUNK_SIZE,
        }
    }
}

impl ChunkLayer {
    /// Creates a new `ChunkLayer` with the specified chunk size.
    pub fn new(chunk_size: usize) -> Self {
        Self { chunk_size }
    }

    /// Sets the chunk size for this layer.
    pub fn chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }
}

impl Layer for ChunkLayer {
    type Service<S: Service> = ChunkService<S>;

    fn layer<S: Service>(&self, inner: S) -> Self::Service<S> {
        ChunkService {
            inner,
            chunk_size: self.chunk_size,
            buffer: BytesMut::new(),
            _marker: PhantomData,
        }
    }
}

/// A service that chunks messages into smaller pieces.
pub struct ChunkService<S: Service> {
    inner: S,
    chunk_size: usize,
    buffer: BytesMut,
    _marker: PhantomData<S>,
}

impl<S: Service> Service for ChunkService<S> {
    async fn send(&mut self, item: Bytes) -> Result<(), CommError> {
        if item.is_empty() {
            trace!("Sending empty message as a single End chunk.");
            let chunk_msg = Chunk::End(Bytes::new());
            let encoded = task::spawn_blocking(move || {
                bincode::encode_to_vec(chunk_msg, bincode::config::standard())
            })
            .await
            .context(JoinSnafu {})?
            .context(EncodeSnafu {})?;
            self.inner.send(Bytes::from(encoded)).await?;
            return Ok(());
        }

        let total_size = item.len();
        let num_chunks = total_size.div_ceil(self.chunk_size);
        trace!(
            total_size,
            num_chunks,
            chunk_size = self.chunk_size,
            "Sending message in chunks."
        );

        let mut chunks = item.chunks(self.chunk_size).peekable();
        let mut chunk_index = 0;
        while let Some(chunk) = chunks.next() {
            let is_last = chunks.peek().is_none();
            trace!(chunk_index, num_chunks, is_last, "Sending chunk.");
            let chunk_msg = if is_last {
                Chunk::End(Bytes::copy_from_slice(chunk))
            } else {
                Chunk::Data(Bytes::copy_from_slice(chunk))
            };
            let encoded = task::spawn_blocking(move || {
                bincode::encode_to_vec(chunk_msg, bincode::config::standard())
            })
            .await
            .context(JoinSnafu {})?
            .context(EncodeSnafu {})?;
            self.inner.send(Bytes::from(encoded)).await?;
            chunk_index += 1;
        }
        Ok(())
    }

    async fn recv(&mut self) -> Result<Option<Bytes>, CommError> {
        loop {
            if let Some(item) = self.inner.recv().await? {
                let (chunk, _): (Chunk, usize) = task::spawn_blocking(move || {
                    bincode::decode_from_slice(&item, bincode::config::standard())
                })
                .await
                .context(JoinSnafu {})?
                .context(DecodeSnafu {})?;
                match chunk {
                    Chunk::Data(data) => {
                        self.buffer.extend_from_slice(&data);
                        trace!(
                            bytes_received = data.len(),
                            total_buffered = self.buffer.len(),
                            "Received data chunk, appending to buffer."
                        );
                    }
                    Chunk::End(data) => {
                        self.buffer.extend_from_slice(&data);
                        let result = self.buffer.split().freeze();
                        trace!(
                            bytes_received = data.len(),
                            final_size = result.len(),
                            "Received end chunk, message reassembled."
                        );
                        return Ok(Some(result));
                    }
                }
            } else {
                return Ok(None);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use bytes::Bytes;
    use rand::RngCore;

    use super::*;
    use crate::CommError;

    struct MockService {
        sent_data: Arc<Mutex<Vec<Bytes>>>,
        recv_data: Arc<Mutex<Vec<Bytes>>>,
    }

    impl MockService {
        fn new(recv_data: Vec<Bytes>) -> Self {
            Self {
                sent_data: Arc::new(Mutex::new(Vec::new())),
                recv_data: Arc::new(Mutex::new(recv_data)),
            }
        }
    }

    impl Service for MockService {
        async fn send(&mut self, item: Bytes) -> Result<(), CommError> {
            self.sent_data.lock().unwrap().push(item);
            Ok(())
        }

        async fn recv(&mut self) -> Result<Option<Bytes>, CommError> {
            let mut guard = self.recv_data.lock().unwrap();
            if guard.is_empty() {
                return Ok(None);
            }
            Ok(Some(guard.remove(0)))
        }
    }

    #[tokio::test]
    async fn test_chunk_send_and_recv() {
        let mut data = vec![0u8; 1024 * 1024 * 2 + 100]; // 2MB + 100 bytes
        rand::rng().fill_bytes(&mut data);
        let data = Bytes::from(data);
        let chunk_size = 1024 * 1024; // 1MB

        let mock_service = MockService::new(vec![]);
        let sent_data = mock_service.sent_data.clone();
        let mut chunk_service = ChunkLayer::new(chunk_size).layer(mock_service);

        chunk_service.send(data.clone()).await.unwrap();

        let sent_chunks = sent_data.lock().unwrap().clone();
        let recv_mock_service = MockService::new(sent_chunks);
        let mut recv_chunk_service = ChunkLayer::new(chunk_size).layer(recv_mock_service);

        let received_data = recv_chunk_service.recv().await.unwrap().unwrap();
        assert_eq!(data, received_data);
    }

    #[tokio::test]
    async fn test_chunk_send_and_recv_small_chunk_size() {
        let mut data = vec![0u8; 1024]; // 1KB
        rand::rng().fill_bytes(&mut data);
        let data = Bytes::from(data);
        let chunk_size = 100;

        let mock_service = MockService::new(vec![]);
        let sent_data = mock_service.sent_data.clone();
        let mut chunk_service = ChunkLayer::new(chunk_size).layer(mock_service);

        chunk_service.send(data.clone()).await.unwrap();

        let sent_chunks = sent_data.lock().unwrap().clone();
        assert_eq!(sent_chunks.len(), 11); // 10 * 100 + 1 * 24

        let recv_mock_service = MockService::new(sent_chunks);
        let mut recv_chunk_service = ChunkLayer::new(chunk_size).layer(recv_mock_service);

        let received_data = recv_chunk_service.recv().await.unwrap().unwrap();
        assert_eq!(data, received_data);
    }

    #[tokio::test]
    async fn test_chunk_send_and_recv_empty() {
        let data = Bytes::new();
        let chunk_size = 1024;

        let mock_service = MockService::new(vec![]);
        let sent_data = mock_service.sent_data.clone();
        let mut chunk_service = ChunkLayer::new(chunk_size).layer(mock_service);

        chunk_service.send(data.clone()).await.unwrap();

        let sent_chunks = sent_data.lock().unwrap().clone();
        let recv_mock_service = MockService::new(sent_chunks);
        let mut recv_chunk_service = ChunkLayer::new(chunk_size).layer(recv_mock_service);

        let received_data = recv_chunk_service.recv().await.unwrap().unwrap();
        assert_eq!(data, received_data);
    }

    #[tokio::test]
    async fn test_chunk_send_and_recv_exact_size() {
        let chunk_size = 1024;
        let mut data = vec![0u8; chunk_size];
        rand::rng().fill_bytes(&mut data);
        let data = Bytes::from(data);

        let mock_service = MockService::new(vec![]);
        let sent_data = mock_service.sent_data.clone();
        let mut chunk_service = ChunkLayer::new(chunk_size).layer(mock_service);

        chunk_service.send(data.clone()).await.unwrap();

        let sent_chunks = sent_data.lock().unwrap().clone();
        assert_eq!(sent_chunks.len(), 1);

        let recv_mock_service = MockService::new(sent_chunks);
        let mut recv_chunk_service = ChunkLayer::new(chunk_size).layer(recv_mock_service);

        let received_data = recv_chunk_service.recv().await.unwrap().unwrap();
        assert_eq!(data, received_data);
    }

    #[tokio::test]
    async fn test_chunk_recv_none() {
        let chunk_size = 1024;
        let mock_service = MockService::new(vec![]);
        let mut chunk_service = ChunkLayer::new(chunk_size).layer(mock_service);

        let result = chunk_service.recv().await.unwrap();
        assert!(result.is_none());
    }
}
