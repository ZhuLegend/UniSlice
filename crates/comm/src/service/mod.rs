//! This module defines the `Service` trait for sending and receiving byte streams,

use bytes::Bytes;

use crate::CommError;

pub(crate) mod chunk;
mod connection;
mod retry;

pub use chunk::ChunkLayer;
pub use connection::ConnectionService;
pub use retry::{Backoff, RetryLayer, RetryPolicy};

/// A trait for services that can send and receive byte streams.
pub trait Service: Send + Sized {
    /// Sends a `Bytes` object.
    fn send(&mut self, item: Bytes) -> impl Future<Output = Result<(), CommError>> + Send;

    /// Receives a `Bytes` object.
    fn recv(&mut self) -> impl Future<Output = Result<Option<Bytes>, CommError>> + Send;
}

/// A trait for creating a layer that wraps a `Service`.
pub trait Layer {
    /// The service type produced by this layer.
    type Service<S: Service>: Service;

    /// Wraps a `Service` to create a new `Service`.
    fn layer<S: Service>(&self, inner: S) -> Self::Service<S>;
}
