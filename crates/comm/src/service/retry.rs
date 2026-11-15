//! Retry layer for services.

use std::time::Duration;

use bytes::Bytes;
use rand::Rng;
use tokio::time;
use tracing::warn;

use crate::{
    CommError,
    service::{Layer, Service},
};

/// The backoff strategy for retries.
#[derive(Debug, Clone, Copy)]
pub enum Backoff {
    /// Fixed backoff.
    Fixed(Duration),
    /// Exponential backoff.
    Exponential(Duration),
}

/// The retry policy for a service.
#[derive(Debug, Clone, Copy)]
pub struct RetryPolicy {
    max_retries: usize,
    backoff: Backoff,
    jitter: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            backoff: Backoff::Exponential(Duration::from_millis(100)),
            jitter: true,
        }
    }
}

impl RetryPolicy {
    /// Creates a new `RetryPolicy`.
    pub fn new(max_retries: usize, backoff: Backoff, jitter: bool) -> Self {
        Self {
            max_retries,
            backoff,
            jitter,
        }
    }
}

/// A layer that adds retry logic to a service.
#[derive(Debug, Clone, Copy)]
pub struct RetryLayer {
    policy: RetryPolicy,
}

impl RetryLayer {
    /// Creates a new `RetryLayer` with the given policy.
    pub fn new(policy: RetryPolicy) -> Self {
        Self { policy }
    }
}

impl Layer for RetryLayer {
    type Service<S: Service> = RetryService<S>;

    fn layer<S: Service>(&self, inner: S) -> Self::Service<S> {
        RetryService {
            inner,
            policy: self.policy,
        }
    }
}

/// A service that adds retry logic to an inner service.
#[derive(Debug, Clone)]
pub struct RetryService<S: Service> {
    inner: S,
    policy: RetryPolicy,
}

impl<S: Service> RetryService<S> {
    async fn retry<F, T>(&mut self, mut f: F) -> Result<T, CommError>
    where
        F: AsyncFnMut(&mut S) -> Result<T, CommError>,
    {
        let mut retries = 0;
        loop {
            match f(&mut self.inner).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    if retries >= self.policy.max_retries {
                        warn!(error = ?e, "Operation failed after {} retries.", self.policy.max_retries);
                        return Err(e);
                    }
                    if !matches!(e, CommError::Io { .. }) {
                        return Err(e);
                    }
                    let backoff = match self.policy.backoff {
                        Backoff::Fixed(duration) => duration,
                        Backoff::Exponential(duration) => duration * 2_u32.pow(retries as u32),
                    };
                    let backoff = if self.policy.jitter {
                        let jitter = rand::rng()
                            .random_range(Duration::from_secs(0)..Duration::from_millis(100));
                        backoff + jitter
                    } else {
                        backoff
                    };

                    warn!(
                        error = ?e,
                        retry_attempt = retries + 1,
                        max_retries = self.policy.max_retries,
                        backoff = ?backoff,
                        "Operation failed, retrying..."
                    );

                    time::sleep(backoff).await;
                    retries += 1;
                }
            }
        }
    }
}

impl<S: Service> Service for RetryService<S> {
    async fn send(&mut self, item: Bytes) -> Result<(), CommError> {
        self.retry(async move |s| s.send(item.clone()).await).await
    }

    async fn recv(&mut self) -> Result<Option<Bytes>, CommError> {
        self.retry(async |s| s.recv().await).await
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io,
        sync::{
            Arc, Mutex,
            atomic::{AtomicUsize, Ordering},
        },
    };

    use bytes::Bytes;

    use super::*;
    use crate::CommError;

    struct MockService {
        send_errors: Arc<Mutex<Vec<Option<CommError>>>>,
        recv_errors: Arc<Mutex<Vec<Option<CommError>>>>,
        send_calls: Arc<AtomicUsize>,
        recv_calls: Arc<AtomicUsize>,
    }

    impl MockService {
        fn new(send_errors: Vec<Option<CommError>>, recv_errors: Vec<Option<CommError>>) -> Self {
            Self {
                send_errors: Arc::new(Mutex::new(send_errors)),
                recv_errors: Arc::new(Mutex::new(recv_errors)),
                send_calls: Arc::new(AtomicUsize::new(0)),
                recv_calls: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    impl Service for MockService {
        async fn send(&mut self, _item: Bytes) -> Result<(), CommError> {
            self.send_calls.fetch_add(1, Ordering::SeqCst);
            let mut guard = self.send_errors.lock().unwrap();
            if guard.is_empty() {
                return Ok(());
            }
            if let Some(err) = guard.remove(0) {
                return Err(err);
            }
            Ok(())
        }

        async fn recv(&mut self) -> Result<Option<Bytes>, CommError> {
            self.recv_calls.fetch_add(1, Ordering::SeqCst);
            let mut guard = self.recv_errors.lock().unwrap();
            if guard.is_empty() {
                return Ok(Some(Bytes::new()));
            }
            if let Some(err) = guard.remove(0) {
                return Err(err);
            }
            Ok(Some(Bytes::new()))
        }
    }

    #[tokio::test]
    async fn test_retry_send_ok() {
        let mock_service = MockService::new(
            vec![
                Some(CommError::Io {
                    source: io::Error::other("test"),
                }),
                None,
            ],
            vec![],
        );
        let send_calls = mock_service.send_calls.clone();
        let mut retry_service = RetryLayer::new(Default::default()).layer(mock_service);

        retry_service.send(Bytes::new()).await.unwrap();
        assert_eq!(send_calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_retry_send_fail() {
        let mock_service = MockService::new(
            vec![
                Some(CommError::Io {
                    source: io::Error::other("test"),
                }),
                Some(CommError::Io {
                    source: io::Error::other("test"),
                }),
                Some(CommError::Io {
                    source: io::Error::other("test"),
                }),
                Some(CommError::Io {
                    source: io::Error::other("test"),
                }),
            ],
            vec![],
        );
        let send_calls = mock_service.send_calls.clone();
        let mut retry_service = RetryLayer::new(Default::default()).layer(mock_service);

        let result = retry_service.send(Bytes::new()).await;
        assert!(result.is_err());
        assert_eq!(send_calls.load(Ordering::SeqCst), 4);
    }

    #[tokio::test]
    async fn test_retry_recv_ok() {
        let mock_service = MockService::new(
            vec![],
            vec![
                Some(CommError::Io {
                    source: io::Error::other("test"),
                }),
                None,
            ],
        );
        let recv_calls = mock_service.recv_calls.clone();
        let mut retry_service = RetryLayer::new(Default::default()).layer(mock_service);

        retry_service.recv().await.unwrap();
        assert_eq!(recv_calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_retry_recv_fail() {
        let mock_service = MockService::new(
            vec![],
            vec![
                Some(CommError::Io {
                    source: io::Error::other("test"),
                }),
                Some(CommError::Io {
                    source: io::Error::other("test"),
                }),
                Some(CommError::Io {
                    source: io::Error::other("test"),
                }),
                Some(CommError::Io {
                    source: io::Error::other("test"),
                }),
            ],
        );
        let recv_calls = mock_service.recv_calls.clone();
        let mut retry_service = RetryLayer::new(Default::default()).layer(mock_service);

        let result = retry_service.recv().await;
        assert!(result.is_err());
        assert_eq!(recv_calls.load(Ordering::SeqCst), 4);
    }

    #[tokio::test]
    async fn test_no_retry_for_non_io_error() {
        let mock_service = MockService::new(vec![Some(CommError::Timeout)], vec![]);
        let send_calls = mock_service.send_calls.clone();
        let mut retry_service = RetryLayer::new(Default::default()).layer(mock_service);

        let result = retry_service.send(Bytes::new()).await;
        assert!(result.is_err());
        assert!(matches!(result, Err(CommError::Timeout)));
        assert_eq!(send_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_retry_fixed_backoff() {
        let mock_service = MockService::new(
            vec![
                Some(CommError::Io {
                    source: io::Error::other("test"),
                }),
                None,
            ],
            vec![],
        );
        let send_calls = mock_service.send_calls.clone();
        let policy = RetryPolicy::new(3, Backoff::Fixed(Duration::from_millis(10)), false);
        let mut retry_service = RetryLayer::new(policy).layer(mock_service);

        retry_service.send(Bytes::new()).await.unwrap();
        assert_eq!(send_calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_max_retries_zero() {
        let mock_service = MockService::new(
            vec![Some(CommError::Io {
                source: io::Error::other("test"),
            })],
            vec![],
        );
        let send_calls = mock_service.send_calls.clone();
        let policy = RetryPolicy::new(0, Backoff::Fixed(Duration::from_millis(10)), false);
        let mut retry_service = RetryLayer::new(policy).layer(mock_service);

        let result = retry_service.send(Bytes::new()).await;
        assert!(result.is_err());
        assert_eq!(send_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_no_jitter() {
        let mock_service = MockService::new(
            vec![
                Some(CommError::Io {
                    source: io::Error::other("test"),
                }),
                None,
            ],
            vec![],
        );
        let send_calls = mock_service.send_calls.clone();
        let policy = RetryPolicy::new(3, Backoff::Exponential(Duration::from_millis(10)), false);
        let mut retry_service = RetryLayer::new(policy).layer(mock_service);

        let now = tokio::time::Instant::now();
        retry_service.send(Bytes::new()).await.unwrap();
        let elapsed = now.elapsed();

        assert_eq!(send_calls.load(Ordering::SeqCst), 2);
        // Without jitter, the backoff should be exactly 10ms.
        // We allow a small delta for timer precision.
        assert!(elapsed >= Duration::from_millis(10));
        assert!(elapsed < Duration::from_millis(20));
    }
}
