//! HTTP/1 compatibility and explicitly budgeted HTTP/2 multiplexing.
use std::{future::Future, time::Duration};

use axum::{Router, body::Body};
use hyper::service::service_fn;
use hyper_util::{
    rt::{TokioExecutor, TokioIo},
    server::conn::auto::Builder,
};
use tokio::{net::TcpListener, sync::watch, task::JoinSet};
use tower::ServiceExt;

/// Serve existing Axum routes over HTTP/1 or HTTP/2 with a configurable stream budget.
pub async fn serve(
    listener: TcpListener,
    app: Router,
    max_streams: u32,
    shutdown: impl Future<Output = ()>,
) -> std::io::Result<()> {
    if max_streams == 0 {
        return Err(std::io::Error::other(
            "HTTP/2 stream budget must be positive",
        ));
    }
    let (closing, closed) = watch::channel(false);
    let mut connections = JoinSet::new();
    tokio::pin!(shutdown);
    loop {
        tokio::select! {
            _ = &mut shutdown => break,
            Some(result) = connections.join_next(), if !connections.is_empty() => {
                if let Err(error) = result { tracing::warn!(%error, "HTTP connection task failed"); }
            }
            accepted = listener.accept() => {
                let (stream, peer) = accepted?;
                tracing::info!(target: "http_transport", event = "connection_opened", %peer);
                let app = app.clone();
                let mut closed = closed.clone();
                connections.spawn(async move {
                    let mut builder = Builder::new(TokioExecutor::new());
                    builder.http2().max_concurrent_streams(max_streams).adaptive_window(true);
                    let service = service_fn(move |request| {
                        let mut request = request.map(Body::new);
                        request.extensions_mut().insert(peer);
                        app.clone().oneshot(request)
                    });
                    let connection = builder.serve_connection_with_upgrades(TokioIo::new(stream), service);
                    tokio::pin!(connection);
                    tokio::select! {
                        result = &mut connection => {
                            if let Err(error) = result { tracing::debug!(%peer, %error, "HTTP connection closed"); }
                        }
                        _ = async { let _ = closed.wait_for(|closing| *closing).await; } => {
                            connection.as_mut().graceful_shutdown();
                            let _ = connection.await;
                        }
                    }
                });
            }
        }
    }
    closing.send_replace(true);
    if tokio::time::timeout(Duration::from_secs(10), async {
        while connections.join_next().await.is_some() {}
    })
    .await
    .is_err()
    {
        connections.abort_all();
        while connections.join_next().await.is_some() {}
        tracing::warn!("HTTP shutdown deadline reached; unfinished connections closed");
    }
    Ok(())
}
