//! HTTP客户端模块
//!
//! 负责与scheduler_daemon进行HTTP通信

use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use compute_node::{compute_node::r#virtual::virtual_compute_node::WorkTask, util::Input};
use reqwest::Client;
use tracing::debug;

use crate::types::HttpProtocol;

// 自定义错误类型
#[derive(Debug)]
pub struct HttpError(String);

impl HttpError {
    pub fn failure_reason(&self) -> &'static str {
        if self.0.contains("too_many_internal_resets") {
            "http2_reset_limit"
        } else if self.0.contains("http2 error") {
            "http2_connection_error"
        } else if self.0.contains("deadline exceeded") {
            "application_deadline"
        } else if self.0.contains("os error 49") {
            "address_unavailable"
        } else if self.0.contains("os error 54") {
            "connection_reset"
        } else if self.0.contains("No devices support") {
            "no_serving_pool"
        } else if self.0.starts_with("任务处理失败") || self.0.starts_with("任务提交失败")
        {
            "application_error"
        } else if self.0.contains("timed out") {
            "timeout"
        } else {
            "request_error"
        }
    }
}

impl std::fmt::Display for HttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(all(test, feature = "scheduler_daemon"))]
mod tests {
    use std::{collections::HashSet, net::SocketAddr, sync::Mutex};

    use axum::{
        Router,
        body::Bytes,
        extract::{Extension, State},
        routing::{get, post},
    };
    use compute_node::compute_node::r#virtual::BaseWork;

    use super::*;

    #[test]
    fn deadline_errors_are_separate_from_shared_connection_errors() {
        for (message, reason) in [
            (
                "任务处理失败: request deadline exceeded while waiting for pool service",
                "application_deadline",
            ),
            ("operation timed out", "timeout"),
            ("http2 error: too_many_internal_resets", "http2_reset_limit"),
            (
                "http2 error: connection error detected",
                "http2_connection_error",
            ),
        ] {
            assert_eq!(HttpError(message.into()).failure_reason(), reason);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn thousands_of_deadline_responses_keep_the_data_connections() {
        use axum::response::IntoResponse;
        let peers = Arc::new(Mutex::new(HashSet::<SocketAddr>::new()));
        let app = Router::new()
            .route("/health", get(|| async { "ok" }))
            .route(
                "/tasks",
                post(
                    |State(peers): State<Arc<Mutex<HashSet<SocketAddr>>>>,
                     Extension(peer): Extension<SocketAddr>,
                     body: Bytes| async move {
                        peers.lock().unwrap().insert(peer);
                        let (task, _): (WorkTask<Input>, _) =
                            bincode::decode_from_slice(&body, bincode::config::standard()).unwrap();
                        if task.model == "fast" {
                            ([("X-Task-Status", "success")], body).into_response()
                        } else {
                            let _ = tokio::time::timeout(
                                Duration::from_millis(50),
                                std::future::pending::<()>(),
                            )
                            .await;
                            (
                                [
                                    ("X-Task-Status", "failed"),
                                    (
                                        "X-Error-Message",
                                        "request deadline exceeded while waiting for pool service",
                                    ),
                                ],
                                Bytes::new(),
                            )
                                .into_response()
                        }
                    },
                ),
            )
            .with_state(peers.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let (close, closed) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(scheduler::daemon::http_server::serve(
            listener,
            app,
            65536,
            async {
                let _ = closed.await;
            },
        ));
        // The application owns the earlier deadline; the client waits for its
        // explicit response rather than resetting multiplexed requests first.
        let client = HttpClient::new(&url, Duration::from_secs(3), HttpProtocol::Http2, 4).unwrap();
        client.warm_data_connections().await.unwrap();
        let task = |name: &str| WorkTask {
            model: name.into(),
            work: BaseWork::<Input>::new(
                uuid::Uuid::new_v4(),
                ndarray::Array2::<f32>::zeros((1, 1)).into(),
            ),
        };
        let responses =
            futures::future::join_all((0..8000).map(|index| {
                client.submit_task(task(if index % 4 == 0 { "fast" } else { "slow" }))
            }))
            .await;
        assert_eq!(responses.iter().filter(|r| r.is_ok()).count(), 2000);
        let failures: Vec<_> = responses.iter().filter_map(|r| r.as_ref().err()).collect();
        assert_eq!(failures.len(), 6000);
        assert!(
            failures
                .iter()
                .all(|e| e.failure_reason() == "application_deadline")
        );
        for _ in 0..16 {
            client.submit_task(task("fast")).await.unwrap();
        }
        assert_eq!(
            peers.lock().unwrap().len(),
            4,
            "deadline responses must keep the original data connections"
        );
        client.health_check().await.unwrap();
        close.send(()).unwrap();
        server.await.unwrap().unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn http2_keeps_four_data_connections_under_thousands_of_waiting_requests() {
        let peers = Arc::new(Mutex::new(HashSet::<SocketAddr>::new()));
        let app = Router::new()
            .route("/health", get(|| async { "ok" }))
            .route(
                "/tasks",
                post(
                    |State(peers): State<Arc<Mutex<HashSet<SocketAddr>>>>,
                     Extension(peer): Extension<SocketAddr>,
                     body: Bytes| async move {
                        peers.lock().unwrap().insert(peer);
                        tokio::time::sleep(Duration::from_millis(500)).await;
                        ([("X-Task-Status", "success")], body)
                    },
                ),
            )
            .with_state(peers.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let (close, closed) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(scheduler::daemon::http_server::serve(
            listener,
            app,
            65536,
            async {
                let _ = closed.await;
            },
        ));
        let client =
            HttpClient::new(&url, Duration::from_secs(30), HttpProtocol::Http2, 4).unwrap();
        client.warm_data_connections().await.unwrap();
        let sending = {
            let client = client.clone();
            tokio::spawn(async move {
                let responses = futures::future::join_all((0..2000).map(|_| {
                    let client = client.clone();
                    async move {
                        let task = WorkTask {
                            model: "m".into(),
                            work: BaseWork::<Input>::new(
                                uuid::Uuid::new_v4(),
                                ndarray::Array2::<f32>::zeros((1, 1)).into(),
                            ),
                        };
                        client.submit_task(task).await
                    }
                }))
                .await;
                assert!(responses.iter().all(Result::is_ok));
            })
        };
        tokio::time::sleep(Duration::from_millis(100)).await;
        tokio::time::timeout(Duration::from_secs(1), client.health_check())
            .await
            .unwrap()
            .unwrap();
        sending.await.unwrap();
        assert_eq!(
            peers.lock().unwrap().len(),
            4,
            "in-flight tasks must multiplex on the prewarmed data connections"
        );
        close.send(()).unwrap();
        server.await.unwrap().unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "wall clock ingress acceptance: 3200/4000 requests/s with 5s responses"]
    async fn ingress_acceptance_at_3200_and_4000_requests_per_second() {
        use std::sync::atomic::{AtomicU64, Ordering};
        let mut reports = Vec::new();
        for rate in [3200u64, 4000] {
            let peers = Arc::new(Mutex::new(HashSet::<SocketAddr>::new()));
            let received = Arc::new(AtomicU64::new(0));
            let pending = Arc::new(AtomicU64::new(0));
            let peak = Arc::new(AtomicU64::new(0));
            let app = Router::new()
                .route("/health", get(|| async { "ok" }))
                .route(
                    "/tasks",
                    post({
                        let (peers, received, pending, peak) = (
                            peers.clone(),
                            received.clone(),
                            pending.clone(),
                            peak.clone(),
                        );
                        move |Extension(peer): Extension<SocketAddr>, body: Bytes| {
                            let (peers, received, pending, peak) = (
                                peers.clone(),
                                received.clone(),
                                pending.clone(),
                                peak.clone(),
                            );
                            async move {
                                peers.lock().unwrap().insert(peer);
                                received.fetch_add(1, Ordering::Relaxed);
                                let active = pending.fetch_add(1, Ordering::Relaxed) + 1;
                                peak.fetch_max(active, Ordering::Relaxed);
                                tokio::time::sleep(Duration::from_secs(5)).await;
                                pending.fetch_sub(1, Ordering::Relaxed);
                                ([("X-Task-Status", "success")], body)
                            }
                        }
                    }),
                );
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let client = HttpClient::new(
                &format!("http://{}", listener.local_addr().unwrap()),
                Duration::from_secs(30),
                HttpProtocol::Http2,
                4,
            )
            .unwrap();
            let (close, closed) = tokio::sync::oneshot::channel();
            let server = tokio::spawn(scheduler::daemon::http_server::serve(
                listener,
                app,
                65536,
                async {
                    let _ = closed.await;
                },
            ));
            client.warm_data_connections().await.unwrap();
            let start = tokio::time::Instant::now();
            let mut tasks = tokio::task::JoinSet::new();
            let mut max_lag_ms = 0f64;
            let expected = rate * 6;
            for index in 0..expected {
                let due = start + Duration::from_secs_f64(index as f64 / rate as f64);
                tokio::time::sleep_until(due).await;
                max_lag_ms = max_lag_ms.max(due.elapsed().as_secs_f64() * 1000.);
                let client = client.clone();
                tasks.spawn(async move {
                    client
                        .submit_task(WorkTask {
                            model: "probe".into(),
                            work: BaseWork::<Input>::new(
                                uuid::Uuid::new_v4(),
                                ndarray::Array2::<f32>::zeros((1, 1)).into(),
                            ),
                        })
                        .await
                });
            }
            let mut failed = 0;
            while let Some(result) = tasks.join_next().await {
                if !matches!(result, Ok(Ok(_))) {
                    failed += 1;
                }
            }
            let report = serde_json::json!({"offered_rate": rate, "sending_seconds": 6, "response_delay_seconds": 5,
                "offered": expected, "received": received.load(Ordering::Relaxed), "failed": failed,
                "data_connections": peers.lock().unwrap().len(), "peak_pending": peak.load(Ordering::Relaxed),
                "max_replay_lag_ms": max_lag_ms, "total_seconds": start.elapsed().as_secs_f64()});
            println!("{report}");
            reports.push(report);
            assert_eq!(received.load(Ordering::Relaxed), expected);
            assert_eq!(failed, 0);
            assert_eq!(peers.lock().unwrap().len(), 4);
            assert!(
                max_lag_ms < 500.,
                "the arrival generator must not be silently throttled"
            );
            assert!(peak.load(Ordering::Relaxed) > rate * 4);
            close.send(()).unwrap();
            server.await.unwrap().unwrap();
        }
        if let Ok(path) = std::env::var("HSGS_INGRESS_REPORT") {
            std::fs::write(path, serde_json::to_vec_pretty(&reports).unwrap()).unwrap();
        }
    }
}

impl std::error::Error for HttpError {}

impl From<reqwest::Error> for HttpError {
    fn from(err: reqwest::Error) -> Self {
        HttpError(err.to_string())
    }
}

impl From<String> for HttpError {
    fn from(msg: String) -> Self {
        HttpError(msg)
    }
}

/// HTTP客户端
#[derive(Debug, Clone)]
pub struct HttpClient {
    // Control traffic has an independent connection and a short deadline.
    client: Client,
    data_clients: Vec<Client>,
    next_connection: Arc<AtomicUsize>,
    protocol: HttpProtocol,
    base_url: String,
}

impl HttpClient {
    /// 创建新的HTTP客户端
    pub fn new(
        daemon_url: &str,
        request_timeout: Duration,
        protocol: HttpProtocol,
        connections: usize,
    ) -> Result<Self, HttpError> {
        if connections == 0 {
            return Err("HTTP connection count must be positive".to_owned().into());
        }
        let make_builder = |timeout| {
            let mut builder = Client::builder().timeout(timeout);
            // Local simulation/control traffic must not inherit a desktop HTTP proxy.
            // Keep configured proxy behavior for remote daemon endpoints.
            if reqwest::Url::parse(daemon_url)
                .ok()
                .and_then(|url| {
                    url.host_str().map(|host| {
                        host == "localhost"
                            || host
                                .trim_matches(['[', ']'])
                                .parse::<std::net::IpAddr>()
                                .is_ok_and(|ip| ip.is_loopback())
                    })
                })
                .unwrap_or(false)
            {
                builder = builder.no_proxy();
            }
            builder
        };
        let client = make_builder(Duration::from_secs(3)).build()?;
        let mut data_clients = Vec::with_capacity(connections);
        for _ in 0..connections {
            let mut builder = make_builder(request_timeout).pool_max_idle_per_host(1);
            if protocol == HttpProtocol::Http2 {
                builder = builder.http2_prior_knowledge();
            }
            data_clients.push(builder.build()?);
        }

        let base_url = daemon_url.trim_end_matches('/').to_string();

        Ok(Self {
            client,
            data_clients,
            protocol,
            next_connection: Arc::new(AtomicUsize::new(0)),
            base_url,
        })
    }

    pub async fn warm_data_connections(&self) -> Result<(), HttpError> {
        for client in &self.data_clients {
            let response = client
                .get(format!("{}/health", self.base_url))
                .send()
                .await?;
            if !response.status().is_success() {
                return Err("data connection health check failed".to_owned().into());
            }
            if self.protocol == HttpProtocol::Http2
                && response.version() != reqwest::Version::HTTP_2
            {
                return Err("HTTP/2 requested but peer did not use HTTP/2"
                    .to_owned()
                    .into());
            }
            response.bytes().await?;
        }
        tracing::info!(protocol = ?self.protocol, connections = self.data_clients.len(), "Data connections ready");
        Ok(())
    }

    /// 健康检查
    pub async fn health_check(&self) -> Result<(), HttpError> {
        let url = format!("{}/health", self.base_url);

        debug!("发送健康检查请求: {}", url);

        let response = self.client.get(&url).send().await?;

        if response.status().is_success() {
            response.bytes().await?;
            debug!("健康检查成功");
            Ok(())
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            Err(HttpError(format!("健康检查失败: {} - {}", status, body)))
        }
    }

    /// 提交任务（使用Bincode序列化）
    pub async fn submit_task(
        &self,
        task: WorkTask<Input>,
    ) -> Result<
        compute_node::compute_node::r#virtual::virtual_compute_node::WorkTask<
            compute_node::util::Output,
        >,
        HttpError,
    > {
        let url = format!("{}/tasks", self.base_url);

        debug!("提交Bincode任务到: {}", url);

        // 序列化为Bincode
        let config = bincode::config::standard();
        let serialized = bincode::encode_to_vec(&task, config)
            .map_err(|e| HttpError(format!("Bincode序列化失败: {}", e)))?;

        let index = self.next_connection.fetch_add(1, Ordering::Relaxed) % self.data_clients.len();
        let response = self.data_clients[index]
            .post(&url)
            .header("Content-Type", "application/octet-stream")
            .body(serialized)
            .send()
            .await?;

        if self.protocol == HttpProtocol::Http2 && response.version() != reqwest::Version::HTTP_2 {
            return Err("task response used an unexpected HTTP protocol"
                .to_owned()
                .into());
        }

        if response.status().is_success() {
            // 从头部获取任务ID和状态
            let task_id = response
                .headers()
                .get("X-Task-ID")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("unknown")
                .to_string();

            let task_status = response
                .headers()
                .get("X-Task-Status")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("unknown");

            debug!("任务提交成功: {} (状态: {})", task_id, task_status);

            if task_status == "success" {
                // 反序列化Bincode响应体
                let body_bytes = response.bytes().await?;
                let config = bincode::config::standard();
                let (result_task, _) = bincode::decode_from_slice(&body_bytes, config)
                    .map_err(|e| HttpError(format!("响应反序列化失败: {}", e)))?;
                Ok(result_task)
            } else {
                // 任务处理失败
                let error_msg = response
                    .headers()
                    .get("X-Error-Message")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("未知错误")
                    .to_owned();
                response.bytes().await?;
                Err(HttpError(format!("任务处理失败: {}", error_msg)))
            }
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            crate::error_display::print_error(&format!("任务提交失败: {} - {}", status, body));
            Err(HttpError(format!("任务提交失败: {} - {}", status, body)))
        }
    }

    /// 获取daemon基本统计信息
    pub async fn get_daemon_stats(&self) -> Result<serde_json::Value, HttpError> {
        let url = format!("{}/stats", self.base_url);

        debug!("获取daemon统计信息: {}", url);

        let response = self.client.get(&url).send().await?;

        if response.status().is_success() {
            let stats: serde_json::Value = response.json().await?;
            debug!("成功获取daemon统计信息");
            Ok(stats)
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            Err(HttpError(format!(
                "获取daemon统计信息失败: {} - {}",
                status, body
            )))
        }
    }

    /// 获取daemon预测统计信息
    pub async fn get_prediction_stats(&self) -> Result<serde_json::Value, HttpError> {
        let url = format!("{}/predictions", self.base_url);

        debug!("获取daemon预测统计信息: {}", url);

        let response = self.client.get(&url).send().await?;

        if response.status().is_success() {
            let stats: serde_json::Value = response.json().await?;
            debug!("成功获取daemon预测统计信息");
            Ok(stats)
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            Err(HttpError(format!(
                "获取daemon预测统计信息失败: {} - {}",
                status, body
            )))
        }
    }

    /// 获取daemon调度器统计信息
    pub async fn get_scheduler_stats(&self) -> Result<serde_json::Value, HttpError> {
        let url = format!("{}/scheduler/stats", self.base_url);

        debug!("获取daemon调度器统计信息: {}", url);

        let response = self.client.get(&url).send().await?;

        if response.status().is_success() {
            let stats: serde_json::Value = response.json().await?;
            debug!("成功获取daemon调度器统计信息");
            Ok(stats)
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            Err(HttpError(format!(
                "获取daemon调度器统计信息失败: {} - {}",
                status, body
            )))
        }
    }
}
