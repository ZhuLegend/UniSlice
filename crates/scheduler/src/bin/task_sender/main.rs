//! 任务发送器主入口点
//!
//! 用于向scheduler_daemon发送任务并进行性能监控
//! 支持按泊松分布发送任务，实时统计发送速率和处理时延

mod config;
mod error_display;
mod http_client;
mod live_display;
mod metrics_tracker;
mod poisson_sender;
mod stats_collector;
mod types;

use std::{sync::Arc, time::Duration};

use clap::Parser;
use tokio::signal;
use tracing::{debug, error, info};

use crate::{
    config::{load_config, validate_config},
    http_client::HttpClient,
    live_display::LiveDisplay,
    metrics_tracker::MetricsTracker,
    poisson_sender::{PoissonSender, replay_request_id},
    stats_collector::StatsCollector,
    types::Args,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 初始化日志
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    info!("启动任务发送器");
    info!("配置文件: {:?}", args.config);
    // Merge CLI overrides before validation and constructing the client.
    let mut config = load_config(&args.config).await?;
    if let Some(url) = args.daemon_url {
        config.sender.daemon_url = url;
    }
    info!("目标daemon地址: {}", config.sender.daemon_url);
    validate_config(&config).map_err(|e| format!("配置验证失败: {}", e))?;

    info!("配置加载成功");
    info!("监控模型数量: {}", config.models.len());
    info!("状态收集间隔: {}秒", config.sender.stats_interval);

    // 创建核心组件
    let http_client = Arc::new(HttpClient::new(
        &config.sender.daemon_url,
        Duration::from_secs(args.request_timeout_secs),
        args.http_protocol,
        args.http_connections,
    )?);
    let mut metrics_tracker = MetricsTracker::new(config.metrics.clone());
    if let Some(path) = &args.terminal_log {
        metrics_tracker.enable_terminal_log(path).await?;
    }
    let metrics_tracker = Arc::new(metrics_tracker);

    // 测试daemon连接
    if let Err(e) = http_client.health_check().await {
        error!("无法连接到scheduler daemon: {}", e);
        return Err(format!("Daemon连接失败: {}", e).into());
    }
    info!("Daemon连接测试成功");
    http_client.warm_data_connections().await?;
    if args.max_in_flight == 0 {
        return Err("max-in-flight must be positive".into());
    }

    // 创建任务发送器
    let poisson_sender = Arc::new(PoissonSender::new(
        config.clone(),
        http_client.clone(),
        metrics_tracker.clone(),
        args.send_batch_size_only,
        args.max_in_flight,
    ));

    // 创建状态收集器
    let mut stats_collector = StatsCollector::new(
        metrics_tracker.clone(),
        http_client.clone(),
        &config.sender.log_dir,
    )?;

    if let Some(seconds) = args.duration_secs {
        if seconds == 0 || args.request_timeout_secs == 0 {
            return Err("duration and request timeout must be positive".into());
        }
        let replay: Option<Vec<(f64, String, usize)>> = if let Some(path) = &args.replay {
            let rows: Vec<(f64, String, usize)> =
                serde_json::from_slice(&tokio::fs::read(path).await?)?;
            let mut last = 0.0;
            for (offset, name, batch) in &rows {
                let model = config
                    .models
                    .iter()
                    .find(|m| m.name == *name)
                    .ok_or("unknown replay model")?;
                if !offset.is_finite()
                    || *offset < last
                    || *offset >= seconds as f64
                    || *batch < model.batch_size_range[0] as usize
                    || *batch > model.batch_size_range[1] as usize
                {
                    return Err("invalid or unsorted replay event".into());
                }
                last = *offset;
            }
            Some(rows)
        } else {
            None
        };
        let epoch_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_millis();
        let started = tokio::time::Instant::now();
        let planned_origin_ms = metrics_tracker.unix_ms(started.into_std());
        let planned_requests = replay.as_ref().map(|rows| rows.len());
        let progress_handle = {
            let tracker = metrics_tracker.clone();
            let path = std::path::Path::new(&config.sender.log_dir).join("progress.json");
            tokio::spawn(async move {
                let mut tick = tokio::time::interval(Duration::from_secs(1));
                loop {
                    tick.tick().await;
                    tracker.cleanup_expired_tasks().await;
                    let mut snapshot = client_snapshot(
                        &tracker,
                        epoch_ms,
                        seconds,
                        started.elapsed().as_secs_f64(),
                    )
                    .await;
                    snapshot["planned_requests"] = serde_json::json!(planned_requests);
                    if let Err(error) = write_snapshot(&path, &snapshot).await {
                        tracing::error!(%error, "could not persist client progress");
                    }
                }
            })
        };
        let mut max_send_lag_ms: f64 = 0.0;
        let mut lag_total_ms = 0.0;
        let mut replay_count = 0usize;
        let deadline = started + Duration::from_secs(seconds);
        if let Some(rows) = replay {
            for (ordinal, (offset, model, batch)) in rows.into_iter().enumerate() {
                let due = started + Duration::from_secs_f64(offset);
                tokio::time::sleep_until(due).await;
                let lag = due.elapsed().as_secs_f64() * 1000.0;
                max_send_lag_ms = max_send_lag_ms.max(lag);
                lag_total_ms += lag;
                replay_count += 1;
                let model = config.models.iter().find(|m| m.name == model).unwrap();
                let task_id = args
                    .replay_request_id_seed
                    .map(|seed| replay_request_id(seed, ordinal as u64));
                poisson_sender
                    .generate_and_send_task_at(model, batch, due.into_std(), task_id)
                    .await
                    .map_err(|e| e.to_string())?;
            }
            tokio::time::sleep_until(deadline).await;
        } else {
            while tokio::time::Instant::now() < deadline {
                for model in &config.models {
                    poisson_sender
                        .update_model_rate(&model.name, &model.rate_range)
                        .await;
                }
                let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
                poisson_sender
                    .run_cycle(
                        config.models.clone(),
                        remaining.min(Duration::from_secs(config.sender.cycle_duration)),
                    )
                    .await;
                stats_collector.collect_and_save().await?;
            }
        }
        let drain = tokio::time::Instant::now() + Duration::from_secs(args.drain_timeout_secs);
        while metrics_tracker.get_sample_counts().await.2 > 0 && tokio::time::Instant::now() < drain
        {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        let pending_at_drain = metrics_tracker.get_sample_counts().await.2;
        // Save before cancellation and before any remote statistics request.
        let final_path = std::path::Path::new(&config.sender.log_dir).join("final_summary.json");
        let mut summary = client_snapshot(
            &metrics_tracker,
            epoch_ms,
            seconds,
            started.elapsed().as_secs_f64(),
        )
        .await;
        summary["planned_requests"] = serde_json::json!(planned_requests);
        summary["planned_origin_ms"] = serde_json::json!(planned_origin_ms);
        summary["replay_requests"] = serde_json::json!(replay_count);
        summary["replay_request_id_seed"] = serde_json::json!(args.replay_request_id_seed);
        summary["request_id_scheme"] =
            serde_json::json!(if args.replay_request_id_seed.is_some() {
                "explicit_namespace_u64_and_original_ordinal_u64_v1"
            } else {
                "random_uuid_v4"
            });
        summary["max_send_lag_ms"] = serde_json::json!(max_send_lag_ms);
        summary["mean_send_lag_ms"] = serde_json::json!(if replay_count > 0 {
            lag_total_ms / replay_count as f64
        } else {
            0.0
        });
        summary["pending_at_drain"] = serde_json::json!(pending_at_drain);
        summary["http_protocol"] = serde_json::json!(format!("{:?}", args.http_protocol));
        summary["http_connections"] = serde_json::json!(args.http_connections);
        summary["max_in_flight"] = serde_json::json!(args.max_in_flight);
        write_snapshot(&final_path, &summary).await?;
        if pending_at_drain > 0 {
            poisson_sender.cancel_outstanding();
            let cancel_deadline = tokio::time::Instant::now() + Duration::from_secs(3);
            while metrics_tracker.get_sample_counts().await.2 > 0
                && tokio::time::Instant::now() < cancel_deadline
            {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            let terminal = client_snapshot(
                &metrics_tracker,
                epoch_ms,
                seconds,
                started.elapsed().as_secs_f64(),
            )
            .await;
            for (key, value) in terminal.as_object().unwrap() {
                summary[key] = value.clone();
            }
            write_snapshot(&final_path, &summary).await?;
        }
        progress_handle.abort();
        metrics_tracker.flush_terminal_log().await?;
        if let Err(error) =
            tokio::time::timeout(Duration::from_secs(10), stats_collector.collect_and_save()).await
        {
            tracing::warn!(%error, "remote statistics deadline exceeded; local results already saved");
        }
        if pending_at_drain > 0 {
            return Err(
                "drain deadline exceeded; pending counts saved in final_summary.json".into(),
            );
        }
        return Ok(());
    }

    // 创建实时显示器
    let model_names: Vec<String> = config.models.iter().map(|m| m.name.clone()).collect();
    let live_display =
        LiveDisplay::new(model_names, metrics_tracker.clone(), poisson_sender.clone());

    // 设置全局错误显示器的进度条
    error_display::set_global_progress_bar(live_display.summary_bar.clone());

    info!("所有组件初始化完成");

    // 启动任务清理循环（每分钟清理一次过期任务）
    let cleanup_handle = {
        let metrics_tracker = metrics_tracker.clone();
        let cleanup_interval = 60; // 每60秒清理一次

        tokio::spawn(async move {
            task_cleanup_loop(metrics_tracker, cleanup_interval).await;
        })
    };

    // 启动状态收集循环
    let stats_collection_handle = {
        let stats_interval = config.sender.stats_interval;

        tokio::spawn(async move {
            stats_collection_loop(stats_collector, stats_interval).await;
        })
    };

    // 启动任务发送循环
    let task_sending_handle = {
        let poisson_sender = poisson_sender.clone();
        let models = config.models.clone();
        let cycle_duration = config.sender.cycle_duration;

        tokio::spawn(async move {
            task_sending_loop(poisson_sender, models, cycle_duration).await;
        })
    };

    info!("所有后台任务已启动");

    // 保持 live_display 的引用以便稍后清理
    let live_display = Arc::new(live_display);

    // 等待停止信号
    tokio::select! {
        _ = signal::ctrl_c() => {
            info!("收到停止信号，正在关闭...");
        }
        _ = cleanup_handle => {
            error_display::print_error("任务清理循环意外结束");
        }
        _ = stats_collection_handle => {
            error_display::print_error("状态收集循环意外结束");
        }
        _ = task_sending_handle => {
            error_display::print_error("任务发送循环意外结束");
        }

    }

    // 停止实时显示并打印最终统计信息
    live_display.stop();
    live_display.print_final_stats(&metrics_tracker).await;

    info!("任务发送器已停止");
    Ok(())
}

async fn client_snapshot(
    tracker: &MetricsTracker,
    epoch_ms: u128,
    seconds: u64,
    elapsed: f64,
) -> serde_json::Value {
    let counts = tracker.get_sample_counts().await;
    serde_json::json!({"started_at_ms": epoch_ms, "sending_seconds": seconds, "elapsed_seconds": elapsed,
        "sent_tasks": counts.0, "completed_tasks": counts.1, "pending_tasks": counts.2, "failed_tasks": counts.3,
        "sent_samples": counts.4, "completed_samples": counts.5, "pending_samples": counts.6, "failed_samples": counts.7,
        "transport": tracker.transport_snapshot().await,
        "slo_scope": "scheduled local arrival through receipt of the complete HTTP result; all offered requests are the denominator"})
}

async fn write_snapshot(
    path: &std::path::Path,
    value: &serde_json::Value,
) -> Result<(), std::io::Error> {
    let temporary = path.with_extension("json.tmp");
    tokio::fs::write(&temporary, serde_json::to_vec_pretty(value)?).await?;
    tokio::fs::rename(temporary, path).await
}

/// 任务清理循环（仅清理过期任务，状态更新现在在响应时直接处理）
async fn task_cleanup_loop(metrics_tracker: Arc<MetricsTracker>, cleanup_interval: u64) {
    let mut interval = tokio::time::interval(Duration::from_secs(cleanup_interval));

    loop {
        interval.tick().await;

        // 清理过期任务
        metrics_tracker.cleanup_expired_tasks().await;
        debug!("清理过期任务完成");
    }
}

/// 状态收集循环
async fn stats_collection_loop(mut stats_collector: StatsCollector, stats_interval: u64) {
    let mut interval = tokio::time::interval(Duration::from_secs(stats_interval));

    loop {
        interval.tick().await;

        if let Err(e) = stats_collector.collect_and_save().await {
            error_display::print_error(&format!("收集状态失败: {}", e));
        } else {
            info!("状态收集完成");
        }
    }
}

/// 任务发送循环
async fn task_sending_loop(
    poisson_sender: Arc<PoissonSender>,
    models: Vec<crate::config::ModelConfig>,
    cycle_duration: u64,
) {
    let mut cycle_interval = tokio::time::interval(Duration::from_secs(cycle_duration));

    loop {
        cycle_interval.tick().await;

        info!("开始新的发送周期");

        // 为每个模型更新目标样本速率
        for model in &models {
            let new_sample_rate = poisson_sender
                .update_model_rate(&model.name, &model.rate_range)
                .await;
            info!(
                "模型 {} 新样本速率: {:.2} samples/sec",
                model.name, new_sample_rate
            );
        }

        // 启动本周期的发送任务
        let cycle_handle = {
            let sender = poisson_sender.clone();
            let models = models.clone();
            let duration = Duration::from_secs(cycle_duration);

            tokio::spawn(async move {
                sender.run_cycle(models, duration).await;
            })
        };

        // 等待周期结束或接收到下一个周期信号
        tokio::select! {
            _ = cycle_handle => {
                info!("发送周期正常结束");
            }
            _ = cycle_interval.tick() => {
                info!("开始下一个周期，当前周期将被中断");
            }
        }
    }
}
