//! 通用守护进程启动器

use std::{collections::HashMap, error::Error, pin::Pin, sync::Arc, time::Duration};

use axum::{
    Router,
    extract::DefaultBodyLimit,
    routing::{get, post},
};
use compute_node::compute_node::r#virtual::virtual_compute_node::{
    lifecycle_events_layer, task_tracking_layer, worker_perf_layer,
};
use dashmap::DashMap;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
#[allow(unused_imports)]
use tracing_subscriber::{EnvFilter, Layer, layer::SubscriberExt, util::SubscriberInitExt};

#[cfg(all(feature = "mock", feature = "nvidia"))]
use crate::daemon::experiment_allocation::{AllocationPlan, FixedAllocationScheduler};
use crate::{
    daemon::{
        config::{load_config, validate_config},
        config_builder::{HeterogeneousSystemBuilder, template_pool_ids},
        csv_logger::CsvTaskLogger,
        handlers,
        predictor::load_latency_csv,
        scheduler::{SchedulingPolicy, scheduler_loop_with_policy},
        stats::SchedulerStatsManager,
        types::{AppState, Config, ModelRequestRatePredictor},
        utils::{parse_listen_address, shutdown_signal},
    },
    schedule_system::virtual_compute_node::system::heterogeneous::system::HeterogeneousSystem,
};

/// 命令行参数
#[derive(clap::Parser, Debug)]
pub struct Args {
    /// HTTP监听端口
    #[arg(short, long, default_value = "3000")]
    pub port: u16,

    /// HTTP监听主机
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,

    /// 配置文件路径
    #[arg(short, long, default_value = "config.toml")]
    pub config: std::path::PathBuf,

    /// 调度间隔（秒）
    #[arg(short, long, default_value = "300")]
    pub interval: u64,

    /// Concurrent HTTP/2 requests per persistent data connection.
    #[arg(long, default_value_t = 65536)]
    pub http2_max_streams: u32,

    /// Global accepted request limit, including work left by disconnected callers.
    #[arg(long, default_value_t = 65536)]
    pub max_pending_requests: usize,

    /// Global encoded-input budget in MiB.
    #[arg(long, default_value_t = 256)]
    pub request_memory_mib: usize,

    /// 延迟预测CSV文件路径
    #[arg(short, long)]
    pub latency_csv: Option<std::path::PathBuf>,

    /// Optional template-to-prediction-CSV map; paths are relative to the JSON file.
    #[arg(long)]
    pub prediction_csvs: Option<std::path::PathBuf>,

    /// Shared simulation topology, aliases and lifecycle contract.
    #[arg(long)]
    pub simulation_config: Option<std::path::PathBuf>,

    /// Install a supplied allocation once and keep normal local dynamic batching.
    #[arg(
        long,
        requires = "simulation_config",
        conflicts_with = "experiment_freeze_greedy"
    )]
    pub experiment_fixed_allocation: Option<std::path::PathBuf>,

    /// Hold the native greedy scheduler's first allocation for a static comparison.
    #[arg(
        long,
        requires = "simulation_config",
        conflicts_with = "experiment_fixed_allocation"
    )]
    pub experiment_freeze_greedy: bool,

    /// 日志输出目录
    #[arg(long, default_value = "logs")]
    pub log_dir: std::path::PathBuf,
}

#[cfg(test)]
mod experiment_cli_tests {
    use clap::Parser;

    use super::Args;

    #[test]
    fn experiment_policies_require_simulation_and_are_mutually_exclusive() {
        assert!(Args::try_parse_from(["daemon", "--experiment-freeze-greedy"]).is_err());
        assert!(
            Args::try_parse_from(["daemon", "--experiment-fixed-allocation", "plan.json"]).is_err()
        );
        assert!(
            Args::try_parse_from([
                "daemon",
                "--simulation-config",
                "simulation.json",
                "--experiment-freeze-greedy",
                "--experiment-fixed-allocation",
                "plan.json",
            ])
            .is_err()
        );
        let args = Args::try_parse_from([
            "daemon",
            "--simulation-config",
            "simulation.json",
            "--experiment-freeze-greedy",
        ])
        .unwrap();
        assert!(args.experiment_freeze_greedy);
        assert!(args.experiment_fixed_allocation.is_none());
    }

    #[test]
    fn default_cli_retains_dynamic_greedy() {
        let args = Args::try_parse_from(["daemon"]).unwrap();
        assert!(!args.experiment_freeze_greedy);
        assert!(args.experiment_fixed_allocation.is_none());
    }
}

/// 启动守护进程
pub async fn run(args: Args) -> Result<(), Box<dyn Error>> {
    let request_bytes = args
        .request_memory_mib
        .checked_mul(1024 * 1024)
        .filter(|&bytes| bytes > 0 && bytes <= tokio::sync::Semaphore::MAX_PERMITS)
        .ok_or("request-memory-mib must be positive and fit the semaphore budget")?;
    if args.max_pending_requests == 0
        || args.max_pending_requests > tokio::sync::Semaphore::MAX_PERMITS
    {
        return Err("max-pending-requests must be positive and fit the semaphore budget".into());
    }
    // 初始化日志
    let (task_tracking_layer, _task_guard) = task_tracking_layer(&args.log_dir);
    let (lifecycle_layer, _lifecycle_guard) = lifecycle_events_layer(&args.log_dir);
    let (worker_perf_layer, _worker_perf_guard) = worker_perf_layer(&args.log_dir);
    #[cfg(feature = "console")]
    {
        use tracing_subscriber::prelude::*;
        tracing_subscriber::registry()
            .with(console_subscriber::spawn())
            .with(
                tracing_subscriber::fmt::layer()
                    .pretty()
                    .with_filter(EnvFilter::from_default_env()),
            )
            .with(task_tracking_layer)
            .with(worker_perf_layer)
            .with(lifecycle_layer)
            .init();
        tracing::info!("tokio-console is enabled.");
    }
    #[cfg(not(feature = "console"))]
    {
        tracing_subscriber::registry()
            .with(
                tracing_subscriber::fmt::layer()
                    .pretty()
                    .with_filter(EnvFilter::from_default_env()),
            )
            .with(task_tracking_layer)
            .with(worker_perf_layer)
            .with(lifecycle_layer)
            .init();
    }

    tracing::info!("启动异构调度器守护进程");
    tracing::info!("HTTP监听地址: {}:{}", args.host, args.port);
    tracing::info!("配置文件: {:?}", args.config);

    // 加载配置
    let mut config: Config = load_config(&args.config).await?;
    validate_config(&config)?;

    // 加载延迟数据
    let mut profile_aliases = HashMap::new();
    let mut latency_data = if let Some(path) = &args.simulation_config {
        #[cfg(not(feature = "mock"))]
        return Err("simulation requires the mock feature".into());
        #[cfg(feature = "mock")]
        {
            let simulation = compute_node::simulation::SimulationConfig::load(path)?;
            crate::daemon::config::configure_simulation(&mut config, &simulation)?;
            profile_aliases = simulation.profile_aliases.clone();
            simulation.profiles()?
        }
    } else {
        load_latency_csv(
            args.latency_csv
                .as_deref()
                .ok_or("--latency-csv is required without --simulation-config")?,
        )
        .await?
    };
    // Overrides feed predictors only. Workers retain the original simulation contract.
    if args.simulation_config.is_some() {
        if let Some(path) = &args.latency_csv {
            let mut predictions = load_latency_csv(path).await?;
            latency_profile::apply_aliases(&mut predictions, &profile_aliases);
            crate::daemon::predictor::validate_latency_coverage(&latency_data, &predictions)?;
            latency_data = predictions;
        }
    }
    let mut template_latency_data = HashMap::new();
    if let Some(path) = &args.prediction_csvs {
        let paths: HashMap<String, std::path::PathBuf> =
            serde_json::from_slice(&tokio::fs::read(path).await?)?;
        for (template, csv) in paths {
            if !config.device_combinations.contains_key(&template) {
                return Err(format!("Unknown prediction CSV template: {template}").into());
            }
            let csv = path.parent().unwrap_or(std::path::Path::new(".")).join(csv);
            let mut predictions = load_latency_csv(&csv).await?;
            latency_profile::apply_aliases(&mut predictions, &profile_aliases);
            crate::daemon::predictor::validate_latency_coverage(&latency_data, &predictions)?;
            template_latency_data.insert(template, Arc::new(predictions));
        }
    }
    let models = if args.simulation_config.is_some() {
        config.init_arrival_rate.keys().cloned().collect::<Vec<_>>()
    } else {
        latency_data.keys().cloned().collect::<Vec<_>>()
    };

    if models.is_empty() {
        return Err("No supported models configured".into());
    }
    // These experimental policies are available only on the shared mock contract.
    if (args.experiment_fixed_allocation.is_some() || args.experiment_freeze_greedy)
        && args.simulation_config.is_none()
    {
        return Err("experiment allocation policies require --simulation-config".into());
    }
    let pool_names = template_pool_ids(&config)
        .into_iter()
        .map(|(name, id)| (id, name))
        .collect();
    let policy = if let Some(path) = &args.experiment_fixed_allocation {
        #[cfg(all(feature = "mock", feature = "nvidia"))]
        {
            let plan: AllocationPlan = serde_json::from_slice(&tokio::fs::read(path).await?)?;
            SchedulingPolicy::Fixed(FixedAllocationScheduler::from_plan(
                plan,
                &config,
                &latency_data,
            )?)
        }
        #[cfg(not(all(feature = "mock", feature = "nvidia")))]
        return Err(format!(
            "fixed allocation {:?} requires mock and nvidia features",
            path
        )
        .into());
    } else if args.experiment_freeze_greedy {
        #[cfg(all(feature = "mock", feature = "nvidia"))]
        {
            if !config
                .device_combinations
                .values()
                .all(|d| matches!(d, crate::daemon::types::DeviceConfig::MockNvidia { .. }))
            {
                return Err("experiment-freeze-greedy currently requires MockNvidia pools".into());
            }
            SchedulingPolicy::FrozenGreedy
        }
        #[cfg(not(all(feature = "mock", feature = "nvidia")))]
        return Err("experiment-freeze-greedy requires mock and nvidia features".into());
    } else {
        SchedulingPolicy::Dynamic
    };
    let supported_models = Arc::new(models.iter().cloned().collect());

    // 初始化组件
    let csv_logger = CsvTaskLogger::new(&args.log_dir)
        .await
        .map_err(|e| e.to_string())?;
    let scheduler_stats = SchedulerStatsManager::new(&args.log_dir)
        .await
        .map_err(|e| e.to_string())?;

    // 创建模型请求速率预测器
    let model_predictor = ModelRequestRatePredictor::with_initial_rates(&config.init_arrival_rate);

    let simulation_startup = args.simulation_config.as_ref().map(|_| {
        Arc::new(crate::daemon::types::SimulationStartup {
            ready: std::sync::atomic::AtomicBool::new(false),
            started: tokio::sync::watch::channel(false).0,
            initial_rates: config.init_arrival_rate.clone(),
        })
    });

    // 使用配置构建器创建异构系统
    let heterogeneous_system = HeterogeneousSystemBuilder::build_from_config_and_rate_predictor(
        config,
        create_predictor_closure(model_predictor.clone(), models, simulation_startup.clone()),
        Arc::new(latency_data),
        template_latency_data,
    )
    .await?;

    // 创建应用状态
    let state = AppState {
        request_slots: Arc::new(tokio::sync::Semaphore::new(args.max_pending_requests)),
        request_bytes: Arc::new(tokio::sync::Semaphore::new(request_bytes)),
        task_totals: Arc::default(),
        terminal_history: Arc::default(),
        supported_models,
        simulation_startup,
        task_stats: Arc::new(DashMap::new()),
        schedule_system: Arc::new(heterogeneous_system),
        rate_predictor: model_predictor,
        csv_logger,
        scheduler_stats,
    };

    // 构建并启动服务器
    let app = build_router(state.clone());
    let addr = parse_listen_address(&args.host, args.port)?;
    tracing::info!("HTTP服务器启动在 http://{}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await?;

    // 启动后台任务
    let schedule_handle = tokio::spawn(scheduler_loop_with_policy(
        state.clone(),
        args.interval,
        policy,
        pool_names,
    ));
    let cleanup_handle = tokio::spawn(cleanup_loop(state.rate_predictor));
    let perform_latency_update_handle =
        tokio::spawn(perform_latency_update_loop(state.schedule_system.clone()));

    let result: Result<(), Box<dyn std::error::Error>> = tokio::select! {
        result = super::http_server::serve(listener, app, args.http2_max_streams, shutdown_signal()) => {
            result.map_err(|error| error.into())
        }
        result = schedule_handle => result.map_err(|error| error.into()),
        result = cleanup_handle => result.map_err(|error| error.into()),
        result = perform_latency_update_handle => result.map_err(|error| error.into()),
    };

    state.csv_logger.flush().await?;
    tracing::info!("守护进程已停止");
    result
}

/// 创建模型请求速率预测器的闭包
fn create_predictor_closure(
    predictor: ModelRequestRatePredictor,
    models: Vec<String>,
    startup: Option<Arc<crate::daemon::types::SimulationStartup>>,
) -> impl Fn() -> Pin<Box<dyn Future<Output = HashMap<String, f32>> + Send>> + Send + Sync + 'static
{
    let models = Arc::new(models);
    move || {
        let models = models.clone();
        let predictor = predictor.clone();
        let startup = startup.clone();
        Box::pin(async move {
            if let Some(gate) = &startup {
                if !*gate.started.borrow() {
                    return gate.initial_rates.clone();
                }
            }
            let mut res = predictor.predict().await;
            res.retain(|model, _| models.contains(model));
            for model in &*models {
                if !res.contains_key(model) {
                    res.insert(model.clone(), 0.0);
                }
            }
            res
        })
    }
}

/// 预测数据清理循环
async fn cleanup_loop(rate_predictor: ModelRequestRatePredictor) {
    let mut interval = tokio::time::interval(Duration::from_secs(300)); // 5分钟清理一次
    loop {
        tokio::select! {
            _ = interval.tick() => {
                tracing::debug!("开始清理预测数据");
                rate_predictor.cleanup_old_records(300);
                let remaining_models = rate_predictor.get_tracked_models();
                tracing::info!("预测数据清理完成，剩余跟踪模型数: {}", remaining_models.len());
            }
            _ = tokio::signal::ctrl_c() => break,
        }
    }
    tracing::info!("预测数据清理任务已停止");
}

async fn perform_latency_update_loop(system: Arc<HeterogeneousSystem>) {
    let mut interval = tokio::time::interval(Duration::from_secs(1)); // 每1秒更新一次
    loop {
        tokio::select! {
            _ = interval.tick() => {
                system.perform_latency_update().await;
            }
            _ = tokio::signal::ctrl_c() => break,
        }
    }
    tracing::info!("延迟数据更新任务已停止");
}

/// 构建路由
fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/", get(handlers::health))
        .route("/health", get(handlers::health))
        .route("/tasks", post(handlers::submit_task))
        .route("/tasks", get(handlers::get_task_list))
        .route("/tasks/:task_id", get(handlers::get_task))
        .route("/tasks/status/:status", get(handlers::get_tasks_by_status))
        .route(
            "/tasks/completed",
            axum::routing::delete(handlers::clear_completed_tasks),
        )
        .route("/stats", get(handlers::get_stats))
        .route("/predictions", get(handlers::get_prediction_stats))
        .route("/routing/weights", get(handlers::get_routing_weights))
        .route("/scheduler/stats", get(handlers::scheduler_stats))
        .route(
            "/scheduler/stats/reset",
            axum::routing::delete(handlers::reset_scheduler_stats),
        )
        .layer(
            ServiceBuilder::new()
                .layer(CorsLayer::permissive())
                .layer(DefaultBodyLimit::max(100 * 1024 * 1024)), // 100MB
        )
        .with_state(state)
}
