//! 使用[MockVirtualComputeDevice](compute_node::compute_device::mock::MockVirtualComputeDevice)模拟一个vgpu设备

use std::path::PathBuf;

use clap::Parser;
use compute_node::util::{
    bin_compute_node::BinComputeNodeWrapper, simple_compute_node::SimpleComputeNode,
};
use tokio::net::TcpListener;
use uuid::Uuid;
use worker::mock::{MockModelInitConfig, MockWorker};

#[derive(Parser)]
#[clap(version, name = "mock_vgpu")]
#[command(version, about, long_about = None)]
struct Cli {
    /// vgpu id
    #[arg(long, value_parser = clap::value_parser!(Uuid))]
    id: Uuid,

    /// 虚拟设备目标
    #[arg(long)]
    config: String,

    /// 延迟配置文件路径
    #[arg(long, env)]
    latency_profile: PathBuf,

    /// Resolved model costs supplied by the parent node as JSON.
    #[arg(long, default_value = "{}")]
    model_lifecycle: String,
    /// Resolved inference variation supplied by the parent, not lifecycle cost.
    #[arg(long, default_value = "{\"cv\":0.0,\"seed\":0}")]
    execution_noise: String,
    /// Final per-model seeds for successive genuine loads, supplied by the parent.
    #[arg(long, default_value = "{}")]
    model_noise_seeds: String,

    /// socket监听端口
    #[arg(short, long, default_value_t = 0)]
    port: u16,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .json()
        .with_writer(std::io::stderr)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let cli = Cli::parse();
    let costs: std::collections::BTreeMap<String, worker::mock::ModelLifecycle> =
        serde_json::from_str(&cli.model_lifecycle).expect("invalid resolved model lifecycle");
    let noise: worker::mock::ExecutionNoise =
        serde_json::from_str(&cli.execution_noise).expect("invalid execution noise");
    noise.validate().expect("invalid execution noise CV");
    let seeds: std::collections::BTreeMap<String, Vec<u64>> =
        serde_json::from_str(&cli.model_noise_seeds).expect("invalid model noise seed schedules");

    let model_provider = move |model_name: &str| -> MockModelInitConfig {
        let config = MockModelInitConfig::new(
            model_name.to_string(),
            cli.config.clone(),
            cli.latency_profile.clone(),
        )
        .with_lifecycle(costs.get(model_name).copied().unwrap_or_default())
        .with_execution_noise(noise);
        match seeds.get(model_name) {
            Some(seeds) => config.with_execution_noise_seeds(seeds.clone()),
            None => config,
        }
    };

    let tcp_listener = TcpListener::bind(format!("127.0.0.1:{}", cli.port))
        .await
        .unwrap();

    compute_node::util::worker_proxy::bin_simple_worker_proxy::announce_worker_address(
        cli.id,
        tcp_listener.local_addr().unwrap().port(),
    )
    .unwrap();

    BinComputeNodeWrapper::builder()
        .wrap(SimpleComputeNode::new(
            MockWorker::with_id(cli.id.to_string()),
            model_provider,
        ))
        .max_frame_size(64 * 4 * 3 * 224 * 224 + 1024)
        .build()
        .serve_until(
            tcp_listener,
            compute_node::util::bin_compute_node::shutdown_signal(),
        )
        .await?;
    Ok(())
}
