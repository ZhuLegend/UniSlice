//! 使用[Huawei MIG]拆分GPU资源，运行特定worker的可执行文件

use clap::Parser;
use compute_node::{
    compute_device::hw::{dcmi, gpu::HuaweiGpu},
    compute_node::hw::WorkerType,
    model_provider::ModelProvider,
    util::{bin_compute_node::BinComputeNodeWrapper, simple_compute_node::SimpleComputeNode},
};
use hw_dcmi_wrapper::device::card::Card;
use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;
use utils::fs::DirPathBuf;
use uuid::Uuid;
use worker::onnx::worker::OnnxWorker;

#[derive(Parser)]
#[clap(version, name = "hw_vgpu")]
#[command(version, about, long_about = None)]
struct Cli {
    /// vgpu id
    #[arg(long, value_parser = clap::value_parser!(Uuid))]
    id: Uuid,

    /// device id
    #[arg(long, default_value_t = 0)]
    device_id: u32,

    /// 模型文件夹路径
    #[arg(long, env, value_parser = utils::fs::ValidDirPathParser)]
    model_dir: DirPathBuf,

    /// 模型配置文件路径
    #[arg(long, env, value_parser = utils::fs::ValidDirPathParser)]
    model_config: DirPathBuf,

    /// 加载的[worker::Worker]类型
    #[arg(long, value_enum)]
    worker_type: WorkerType,

    /// socket监听端口
    #[arg(short, long, value_parser = clap::value_parser!(u16))]
    port: u16,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "console")]
    {
        use tracing_subscriber::prelude::*;
        tracing_subscriber::registry()
            .with(console_subscriber::spawn())
            .with(
                tracing_subscriber::fmt::layer()
                    .pretty()
                    .with_writer(std::io::stderr)
                    .with_filter(EnvFilter::from_default_env()),
            )
            .init();
        tracing::info!("tokio-console is enabled.");
    }
    #[cfg(not(feature = "console"))]
    {
        tracing_subscriber::fmt::fmt()
            .with_env_filter(EnvFilter::from_default_env())
            .pretty()
            .with_writer(std::io::stderr)
            .init();
    }

    let cli = Cli::parse();

    let model_provider = {
        let dcmi = dcmi();
        let device = Card::new_unchecked(dcmi, 0);
        ModelProvider::new(cli.model_dir, cli.model_config, HuaweiGpu::new(device))
    };

    let tcp_listener = TcpListener::bind(format!("0.0.0.0:{}", cli.port))
        .await
        .unwrap();
    compute_node::util::worker_proxy::bin_simple_worker_proxy::announce_worker_address(
        cli.id,
        tcp_listener.local_addr().unwrap().port(),
    )
    .unwrap();

    match cli.worker_type {
        WorkerType::Onnx => {
            BinComputeNodeWrapper::builder()
                .wrap(SimpleComputeNode::new(
                    OnnxWorker::default(),
                    model_provider.onnx_provider(),
                ))
                .max_frame_size(64 * 4 * 3 * 224 * 224 + 1024)
                .build()
                .serve_until(
                    tcp_listener,
                    compute_node::util::bin_compute_node::shutdown_signal(),
                )
                .await?;
        }
    }

    Ok(())
}
