use std::path::PathBuf;

use clap::Parser;
use compute_node::{
    compute_device::nv::vgpu::mig::NvidiaMigShareableGpu,
    compute_node::{
        nv::{NvMigVirtualGpuWorkerProxy, WorkerType},
        r#virtual::base::{BaseVirtualComputeNode, Work},
    },
    util::{Input, Output, bin_compute_node::BinComputeNodeWrapper},
};
use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;
use utils::fs::DirPathBuf;
#[cfg(feature = "onnx")]
use worker::onnx::worker::OnnxWorker;
use worker::torch::worker::TorchWorker;

#[derive(Parser)]
#[clap(version, name = "nv_virtual")]
#[command(version, about, long_about = None)]
struct Cli {
    /// 需要加载的GPU id
    #[arg(long)]
    device_id: Vec<usize>,

    /// 模型文件夹路径
    #[arg(long, env, value_parser = utils::fs::ValidDirPathParser)]
    model_dir: DirPathBuf,

    /// 模型配置文件路径
    #[arg(long, env, value_parser = utils::fs::ValidDirPathParser)]
    model_config: DirPathBuf,

    /// 需要加载的[Worker](worker::Worker)可执行文件路径
    #[arg(long)]
    bin_worker_path: PathBuf,

    /// 加载的[worker::Worker]类型
    #[arg(long, value_enum)]
    worker_type: WorkerType,

    /// socket监听端口
    #[arg(short, long, value_parser = clap::value_parser ! (u16).range(1024..))]
    port: u16,

    /// 接受的工作负载是否仅有batch size
    #[arg(long)]
    batch_size_only: bool,
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
            .init();
    }

    let cli = Cli::parse();

    let tcp_listener = TcpListener::bind(format!("0.0.0.0:{}", cli.port))
        .await
        .unwrap();

    let (input_transformer, output_transformer) = if cli.batch_size_only {
        (
            Some(|work: Work<Input>| work.with_batch_size_only(vec![3, 224, 224])),
            Some(Work::<Output>::keep_batch_size_only),
        )
    } else {
        (None, None)
    };

    type NvVirtualComputeNode<Worker> =
        BaseVirtualComputeNode<Worker, NvidiaMigShareableGpu, NvMigVirtualGpuWorkerProxy>;
    match cli.worker_type {
        #[cfg(feature = "torch")]
        WorkerType::Torch => {
            let wrapper = BinComputeNodeWrapper::builder()
                .wrap(NvVirtualComputeNode::<TorchWorker>::new(
                    cli.device_id,
                    cli.model_dir.into(),
                    cli.model_config.into(),
                    cli.bin_worker_path,
                ))
                .max_frame_size(64 * 4 * 3 * 224 * 224 + 1024)
                .maybe_input_transformer(input_transformer)
                .maybe_output_transformer(output_transformer)
                .build();
            wrapper
                .serve_until(
                    tcp_listener,
                    compute_node::util::bin_compute_node::shutdown_signal(),
                )
                .await?;
        }
        #[cfg(feature = "onnx")]
        WorkerType::Onnx => {
            let wrapper = BinComputeNodeWrapper::builder()
                .wrap(NvVirtualComputeNode::<OnnxWorker>::new(
                    cli.device_id,
                    cli.model_dir.into(),
                    cli.model_config.into(),
                    cli.bin_worker_path,
                ))
                .max_frame_size(64 * 4 * 3 * 224 * 224 + 1024)
                .maybe_input_transformer(input_transformer)
                .maybe_output_transformer(output_transformer)
                .build();
            wrapper
                .serve_until(
                    tcp_listener,
                    compute_node::util::bin_compute_node::shutdown_signal(),
                )
                .await?;
        }
    };
    Ok(())
}
