use clap::Parser;
use comm::{CommChannel, TypedConnection, service::Service};
use compute_node::{
    ComputeNode,
    compute_device::{
        nv::vgpu::mps::NvidiaMpsShareableGpu,
        vgpu::{resource_spec::nv::mps::NvidiaMpsShareTarget, share_target::ShareTarget},
    },
    compute_node::{
        nv::NvMpsVirtualGpuWorkerProxy,
        r#virtual::{
            BaseWork,
            base::{BaseVirtualComputeNode, State, Work},
        },
    },
    util::bin_compute_node::{
        BinComputeNodeRequest, BinComputeNodeResponse, connect_bin_compute_node,
    },
};
use frontend::listen_input;
use tch::{Tensor, vision::imagenet::load_image_and_resize224};
use tokio::{net::ToSocketAddrs, sync::mpsc};
use uuid::Uuid;
use worker::torch::worker::TorchWorker;

type NvMpsVirtualComputeNode =
    BaseVirtualComputeNode<TorchWorker, NvidiaMpsShareableGpu, NvMpsVirtualGpuWorkerProxy>;
type NvVirtualComputeNodeCommand = <NvMpsVirtualComputeNode as ComputeNode>::Command;
type NvVirtualComputeNodeState = <NvMpsVirtualComputeNode as ComputeNode>::State;
type NvVirtualComputeNodeStatus = <NvMpsVirtualComputeNode as ComputeNode>::Status;
type NvComputeNodeInput = <NvMpsVirtualComputeNode as ComputeNode>::InputData;
type NvComputeNodeOutput = <NvMpsVirtualComputeNode as ComputeNode>::OutputData;

type ComputeNodeReq = BinComputeNodeRequest<NvVirtualComputeNodeCommand, NvComputeNodeInput>;
type ComputeNodeRes = BinComputeNodeResponse<
    NvVirtualComputeNodeState,
    NvVirtualComputeNodeStatus,
    NvComputeNodeOutput,
>;

#[derive(Parser)]
struct Cli {
    /// Node id
    #[clap(long)]
    #[arg(default_value = "127.0.0.1")]
    ip: String,

    /// 监听端口
    #[arg(long, env, value_parser = clap::value_parser ! (u16).range(1..))]
    port: u16,
}

async fn connect_gpulet(
    addr: impl ToSocketAddrs,
) -> TypedConnection<impl Service, ComputeNodeReq, ComputeNodeRes> {
    connect_bin_compute_node::<
        NvVirtualComputeNodeCommand,
        NvVirtualComputeNodeState,
        NvVirtualComputeNodeStatus,
        NvComputeNodeInput,
        NvComputeNodeOutput,
    >(addr)
    .await
    .expect("Could not connect to compute node")
}

async fn state(addr: impl ToSocketAddrs) -> Option<NvVirtualComputeNodeState> {
    let mut channel = connect_gpulet(addr).await;
    channel
        .send(&ComputeNodeReq::State)
        .await
        .expect("Could not send state command");

    match channel.recv().await {
        Ok(Some(ComputeNodeRes::State(state))) => Some(state),
        _ => None,
    }
}

async fn command(
    addr: impl ToSocketAddrs,
    command: NvVirtualComputeNodeCommand,
) -> Option<NvVirtualComputeNodeState> {
    let mut channel = connect_gpulet(addr).await;
    channel
        .send(&ComputeNodeReq::command(command))
        .await
        .expect("Could not send command");

    match channel.recv().await {
        Ok(res) => match res {
            Some(BinComputeNodeResponse::State(state)) => {
                print!("Compute State: {state:?}");
                return Some(state);
            }
            other => print!("Unexpected Res: {other:?}"),
        },
        Err(e) => println!("Command Error: {e:?}"),
    }

    None
}

async fn compute(
    addr: impl ToSocketAddrs,
    work_id: Uuid,
    gpulet_id: Uuid,
    tensor: Tensor,
) -> Option<Tensor> {
    let mut channel = connect_gpulet(addr).await;
    channel
        .send(&ComputeNodeReq::compute(Work {
            target_worker: gpulet_id,
            work: BaseWork::new(work_id, tensor.into()),
        }))
        .await
        .expect("Could not send compute command");

    match channel.recv().await {
        Ok(Some(ComputeNodeRes::Compute(work))) => Some(Tensor::from(work.work.take().1)),
        _ => None,
    }
}

async fn handle_create_gpulet(
    addr: &impl ToSocketAddrs,
    rx: &mut mpsc::Receiver<String>,
) -> Option<State<NvidiaMpsShareTarget>> {
    println!("请输入Gpu ID:");
    let gpu_id = Uuid::parse_str(&rx.recv().await.unwrap()).unwrap();
    println!("请输入Gpulet ID:");
    let gpulet_id = Uuid::parse_str(&rx.recv().await.unwrap()).unwrap();
    println!("请输入百分比:");
    let percentage = rx.recv().await.unwrap();
    command(
        addr,
        NvVirtualComputeNodeCommand::CreateWorker {
            id: gpulet_id,
            device_id: gpu_id,
            params: NvidiaMpsShareTarget::from_name(&percentage)?,
        },
    )
    .await
}

async fn handle_load_model(
    addr: &impl ToSocketAddrs,
    rx: &mut mpsc::Receiver<String>,
) -> Option<State<NvidiaMpsShareTarget>> {
    println!("请输入Gpulet ID:");
    let gpulet_id = Uuid::parse_str(&rx.recv().await.unwrap()).unwrap();
    println!("请输入模型名称:");
    let name = rx.recv().await.unwrap();

    command(
        addr,
        NvVirtualComputeNodeCommand::LoadModel {
            id: gpulet_id,
            name,
        },
    )
    .await
}

async fn handle_unload_model(
    addr: &impl ToSocketAddrs,
    rx: &mut mpsc::Receiver<String>,
) -> Option<State<NvidiaMpsShareTarget>> {
    println!("请输入Gpulet ID:");
    let gpulet_id = Uuid::parse_str(&rx.recv().await.unwrap()).unwrap();

    command(
        addr,
        NvVirtualComputeNodeCommand::UnloadModel { id: gpulet_id },
    )
    .await
}

async fn dispatch_subcommand(
    subcommand: &str,
    addr: &impl ToSocketAddrs,
    rx: &mut mpsc::Receiver<String>,
) {
    println!("请输入指令:");
    match subcommand {
        "state" => match state(&addr).await {
            Some(state) => {
                println!("State: {state:?}");
            }
            None => println!("Unexpected frame"),
        },
        "create_gpulet" => {
            let res = handle_create_gpulet(&addr, rx).await;

            if res.is_some() {
                println!("创建成功");
            } else {
                println!("创建失败");
            }
        }
        "load_model" => {
            let res = handle_load_model(&addr, rx).await;

            if res.is_some() {
                println!("加载成功")
            } else {
                println!("加载失败")
            }
        }
        "unload_model" => {
            let res = handle_unload_model(&addr, rx).await;

            if res.is_some() {
                println!("卸载成功")
            } else {
                println!("卸载失败")
            }
        }
        "compute" => {
            println!("请输入Gpulet ID:");
            let gpulet_id = Uuid::parse_str(&rx.recv().await.unwrap()).unwrap();
            println!("请输入图像路径:");
            let path = rx.recv().await.unwrap();
            let tensor = load_image_and_resize224(path).unwrap().unsqueeze(0);

            let res = compute(&addr, Uuid::new_v4(), gpulet_id, tensor).await;

            if let Some(tensor) = res {
                println!("计算结果: {tensor:?}");
            } else {
                println!("计算失败");
            }
        }
        _ => println!("未知指令: {subcommand}"),
    };
}

async fn user_handler(addr: impl ToSocketAddrs, mut rx: mpsc::Receiver<String>) {
    while let Some(line) = rx.recv().await {
        let line = line.trim();
        dispatch_subcommand(line, &addr, &mut rx).await;
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let cli = Cli::parse();
    let (tx, rx) = mpsc::channel(100);

    let addr = (cli.ip.as_str(), cli.port);
    listen_input(tx);

    user_handler(addr, rx).await;
}
