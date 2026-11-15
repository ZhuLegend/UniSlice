use clap::Parser;
use comm::{CommChannel, TypedConnection, service::Service};
use compute_node::{
    ComputeNode,
    compute_device::{hw::vgpu::mig::HuaweiMigShareableGpu, vgpu::resource_spec::hw::VChipScale},
    compute_node::{
        hw::HwVirtualGpuWorkerProxy,
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
use model::tensor::DynTensor;
use ndarray::Array4;
use tokio::{net::ToSocketAddrs, sync::mpsc};
use uuid::Uuid;
use worker::onnx::worker::OnnxWorker;

type HwMigVirtualComputeNode =
    BaseVirtualComputeNode<OnnxWorker, HuaweiMigShareableGpu, HwVirtualGpuWorkerProxy>;
type HwVirtualComputeNodeCommand = <HwMigVirtualComputeNode as ComputeNode>::Command;
type HwVirtualComputeNodeState = <HwMigVirtualComputeNode as ComputeNode>::State;
type HwVirtualComputeNodeStatus = <HwMigVirtualComputeNode as ComputeNode>::Status;
type HwComputeNodeInput = <HwMigVirtualComputeNode as ComputeNode>::InputData;
type HwComputeNodeOutput = <HwMigVirtualComputeNode as ComputeNode>::OutputData;

type ComputeNodeReq = BinComputeNodeRequest<HwVirtualComputeNodeCommand, HwComputeNodeInput>;
type ComputeNodeRes = BinComputeNodeResponse<
    HwVirtualComputeNodeState,
    HwVirtualComputeNodeStatus,
    HwComputeNodeOutput,
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
        HwVirtualComputeNodeCommand,
        HwVirtualComputeNodeState,
        HwVirtualComputeNodeStatus,
        HwComputeNodeInput,
        HwComputeNodeOutput,
    >(addr)
    .await
    .expect("Could not connect to compute node")
}

async fn state(addr: impl ToSocketAddrs) -> Option<HwVirtualComputeNodeState> {
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

async fn status(addr: impl ToSocketAddrs) -> Option<HwVirtualComputeNodeStatus> {
    let mut channel = connect_gpulet(addr).await;
    channel
        .send(&ComputeNodeReq::Status)
        .await
        .expect("Could not send status command");

    match channel.recv().await {
        Ok(Some(ComputeNodeRes::Status(status))) => Some(status),
        _ => None,
    }
}

async fn command(
    addr: impl ToSocketAddrs,
    command: HwVirtualComputeNodeCommand,
) -> Option<HwVirtualComputeNodeState> {
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
    tensor: DynTensor,
) -> Option<DynTensor> {
    let mut channel = connect_gpulet(addr).await;
    channel
        .send(&ComputeNodeReq::compute(Work {
            target_worker: gpulet_id,
            work: BaseWork::new(work_id, tensor),
        }))
        .await
        .expect("Could not send compute command");

    match channel.recv().await {
        Ok(Some(ComputeNodeRes::Compute(work))) => Some(work.work.take().1),
        Err(e) => {
            println!("Compute Error: {e:?}");
            None
        }
        _ => None,
    }
}

async fn handle_create_gpulet(
    addr: &impl ToSocketAddrs,
    rx: &mut mpsc::Receiver<String>,
) -> Option<State<VChipScale>> {
    println!("请输入Gpu ID:");
    let gpu_id = Uuid::parse_str(&rx.recv().await.unwrap()).unwrap();
    println!("请输入Gpulet ID:");
    let gpulet_id = Uuid::parse_str(&rx.recv().await.unwrap()).unwrap();
    println!("请输入切分目标:");
    let split_param = rx.recv().await.unwrap();
    command(
        addr,
        HwVirtualComputeNodeCommand::CreateWorker {
            id: gpulet_id,
            device_id: gpu_id,
            params: VChipScale::from(split_param.as_str()),
        },
    )
    .await
}

async fn handle_destroy_gpulet(
    addr: &impl ToSocketAddrs,
    rx: &mut mpsc::Receiver<String>,
) -> Option<State<VChipScale>> {
    println!("请输入Gpulet ID:");
    let gpulet_id = Uuid::parse_str(&rx.recv().await.unwrap()).unwrap();

    command(
        addr,
        HwVirtualComputeNodeCommand::DestroyWorker { id: gpulet_id },
    )
    .await
}

async fn handle_load_model(
    addr: &impl ToSocketAddrs,
    rx: &mut mpsc::Receiver<String>,
) -> Option<State<VChipScale>> {
    println!("请输入Gpulet ID:");
    let gpulet_id = Uuid::parse_str(&rx.recv().await.unwrap()).unwrap();
    println!("请输入模型名称:");
    let name = rx.recv().await.unwrap();

    command(
        addr,
        HwVirtualComputeNodeCommand::LoadModel {
            id: gpulet_id,
            name,
        },
    )
    .await
}

async fn handle_unload_model(
    addr: &impl ToSocketAddrs,
    rx: &mut mpsc::Receiver<String>,
) -> Option<State<VChipScale>> {
    println!("请输入Gpulet ID:");
    let gpulet_id = Uuid::parse_str(&rx.recv().await.unwrap()).unwrap();

    command(
        addr,
        HwVirtualComputeNodeCommand::UnloadModel { id: gpulet_id },
    )
    .await
}

async fn dispatch_subcommand(
    subcommand: &str,
    addr: &impl ToSocketAddrs,
    rx: &mut mpsc::Receiver<String>,
) {
    match subcommand {
        "state" => match state(&addr).await {
            Some(state) => {
                println!("State: {state:?}");
            }
            None => println!("Unexpected frame"),
        },
        "status" => match status(&addr).await {
            Some(status) => {
                println!("Status: {status:?}");
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
        "destroy_gpulet" => {
            let res = handle_destroy_gpulet(&addr, rx).await;

            if res.is_some() {
                println!("销毁成功");
            } else {
                println!("销毁失败");
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
            println!("请输入批大小:");
            let batch_size: usize = rx.recv().await.unwrap().parse().unwrap();
            let tensor = DynTensor::from(Array4::<f32>::ones((batch_size, 3, 224, 224)));

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
    println!("请输入指令:");
    while let Some(line) = rx.recv().await {
        let line = line.trim();
        dispatch_subcommand(line, &addr, &mut rx).await;
        println!("请输入指令:");
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
