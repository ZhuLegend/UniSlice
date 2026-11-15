use std::{io, io::BufRead, thread};

use clap::Parser;
use comm::{CommChannel, TypedConnection, service::Service};
use compute_node::util::{
    Input, Output,
    bin_compute_node::{BinComputeNodeRequest, BinComputeNodeResponse, connect_bin_compute_node},
    simple_compute_node::{SimpleComputeNodeCommand, SimpleComputeNodeWork},
};
use model::tensor::DynTensor;
use tokio::{net::ToSocketAddrs, sync::mpsc};

pub type SimpleComputeNodeReq =
    BinComputeNodeRequest<SimpleComputeNodeCommand, SimpleComputeNodeWork<Input>>;
pub type SimpleComputeNodeRes =
    BinComputeNodeResponse<Option<String>, (), SimpleComputeNodeWork<Output>>;

#[derive(Parser)]
pub struct Cli {
    #[clap(long)]
    #[arg(default_value = "127.0.0.1")]
    pub ip: String,
    /// gpulet 监听地址
    #[clap(short, long, env)]
    pub port: u16,

    #[clap(long)]
    #[arg(default_value = "1")]
    pub batch_size: i64,

    #[clap(long)]
    pub auto_loop: bool,

    /// 需要加载的模型名称
    #[clap(long)]
    pub model_name: String,
}

pub fn listen_input(tx: mpsc::Sender<String>) {
    // 在一个新的线程中读取用户的输入
    thread::spawn(move || {
        let stdin = io::stdin();
        let lines = stdin.lock().lines();

        for line in lines {
            let line = line.expect("Could not read line");
            tx.blocking_send(line).expect("Could not send line");
        }
    });
}

pub async fn connect_gpulet(
    addr: impl ToSocketAddrs,
) -> TypedConnection<
    impl Service,
    BinComputeNodeRequest<SimpleComputeNodeCommand, SimpleComputeNodeWork<Input>>,
    BinComputeNodeResponse<Option<String>, (), SimpleComputeNodeWork<Output>>,
> {
    connect_bin_compute_node::<
        SimpleComputeNodeCommand,
        Option<String>,
        (),
        SimpleComputeNodeWork<Input>,
        SimpleComputeNodeWork<Output>,
    >(addr)
    .await
    .expect("Could not connect to compute node")
}

pub async fn load_model(addr: impl ToSocketAddrs, model_name: impl AsRef<str>) {
    let start_time = std::time::Instant::now();

    let mut channel = connect_gpulet(addr).await;
    channel
        .send(&SimpleComputeNodeReq::command(
            SimpleComputeNodeCommand::Load(model_name.as_ref().to_string()),
        ))
        .await
        .expect("send load model to gpulet failed");
    let result = channel
        .recv()
        .await
        .expect("recv response from gpulet failed");
    if let Some(BinComputeNodeResponse::State(Some(state))) = result {
        if state == model_name.as_ref() {
            let interval = start_time.elapsed();
            println!("model loaded, using {}ms", interval.as_millis());
            return;
        }
        panic!("error! model not loaded");
    } else {
        panic!("error! gpulet returned {result:?}");
    }
}

pub async fn unload_model(addr: impl ToSocketAddrs) {
    let mut channel = connect_gpulet(addr).await;
    channel
        .send(&SimpleComputeNodeReq::command(
            SimpleComputeNodeCommand::UnLoad,
        ))
        .await
        .expect("send unload model to gpulet failed");
    let result = channel
        .recv()
        .await
        .expect("recv response from gpulet failed");
    if let Some(BinComputeNodeResponse::State(None)) = result {
    } else {
        panic!("error! gpulet returned {result:?}");
    }
}

pub async fn send_task_to_gpulet(addr: impl ToSocketAddrs, tensor: DynTensor) -> DynTensor {
    let instant = std::time::Instant::now();

    let mut channel = connect_gpulet(addr).await;
    channel
        .send(&SimpleComputeNodeReq::Compute(tensor.into()))
        .await
        .expect("send task to gpulet failed");
    let response = channel
        .recv()
        .await
        .ok()
        .flatten()
        .expect("recv response from gpulet failed");

    let result: DynTensor = match response {
        SimpleComputeNodeRes::Compute(work) => work.into(),
        frame => panic!("unexpected response from gpulet: {frame:?}"),
    };

    let interval = instant.elapsed();
    println!(
        "compute finished, result size: {:?}, using {} ms",
        result.dimensions(),
        interval.as_millis()
    );

    result
}

/// Normalize RGB values and put channels before spatial dimensions.
pub fn rgb_to_chw(image: image::RgbImage) -> ndarray::Array3<f32> {
    let (width, height) = image.dimensions();
    let values = image
        .into_raw()
        .into_iter()
        .map(|value| value as f32 / 255.0)
        .collect();
    ndarray::Array3::from_shape_vec((height as usize, width as usize, 3), values)
        .expect("RGB buffer matches its dimensions")
        .permuted_axes([2, 0, 1])
}

#[test]
fn image_channels_preserve_red_and_green_pixel_positions() {
    let image = image::RgbImage::from_raw(2, 1, vec![255, 0, 0, 0, 255, 0]).unwrap();
    let array = rgb_to_chw(image);
    assert_eq!(array.shape(), &[3, 1, 2]);
    assert_eq!(
        array.iter().copied().collect::<Vec<_>>(),
        vec![1., 0., 0., 1., 0., 0.]
    );
}
