use std::io;

use clap::Parser;
use frontend::{Cli, listen_input, load_model, send_task_to_gpulet};
use model::tensor::DynTensor;
use tch::vision::imagenet::load_image_and_resize224;
use tokio::sync::mpsc;

#[tokio::main(flavor = "current_thread")]
async fn main() -> io::Result<()> {
    let cli = Cli::parse();
    let addr = (cli.ip.clone(), cli.port);
    let (tx, mut rx) = mpsc::channel(100);

    load_model(addr.clone(), cli.model_name).await;
    listen_input(tx);

    while let Some(input) = rx.recv().await {
        let tensor = load_image_and_resize224(input.trim()).unwrap().unsqueeze(0);

        let shape_len = tensor.size().len();
        let mut repeat_dims = Vec::with_capacity(shape_len);
        repeat_dims.push(cli.batch_size); // 第一个维度复制 num_copies 次
        repeat_dims.extend(std::iter::repeat_n(1, shape_len - 1));

        // 从路径读取图片为tensor
        let send_tensor = tensor.repeat(repeat_dims.clone());
        println!("tensor size: {:?}", tensor.size());
        // 发送tensor到gpulet
        let _ =
            send_task_to_gpulet((cli.ip.as_str(), cli.port), DynTensor::from(send_tensor)).await;

        if cli.auto_loop {
            loop {
                // 从路径读取图片为tensor
                let tensor = tensor.repeat(repeat_dims.clone());
                println!("tensor size: {:?}", tensor.size());
                // 发送tensor到gpulet
                let _ =
                    send_task_to_gpulet((cli.ip.as_str(), cli.port), DynTensor::from(tensor)).await;
            }
        }
    }

    Ok(())
}
