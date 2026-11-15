use std::{fmt::Debug, io, path::Path};

use clap::Parser;
use frontend::{Cli, listen_input, load_model, rgb_to_chw, send_task_to_gpulet};
use image::ImageReader;
use model::tensor::DynTensor;
use ndarray::Axis;
use ort::{
    tensor::{IntoTensorElementType, PrimitiveTensorElementType},
    value::{DynTensor as OrtDynTensor, Tensor},
};
use tokio::sync::mpsc;

fn load_image_and_resize224<T: AsRef<Path>>(path: T) -> Result<OrtDynTensor, String> {
    let img = ImageReader::open(path)
        .map_err(|e| format!("Failed to open image: {e:?}"))?
        .decode()
        .map_err(|e| format!("Failed to decode image: {e:?}"))?;

    let img_resized = img.resize_exact(224, 224, image::imageops::FilterType::Lanczos3);

    let chw = rgb_to_chw(img_resized.into_rgb8());
    Ok(Tensor::from_array(chw)
        .expect("Cannot create target ort::Tensor")
        .upcast())
}

fn unsqueeze<T>(tensor: OrtDynTensor, dim: i64) -> OrtDynTensor
where
    T: Clone + Debug + PrimitiveTensorElementType + IntoTensorElementType + 'static,
{
    let data = tensor.try_extract_array::<T>().unwrap();
    let new_data = data.insert_axis(Axis(dim as usize));
    Tensor::from_array(new_data.to_owned()).unwrap().upcast()
}

fn repeat_first_dim<T>(tensor: &OrtDynTensor, repeats: usize) -> OrtDynTensor
where
    T: Clone + Debug + PrimitiveTensorElementType + IntoTensorElementType + 'static,
{
    let (shape, data) = tensor.try_extract_tensor::<T>().unwrap();
    let mut shape = shape.to_vec();
    shape[0] *= repeats as i64;

    let mut repeated_data: Vec<T> = Vec::new();
    for _ in 0..repeats {
        repeated_data.extend_from_slice(data);
    }

    Tensor::from_array((shape, repeated_data)).unwrap().upcast()
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> io::Result<()> {
    let cli = Cli::parse();
    let addr = (cli.ip.clone(), cli.port);
    let (tx, mut rx) = mpsc::channel(100);

    load_model(addr.clone(), cli.model_name).await;
    listen_input(tx);

    while let Some(input) = rx.recv().await {
        let tensor = unsqueeze::<f32>(load_image_and_resize224(input.trim()).unwrap(), 0);

        // 从路径读取图片为tensor
        let send_tensor = repeat_first_dim::<f32>(&tensor, cli.batch_size as usize);
        println!("tensor size: {:?}", tensor.shape());
        // 发送tensor到gpulet
        let _ =
            send_task_to_gpulet((cli.ip.as_str(), cli.port), DynTensor::from(send_tensor)).await;

        if cli.auto_loop {
            loop {
                // 从路径读取图片为tensor
                let send_tensor = repeat_first_dim::<f32>(&tensor, cli.batch_size as usize);
                println!("tensor size: {:?}", send_tensor.shape());
                // 发送tensor到gpulet
                let _ =
                    send_task_to_gpulet((cli.ip.as_str(), cli.port), DynTensor::from(send_tensor))
                        .await;
            }
        }
    }

    Ok(())
}
