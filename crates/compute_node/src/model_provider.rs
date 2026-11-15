//! 通过模型名称提供模型加载配置

use std::{collections::HashMap, path::PathBuf, sync::Arc};

use utils::fs::DirPathBuf;
#[cfg(feature = "onnx")]
use worker::onnx::model::DynOnnxModelLoadConfig;
#[cfg(feature = "onnx")]
use worker::onnx::worker::OnnxWorker;
#[cfg(feature = "torch")]
use worker::torch::model::TorchModelLoadConfig;
#[cfg(feature = "torch")]
use worker::torch::worker::TorchWorker;

#[cfg(feature = "onnx")]
use crate::compute_device::config::OnnxWorkerModelLoadConfig;
#[cfg(feature = "torch")]
use crate::compute_device::config::TorchWorkerModelLoadConfig;
use crate::{
    compute_device::ComputeDevice, runnable_on::RunnableOn, util::model_config::ModelConfig,
};

/// 模型配置提供者
pub struct ModelProvider<T: ComputeDevice> {
    model_path_dir: DirPathBuf,
    model_configs: HashMap<String, ModelConfig>,
    device: Arc<T>,
}

impl<T: ComputeDevice> ModelProvider<T> {
    /// 创建一个模型配置提供者
    pub fn new(model_path_dir: DirPathBuf, model_config_dir: DirPathBuf, device: T) -> Self {
        let model_config_dir: PathBuf = model_config_dir.into();
        let mut model_configs = HashMap::new();

        for entry in model_config_dir
            .read_dir()
            .expect("read model config dir failed")
            .flatten()
        {
            let path = entry.path();
            if path.is_file() && path.extension().map(|ext| ext == "json").unwrap_or(false) {
                let model_name = match path.file_stem() {
                    None => continue,
                    Some(file_name) => file_name.to_str().unwrap(),
                };
                let model_config = std::fs::read_to_string(&path)
                    .unwrap_or_else(|_| panic!("read model config failed: {path:?}"));
                let model_config: ModelConfig = serde_json::from_str(&model_config)
                    .unwrap_or_else(|_| panic!("parse model config failed: {path:?}"));

                model_configs.insert(model_name.to_string(), model_config);
            }
        }

        ModelProvider {
            model_path_dir,
            model_configs,
            device: Arc::new(device),
        }
    }
}

#[cfg(feature = "torch")]
impl<T: ComputeDevice + Sync + 'static> ModelProvider<T>
where
    TorchWorker: RunnableOn<T>,
    <TorchWorker as RunnableOn<T>>::DeviceIndependentConfig: From<TorchWorkerModelLoadConfig>,
{
    /// 提供Torch模型加载配置
    pub fn torch_provider(&self) -> impl Fn(&str) -> TorchModelLoadConfig + Send + Sync + 'static {
        let device = self.device.clone();
        let model_path: PathBuf = self.model_path_dir.clone().into();
        let model_config = self.model_configs.clone();

        move |model_name| {
            let mut model_path = model_path.clone();
            model_path.push(format!("{model_name}.pt"));

            let config = model_config.get(model_name).unwrap();
            assert_eq!(config.input_type, config.output_type);
            assert!(config.torch_model_config.is_some());

            let config = TorchWorkerModelLoadConfig {
                model_path,
                dtype: config.input_type.clone(),
                input_dim: config.input_dim.clone(),
                warm_up: config.torch_model_config.as_ref().unwrap().warm_up,
            };

            <TorchWorker as RunnableOn<T>>::generate_config(device.as_ref(), config.into()).into()
        }
    }
}
#[cfg(feature = "onnx")]
impl<T: ComputeDevice + Sync + 'static> ModelProvider<T>
where
    OnnxWorker: RunnableOn<T>,
    <OnnxWorker as RunnableOn<T>>::DeviceIndependentConfig: From<OnnxWorkerModelLoadConfig>,
{
    /// 提供ONNX模型加载配置
    pub fn onnx_provider(&self) -> impl Fn(&str) -> DynOnnxModelLoadConfig + Send + Sync + 'static {
        let device = self.device.clone();
        let model_path: PathBuf = self.model_path_dir.clone().into();
        let model_config = self.model_configs.clone();

        move |model_name| {
            let mut model_path = model_path.clone();
            model_path.push(format!("{model_name}.onnx"));

            let config = model_config.get(model_name).unwrap();
            assert!(config.onnx_model_config.is_some());
            let onnx_model_config = config.onnx_model_config.as_ref().unwrap();

            let config = OnnxWorkerModelLoadConfig {
                model_path,
                optimize_level: onnx_model_config.optimize_level,
                intra_thread: onnx_model_config.intra_thread,
                use_parallel_execution: onnx_model_config.use_parallel_execution,
                inter_thread: onnx_model_config.inter_thread,
                input_dim: config.input_dim.clone(),
                input_type: config.input_type.clone(),
                output_type: config.output_type.clone(),
            };

            <OnnxWorker as RunnableOn<T>>::generate_config(device.as_ref(), config.into()).into()
        }
    }
}
