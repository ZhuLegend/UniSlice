//! 模型加载配置
#[cfg(any(feature = "onnx", feature = "torch", feature = "mock"))]
use std::path::PathBuf;

#[cfg(any(feature = "onnx", feature = "torch", feature = "mock"))]
use bincode::{Decode, Encode};
#[cfg(feature = "onnx")]
use ort::session::builder::GraphOptimizationLevel;
#[cfg(any(feature = "onnx", feature = "torch", feature = "mock"))]
use serde::{Deserialize, Serialize};
#[cfg(feature = "mock")]
use worker::mock::MockModelInitConfig;
#[cfg(any(feature = "onnx", feature = "torch"))]
use worker::model::tensor::TensorElementType;
#[cfg(feature = "onnx")]
use worker::onnx::model::{DynOnnxModelLoadConfig, OnnxModelLoadConfig};

#[cfg(any(feature = "onnx", feature = "torch", feature = "mock"))]
use crate::runnable_on::IndependentModelLoadConfig;

#[cfg(feature = "onnx")]
/// ONNX 配置
#[derive(Serialize, Deserialize, Encode, Decode)]
pub struct OnnxWorkerModelLoadConfig {
    /// 模型文件的路径
    pub model_path: PathBuf,

    /// 优化级别
    pub optimize_level: u8,
    /// 节点内并行线程数
    pub intra_thread: usize,

    /// 是否使用并行执行
    pub use_parallel_execution: bool,
    /// 并行执行图的线程数
    pub inter_thread: usize,

    /// 输入维度
    pub input_dim: Vec<i64>,

    /// 输入数据类型 (例如 Float32, Int64 等)
    pub input_type: TensorElementType,
    /// 输出数据类型 (例如 Float32, Int64 等)
    pub output_type: TensorElementType,
}

#[cfg(feature = "onnx")]
impl IndependentModelLoadConfig for OnnxWorkerModelLoadConfig {
    type RequireDevice = Vec<ort::execution_providers::ExecutionProviderDispatch>;
    type TargetLoadConfig = worker::onnx::model::DynOnnxModelLoadConfig;

    fn with_device(self, getter: impl FnOnce() -> Self::RequireDevice) -> Self::TargetLoadConfig {
        let OnnxWorkerModelLoadConfig {
            model_path,
            optimize_level,
            intra_thread,
            use_parallel_execution,
            inter_thread,
            input_dim,
            input_type,
            output_type,
        } = self;
        let optimize_level = match optimize_level {
            0 => GraphOptimizationLevel::Disable,
            1 => GraphOptimizationLevel::Level1,
            2 => GraphOptimizationLevel::Level2,
            3 => GraphOptimizationLevel::Level3,
            _ => panic!("Invalid optimize level"),
        };

        let onnx_model_load_config = OnnxModelLoadConfig::new(
            model_path,
            optimize_level,
            intra_thread,
            use_parallel_execution,
            inter_thread,
            getter(),
            input_dim,
        );

        DynOnnxModelLoadConfig::new(onnx_model_load_config, input_type, output_type)
    }
}

#[cfg(feature = "torch")]
/// Torch 配置
#[derive(Serialize, Deserialize, Encode, Decode)]
pub struct TorchWorkerModelLoadConfig {
    /// 模型文件的路径
    pub model_path: PathBuf,
    /// 数据类型 (例如 float32, int64 等)
    pub dtype: TensorElementType,
    /// 输入维度
    pub input_dim: Vec<i64>,
    /// 是否进行预热操作
    pub warm_up: bool,
}

#[cfg(feature = "torch")]
impl IndependentModelLoadConfig for TorchWorkerModelLoadConfig {
    type RequireDevice = worker::model::tensor::Device;
    type TargetLoadConfig = worker::torch::model::TorchModelLoadConfig;

    fn with_device(self, getter: impl FnOnce() -> Self::RequireDevice) -> Self::TargetLoadConfig {
        let TorchWorkerModelLoadConfig {
            model_path,
            dtype,
            input_dim,
            warm_up,
        } = self;

        worker::torch::model::TorchModelLoadConfig::new(
            model_path,
            getter(),
            dtype,
            input_dim,
            warm_up,
        )
    }
}

/// Mock configuration
#[cfg(feature = "mock")]
#[derive(Serialize, Deserialize, Encode, Decode)]
pub struct MockWorkerModelLoadConfig {
    /// Model name for loading
    pub model_name: String,
    /// Path to the latency profile file
    pub latency_profile_path: PathBuf,
}

#[cfg(feature = "mock")]
impl IndependentModelLoadConfig for MockWorkerModelLoadConfig {
    type RequireDevice = String;
    type TargetLoadConfig = MockModelInitConfig;

    fn with_device(self, getter: impl FnOnce() -> Self::RequireDevice) -> Self::TargetLoadConfig {
        let device = getter();
        MockModelInitConfig::new(self.model_name, device, self.latency_profile_path)
    }
}
