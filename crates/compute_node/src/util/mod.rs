//! 实用工具

use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};

pub mod bin_compute_node;

pub mod worker_proxy;

pub mod simple_compute_node;

/// 模型配置
pub mod model_config {
    use serde::{Deserialize, Serialize};
    use worker::model::tensor::TensorElementType;

    /// Pytorch 模型配置
    #[derive(Debug, Serialize, Deserialize, Clone)]
    pub struct TorchModelConfig {
        /// 是否进行预热操作
        pub warm_up: bool,
    }

    /// ONNX 模型配置
    #[derive(Debug, Serialize, Deserialize, Clone)]
    pub struct OnnxModelConfig {
        /// 优化级别
        ///
        /// 应在0～3之间
        pub optimize_level: u8,
        /// 节点内并行线程数
        pub intra_thread: usize,
        /// 是否使用并行执行
        ///
        /// 启用时，可以提高多分支模型的性能
        pub use_parallel_execution: bool,
        /// 并行执行图的线程数
        ///
        /// 仅当`use_parallel_execution`为`true`时有效
        pub inter_thread: usize,
    }

    /// 模型配置
    #[derive(Debug, Serialize, Deserialize, Clone)]
    pub struct ModelConfig {
        /// 输入维度
        pub input_dim: Vec<i64>,
        /// 输入数据类型 (例如 Float32, Int64 等)
        pub input_type: TensorElementType,
        /// 输出数据类型 (例如 Float32, Int64 等)
        pub output_type: TensorElementType,
        /// Pytorch 模型配置
        #[serde(default)]
        pub torch_model_config: Option<TorchModelConfig>,
        /// ONNX 模型配置
        #[serde(default)]
        pub onnx_model_config: Option<OnnxModelConfig>,
    }
}

mod sealed {
    use bincode::{Decode, Encode};
    use serde::{Serialize, de::DeserializeOwned};

    pub trait WorkState: Serialize + DeserializeOwned + Encode + Decode<()> {}
}

pub(crate) use sealed::WorkState;

#[derive(Debug, Serialize, Deserialize, Encode, Decode, Copy, Clone)]
/// [Input] state for [crate::ComputeNode::InputData]
pub struct Input;

impl WorkState for Input {}

#[derive(Debug, Serialize, Deserialize, Encode, Decode, Copy, Clone)]
/// [Output] state for [crate::ComputeNode::OutputData]
pub struct Output;

impl WorkState for Output {}
