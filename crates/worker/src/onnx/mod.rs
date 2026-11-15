//! ONNX Runtime 支持

use std::fmt::Debug;

use ::model::tensor::IntoTensorElementType;
use ort::tensor::PrimitiveTensorElementType;

pub mod model;

pub mod worker;

/// ONNX 模型支持的类型
pub trait OnnxWorkerValue:
    PrimitiveTensorElementType + IntoTensorElementType + Send + Sync + Debug + Clone
{
}
