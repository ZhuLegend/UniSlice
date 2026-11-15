//! 推理模型相关定义

use std::borrow::Borrow;

use snafu::Snafu;

/// 模型操作结果
pub type ModelResult<T, E> = Result<T, ModelError<E>>;

/// 推理模型trait
///
/// **该trait只应该为与推理框架绑定的模型实现**
///
/// 例如单纯的预训练`vgg16`模型不应该实现此trait，因为其只表示具有训练好参数的模型，
/// 但是使用`torch`或`ONNX`加载后的模型应该实现此trait，因为后者已被加载，可以进行推理
#[trait_variant::make(Send)]
pub trait InferenceModel: Sized {
    /// 预训练的推理模型
    type Model;

    /// 可以接受的输入类型
    type Input: Send;

    /// 推理后输出类型
    type Output;

    /// 错误类型
    type Error: std::error::Error + 'static;

    /// 加载配置
    type LoadConfig;

    /// 加载模型
    fn load(config: impl Into<Self::LoadConfig>) -> ModelResult<Self, Self::Error>;

    /// 内部的预训练模型信息
    fn model(&self) -> &Self::Model;

    /// 检查输入数据是否正确
    ///
    /// 通常执行数据shape检查、类型检查等
    fn check_input(&self, input: impl Borrow<Self::Input>) -> ModelResult<(), Self::Error>;

    /// 执行推理
    ///
    /// 如果框架支持异步推理，应当充分利用异步性
    async fn forward(
        &self,
        input: impl Into<Self::Input> + Send,
    ) -> ModelResult<Self::Output, Self::Error>;
}

/// 推理模型的错误类型
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum ModelError<E: std::error::Error + 'static> {
    /// 通信错误
    CommunicationError {
        /// 错误信息
        message: String,
    },

    /// 模型加载配置错误
    LoadConfigError {
        /// 错误信息
        message: String,
    },

    /// 输入数据错误
    InvalidInput {
        /// 错误信息
        message: String,
    },

    /// 无法获取输出
    CannotGetOutput {
        /// 错误信息
        message: String,
    },

    /// 框架内部错误
    InnerError {
        /// 错误信息
        source: E,
    },
}
