//! 仅具有加/卸载模型和执行推理功能的[Worker]

#![warn(missing_docs)]

use std::{fmt::Debug, ops::Deref};

// reexport `model`模块
pub use model;
use model::{
    model::{InferenceModel, ModelError},
    tensor::DynTensor,
};
use snafu::Snafu;

#[cfg(feature = "onnx")]
pub mod onnx;

#[cfg(feature = "torch")]
pub mod torch;

#[cfg(feature = "mock")]
pub mod mock;

/// 可以加卸载模型并执行推理工作的`Worker`抽象
#[trait_variant::make(Send)]
pub trait Worker: Sync {
    /// 加卸载和执行推理的模型
    type Model: InferenceModel<Input: From<DynTensor>, Output: Into<DynTensor>>;

    /// 模型的不可变引用
    type ModelRef: Deref<Target = Self::Model> + Send;

    /// 如模型已加载，获取模型的不可变引用
    async fn model(&self) -> Option<Self::ModelRef>;

    /// 加载模型，如果已有被加载的模型，应当卸载之前的模型后重新加载模型
    async fn load_model(
        &mut self,
        load_config: <Self::Model as InferenceModel>::LoadConfig,
    ) -> WorkerResult<Self::ModelRef, <Self::Model as InferenceModel>::Error>;

    /// 卸载模型
    ///
    /// 卸载后，会等待所有持有ModelRef的所有引用失效后释放模型，因此不保证立刻释放模型，但是保证不会新增引用
    async fn unload_model(
        &mut self,
    ) -> WorkerResult<Self::ModelRef, <Self::Model as InferenceModel>::Error>;

    /// 对输入进行检查，然后在模型上进行推理
    async fn execute(
        &self,
        input: DynTensor,
    ) -> WorkerResult<DynTensor, <Self::Model as InferenceModel>::Error> {
        async move {
            let model = self.model().await.ok_or(WorkerError::ModelNotLoaded)?;

            // 将input放在单独的块内，避免编译器抱怨`<Self::Model as InferenceModel>::Input`不是`Send`的
            // `forward`函数获取了input的所有权，后续不会使用，不应当考虑使用await后，input会在线程间转移的问题
            let f = {
                let input: <Self::Model as InferenceModel>::Input = input.into();
                model.check_input(&input)?;
                model.forward(input)
            };

            Ok(f.await.map(|output| output.into())?)
        }
    }
}

/// `Worker`的错误类型
pub type WorkerResult<T, E> = Result<T, WorkerError<E>>;

/// [Worker]可选错误类型
#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum WorkerError<E: std::error::Error + 'static> {
    /// 模型未加载
    #[snafu(display("Model is not loaded, please load the model first"))]
    ModelNotLoaded,

    /// 内部模型错误
    #[snafu(transparent)]
    InnerError {
        /// 错误源
        source: ModelError<E>,
    },
}

#[cfg(feature = "torch")]
unsafe extern "C" {
    /// Dummy function to get CUDA to link properly
    pub fn dummy_cuda_dependency();
}

#[used]
#[cfg(feature = "torch")]
static INIT_ARRAY: [unsafe extern "C" fn(); 1] = [dummy_cuda_dependency];
