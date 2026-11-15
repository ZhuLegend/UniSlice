//! ONNX model

use std::{
    borrow::Borrow,
    marker::PhantomData,
    path::{Path, PathBuf},
};

use model::{
    model::{InferenceModel, ModelResult},
    tensor::{DynTensor, Tensor, TensorElementType},
};
use ort::{
    execution_providers::ExecutionProviderDispatch,
    inputs,
    session::{RunOptions, Session, builder::GraphOptimizationLevel},
    value::DynTensorValueType,
};
use snafu::{OptionExt, ResultExt, ensure};
use tokio::sync::Mutex;

use crate::onnx::OnnxWorkerValue;

#[inline(always)]
fn build_ort_session<P: AsRef<Path>>(
    model_path: P,
    optimize_level: GraphOptimizationLevel,
    intra_thread: usize,
    use_parallel_execution: bool,
    inter_thread: usize,
    execution_provider: Vec<ExecutionProviderDispatch>,
) -> Result<Session, ort::Error> {
    Session::builder()?
        .with_optimization_level(optimize_level)?
        .with_intra_threads(intra_thread)?
        .with_inter_threads(inter_thread)?
        .with_parallel_execution(use_parallel_execution)?
        .with_execution_providers(execution_provider)?
        .with_memory_pattern(false)?
        .commit_from_file(model_path)
}

#[inline(always)]
async fn forward_async(
    session_lock: &Mutex<Session>,
    input: impl Into<ort::value::DynTensor>,
) -> ModelResult<ort::value::DynTensor, ort::Error> {
    let input: ort::value::DynTensor = input.into();
    let input = inputs![input];
    let options = RunOptions::new().context(model::model::InnerSnafu {})?;

    let output = {
        let mut session = session_lock.lock().await;
        let mut output = session
            .run_async(input, &options)
            .context(model::model::InnerSnafu {})?
            .await
            .context(model::model::InnerSnafu {})?;
        let key = output
            .keys()
            .next()
            .context(model::model::CannotGetOutputSnafu {
                message: "No output key founded",
            })?;
        output
            .remove(key)
            .context(model::model::CannotGetOutputSnafu {
                message: format!("Can't get output with {key}"),
            })?
    };

    output
        .downcast::<DynTensorValueType>()
        .context(model::model::InnerSnafu {})
}

/// Onnx推理模型
#[derive(Debug)]
pub struct OnnxModel<InputType: OnnxWorkerValue, OutputType: OnnxWorkerValue> {
    session: Mutex<Session>,
    input_dim: Vec<i64>,
    input_type: PhantomData<InputType>,
    output_type: PhantomData<OutputType>,
}

/// Onnx模型加载配置
#[derive(Debug)]
pub struct OnnxModelLoadConfig {
    model_path: PathBuf,

    optimize_level: GraphOptimizationLevel,
    intra_thread: usize,

    use_parallel_execution: bool,
    inter_thread: usize,

    execution_provider: Vec<ExecutionProviderDispatch>,

    input_dim: Vec<i64>,
}

impl OnnxModelLoadConfig {
    /// 创建Onnx模型加载配置
    pub fn new(
        model_path: PathBuf,
        optimize_level: GraphOptimizationLevel,
        intra_thread: usize,
        use_parallel_execution: bool,
        inter_thread: usize,
        execution_provider: Vec<ExecutionProviderDispatch>,
        input_dim: Vec<i64>,
    ) -> Self {
        OnnxModelLoadConfig {
            model_path,
            optimize_level,
            intra_thread,
            use_parallel_execution,
            inter_thread,
            execution_provider,
            input_dim,
        }
    }
}

impl<InputType: OnnxWorkerValue, OutputType: OnnxWorkerValue> InferenceModel
    for OnnxModel<InputType, OutputType>
{
    type Model = Mutex<Session>;
    type Input = Tensor<InputType>;

    type Output = Tensor<OutputType>;

    type Error = ort::Error;

    type LoadConfig = OnnxModelLoadConfig;

    fn load(config: impl Into<Self::LoadConfig>) -> ModelResult<Self, Self::Error> {
        let Self::LoadConfig {
            optimize_level,
            intra_thread,
            use_parallel_execution,
            inter_thread,
            execution_provider,
            model_path,
            input_dim,
        } = config.into();

        let session = build_ort_session(
            model_path,
            optimize_level,
            intra_thread,
            use_parallel_execution,
            inter_thread,
            execution_provider,
        )
        .context(model::model::InnerSnafu {})?;

        Ok(OnnxModel {
            session: Mutex::new(session),
            input_dim,
            input_type: PhantomData,
            output_type: PhantomData,
        })
    }

    fn model(&self) -> &Self::Model {
        &self.session
    }

    /// 检查输入值是否为[ort::value::Tensor]，若是，进一步检查数据类型和除第一维外的其他维度是否符合预期
    fn check_input(&self, input: impl Borrow<Self::Input>) -> ModelResult<(), Self::Error> {
        let input = input.borrow();
        let dimensions = input.dimensions();
        ensure!(
            dimensions[1..] == self.input_dim,
            model::model::InvalidInputSnafu {
                message: format!(
                    "Input dimension shape should be {:?}, but got {:?}",
                    self.input_dim, dimensions
                )
            }
        );

        Ok(())
    }

    async fn forward(
        &self,
        input: impl Into<Self::Input> + Send,
    ) -> ModelResult<Self::Output, Self::Error> {
        forward_async(&self.session, input.into())
            .await
            .map(|tensor| tensor.into())
    }
}

/// 类型擦除后的`OnnxModel`
#[derive(Debug)]
pub struct DynOnnxModel {
    session: Mutex<Session>,
    input_dim: Vec<i64>,
    input_type: TensorElementType,
    output_type: TensorElementType,
}

impl<InputType: OnnxWorkerValue, OutputType: OnnxWorkerValue> OnnxModel<InputType, OutputType> {
    /// 将类型擦除后的`OnnxModel`转换为指定的类型
    #[inline]
    pub fn into_dyn(self) -> DynOnnxModel {
        DynOnnxModel {
            session: self.session,
            input_dim: self.input_dim,
            input_type:
                <InputType as model::tensor::IntoTensorElementType>::into_tensor_element_type(),
            output_type:
                <OutputType as model::tensor::IntoTensorElementType>::into_tensor_element_type(),
        }
    }
}

impl DynOnnxModel {
    /// 判断是否可以将`DynOnnxModel`转换为指定的类型
    #[inline]
    pub fn can_downcast<InputType: OnnxWorkerValue, OutputType: OnnxWorkerValue>(&self) -> bool {
        self.input_type
            == <InputType as model::tensor::IntoTensorElementType>::into_tensor_element_type()
            && self.output_type
                == <OutputType as model::tensor::IntoTensorElementType>::into_tensor_element_type()
    }

    /// 尝试将类型擦除后的`OnnxModel`转换为指定的类型
    #[inline]
    pub fn downcast<InputType: OnnxWorkerValue, OutputType: OnnxWorkerValue>(
        self,
    ) -> Option<OnnxModel<InputType, OutputType>> {
        if self.can_downcast::<InputType, OutputType>() {
            Some(OnnxModel {
                session: self.session,
                input_dim: self.input_dim,
                input_type: PhantomData,
                output_type: PhantomData,
            })
        } else {
            None
        }
    }
}

/// 类型擦除后的`OnnxModel`加载配置
#[derive(Debug)]
pub struct DynOnnxModelLoadConfig {
    config: OnnxModelLoadConfig,
    input_type: TensorElementType,
    output_type: TensorElementType,
}

impl DynOnnxModelLoadConfig {
    /// 创建类型擦除后的`OnnxModel`加载配置
    pub fn new(
        config: OnnxModelLoadConfig,
        input_type: TensorElementType,
        output_type: TensorElementType,
    ) -> Self {
        DynOnnxModelLoadConfig {
            config,
            input_type,
            output_type,
        }
    }
}

impl InferenceModel for DynOnnxModel {
    type Model = Mutex<Session>;
    type Input = DynTensor;
    type Output = DynTensor;
    type Error = ort::Error;
    type LoadConfig = DynOnnxModelLoadConfig;

    fn load(config: impl Into<Self::LoadConfig>) -> ModelResult<Self, Self::Error> {
        let DynOnnxModelLoadConfig {
            config:
                OnnxModelLoadConfig {
                    model_path,
                    optimize_level,
                    intra_thread,
                    use_parallel_execution,
                    inter_thread,
                    execution_provider,
                    input_dim,
                },
            input_type,
            output_type,
        } = config.into();

        let supported = |dtype: &TensorElementType| {
            dtype.supports_common_io()
                || matches!(
                    dtype,
                    TensorElementType::UInt16 | TensorElementType::UInt | TensorElementType::UInt64
                )
        };
        ensure!(
            supported(&input_type) && supported(&output_type),
            model::model::InvalidInputSnafu {
                message: "Dynamic ONNX worker only supports integer, float and bool types"
            }
        );

        let session = build_ort_session(
            model_path,
            optimize_level,
            intra_thread,
            use_parallel_execution,
            inter_thread,
            execution_provider,
        )
        .context(model::model::InnerSnafu {})?;

        ensure!(
            matches!(session.inputs.first().map(|input| &input.input_type),
            Some(ort::value::ValueType::Tensor { ty, .. }) if *ty == input_type.clone().into())
                && matches!(session.outputs.first().map(|output| &output.output_type),
            Some(ort::value::ValueType::Tensor { ty, .. }) if *ty == output_type.clone().into()),
            model::model::InvalidInputSnafu {
                message: "Configured dynamic ONNX dtypes must match model metadata"
            }
        );

        Ok(DynOnnxModel {
            session: Mutex::new(session),
            input_dim,
            input_type,
            output_type,
        })
    }

    fn model(&self) -> &Self::Model {
        &self.session
    }

    fn check_input(&self, input: impl Borrow<Self::Input>) -> ModelResult<(), Self::Error> {
        let input = input.borrow();
        let dtype = input.dtype();
        let dimensions = input.dimensions();

        ensure!(
            dtype == self.input_type,
            model::model::InvalidInputSnafu {
                message: format!(
                    "Input kind should be {:?}, but got {:?}",
                    self.input_type, dtype
                )
            }
        );
        ensure!(
            dimensions[1..] == self.input_dim,
            model::model::InvalidInputSnafu {
                message: format!(
                    "Input dimension shape should be {:?}, but got {:?}",
                    self.input_dim, dimensions
                )
            }
        );

        Ok(())
    }

    async fn forward(
        &self,
        input: impl Into<Self::Input> + Send,
    ) -> ModelResult<Self::Output, Self::Error> {
        forward_async(&self.session, input.into())
            .await
            .map(|tensor| tensor.into())
    }
}

#[cfg(test)]
#[test]
fn unsupported_dynamic_dtype_is_rejected_before_opening_the_model() {
    let config = OnnxModelLoadConfig::new(
        "missing-model.onnx".into(),
        GraphOptimizationLevel::Level1,
        1,
        false,
        1,
        vec![],
        vec![2],
    );
    let config =
        DynOnnxModelLoadConfig::new(config, TensorElementType::Half, TensorElementType::Float);
    assert!(matches!(DynOnnxModel::load(config).unwrap_err(),
        model::model::ModelError::InvalidInput { message } if message.contains("only supports integer")));
}
