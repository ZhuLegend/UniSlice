//! Pytorch模型

use std::{
    borrow::Borrow,
    path::{Path, PathBuf},
};

use model::{
    model::{InferenceModel, ModelError, ModelResult},
    tensor::{Device, TensorElementType},
};
use snafu::{ResultExt, ensure};
use tch::{CModule, Kind, TchError, Tensor};
use tokio::task::JoinHandle;

/// 通过通道在单独线程上运行的异步CModule
#[derive(Debug)]
pub struct AsyncCModule {
    /// 向线程管理的CModule提交任务的通道
    tx: tokio::sync::mpsc::Sender<(
        Tensor,
        tokio::sync::oneshot::Sender<Result<Tensor, tch::TchError>>,
    )>,
    /// 运行CModule的工作
    _worker: JoinHandle<()>,
}

impl AsyncCModule {
    /// 加载模型
    fn load(
        model_path: impl AsRef<Path>,
        device: tch::Device,
        kind: Kind,
        input_dim: &[i64],
        warm_up: bool,
    ) -> ModelResult<Self, TchError> {
        let mut model =
            CModule::load_on_device(model_path, device).context(model::model::InnerSnafu {})?;
        model.set_eval();

        let mut warm_up_dim = Vec::from(input_dim);
        warm_up_dim.insert(0, 1);
        if warm_up {
            let rand_input = if matches!(kind, Kind::Float | Kind::Double) {
                Tensor::f_rand(&warm_up_dim, (kind, device))
            } else {
                Tensor::f_zeros(&warm_up_dim, (kind, device))
            }
            .context(model::model::InnerSnafu {})?;
            let _ = model
                .forward_ts(&[rand_input])
                .context(model::model::InnerSnafu {})?;
        }

        let (tx, mut rx) = tokio::sync::mpsc::channel::<(
            Tensor,
            tokio::sync::oneshot::Sender<Result<Tensor, tch::TchError>>,
        )>(1);

        let _worker = tokio::task::spawn_blocking(move || {
            // tx 仅唯一保存在AsyncCModule中，当AsyncCModule被销毁时，tx也会被销毁，此时此处rx返回None，循环结束
            while let Some((input, result_tx)) = rx.blocking_recv() {
                let input = input.to_device(device);
                let output = model.forward_ts(&[input]);
                let _ = result_tx.send(output);
            }
        });

        Ok(AsyncCModule { tx, _worker })
    }

    /// 模型推理
    async fn forward(&self, input: Tensor) -> ModelResult<Tensor, TchError> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.tx
            .send((input, tx))
            .await
            .map_err(|e| ModelError::CommunicationError {
                message: format!("AsyncCModule send error: {e:?}"),
            })?;

        rx.await
            .map_err(|e| ModelError::CommunicationError {
                message: format!("AsyncCModule receive error: {e:?}"),
            })?
            .context(model::model::InnerSnafu {})
    }
}

/// Pytorch推理模型
#[derive(Debug)]
pub struct TorchModel {
    /// 底层Torch模型
    model: AsyncCModule,
    /// 输入维度
    input_dim: Vec<i64>,
    /// 数据类型
    kind: Kind,
}

/// 配置 Torch 模型加载时的各项参数
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
pub struct TorchModelLoadConfig {
    /// 模型文件的路径
    model_path: PathBuf,
    /// 设备信息 (例如 CPU 或 GPU)
    device: Device,
    /// 数据类型 (例如 float32, int64 等)
    dtype: TensorElementType,
    /// 输入维度
    input_dim: Vec<i64>,
    /// 是否进行预热操作
    warm_up: bool,
}

impl TorchModelLoadConfig {
    /// 创建一个配置对象
    pub fn new(
        model_path: PathBuf,
        device: Device,
        dtype: TensorElementType,
        input_dim: Vec<i64>,
        warm_up: bool,
    ) -> Self {
        TorchModelLoadConfig {
            model_path,
            device,
            dtype,
            input_dim,
            warm_up,
        }
    }
}

impl InferenceModel for TorchModel {
    type Model = AsyncCModule;

    type Input = Tensor;

    type Output = Tensor;

    type Error = TchError;

    type LoadConfig = TorchModelLoadConfig;

    /// 加载模型
    ///
    /// - model_path: 模型文件路径
    /// - device: 需要加载到的设备
    /// - kind: 模型的数据类型
    /// - input_dim: 输入维度，数字代表每个维度的长度，例如\[3, 244, 244\]代表输入维度应为3x244x244
    /// - output_dim: 输出维度，数字代表每个维度的长度，例如\[3, 244, 244\]代表输出维度应为3x244x244
    fn load(config: impl Into<Self::LoadConfig>) -> ModelResult<Self, Self::Error> {
        let TorchModelLoadConfig {
            model_path,
            device,
            dtype,
            input_dim,
            warm_up,
        } = config.into();
        ensure!(
            dtype.supports_common_io(),
            model::model::InvalidInputSnafu {
                message: "Dynamic Torch worker only supports common integer, float and bool types"
            }
        );
        let kind = dtype.into();
        let device = device.into();

        let model = AsyncCModule::load(model_path, device, kind, &input_dim, warm_up)?;

        Ok(TorchModel {
            model,
            input_dim,
            kind,
        })
    }

    fn model(&self) -> &Self::Model {
        &self.model
    }

    /// 检查输入
    ///
    /// 对于[TorchModel]，先检查输入类型是否正确，然后检查输入维度是否正确
    ///
    /// 默认输入的第一维是batch，仅校验后续其他维度是否匹配
    fn check_input(&self, input: impl Borrow<Self::Input>) -> ModelResult<(), Self::Error> {
        let input = input.borrow();
        ensure!(
            input.kind() == self.kind,
            model::model::InvalidInputSnafu {
                message: format!(
                    "Input kind should be {:?}, but got {:?}",
                    self.kind,
                    input.kind()
                )
            }
        );

        let shape = input.size();

        ensure!(
            shape[1..] == self.input_dim,
            model::model::InvalidInputSnafu {
                message: format!(
                    "Input dimension should be {:?}, but got {:?}",
                    self.input_dim,
                    shape[1..].to_vec()
                )
            }
        );

        Ok(())
    }

    /// 模型推理
    ///
    /// - input: 输入数据
    async fn forward(
        &self,
        input: impl Into<Self::Input> + Send,
    ) -> ModelResult<Self::Output, Self::Error> {
        let input = input.into();

        let output = self.model().forward(input).await?;
        ensure!(
            matches!(
                output.kind(),
                Kind::Uint8
                    | Kind::Int8
                    | Kind::Int16
                    | Kind::Int
                    | Kind::Int64
                    | Kind::Float
                    | Kind::Double
                    | Kind::Bool
            ),
            model::model::InvalidInputSnafu {
                message: "Unsupported dynamic Torch output dtype"
            }
        );
        Ok(output)
    }
}

#[cfg(test)]
#[tokio::test]
async fn supported_integer_and_boolean_models_warm_up_without_random_dtype_panics() {
    let path = std::env::temp_dir().join(format!("hsgs-warmup-{}.pt", std::process::id()));
    let input = Tensor::zeros([1, 2], (Kind::Float, tch::Device::Cpu));
    let module = CModule::create_by_tracing("Identity", "forward", &[input], &mut |inputs| {
        vec![inputs[0].shallow_clone()]
    })
    .unwrap();
    module.save(&path).unwrap();
    for dtype in [TensorElementType::Int64, TensorElementType::Bool] {
        let config = TorchModelLoadConfig::new(
            path.clone(),
            model::tensor::Device::Cpu,
            dtype,
            vec![2],
            true,
        );
        assert!(TorchModel::load(config).is_ok());
    }
    let config = TorchModelLoadConfig::new(
        path.clone(),
        model::tensor::Device::Cpu,
        TensorElementType::Half,
        vec![2],
        false,
    );
    assert!(matches!(TorchModel::load(config).unwrap_err(),
        ModelError::InvalidInput { message } if message.contains("only supports common")));
    std::fs::remove_file(path).unwrap();
}
