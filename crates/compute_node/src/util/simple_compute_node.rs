//! 仅支持单Worker独立运行的[ComputeNode]

use std::marker::PhantomData;

use bincode::{Decode, Encode};
use tokio::sync::{Mutex, RwLock};
use tracing::{error, trace};
use worker::model::{model::InferenceModel, tensor::DynTensor};

use crate::{
    ComputeNode,
    util::{
        Input, Output,
        bin_compute_node::{BinComputeNode, BinComputeNodeRequest, BinComputeNodeResponse},
        sealed::WorkState,
    },
};

/// 仅持有一个[Worker](worker::Worker)的[ComputeNode]
pub struct SimpleComputeNode<Worker, ModelProvider>
where
    Worker: worker::Worker,
    ModelProvider: Fn(&str) -> <Worker::Model as InferenceModel>::LoadConfig + Send + Sync,
{
    worker: RwLock<Worker>,
    model_provider: ModelProvider,
    model_name: Mutex<Option<String>>,
}

impl<Worker, ModelProvider> SimpleComputeNode<Worker, ModelProvider>
where
    Worker: worker::Worker,
    ModelProvider: Fn(&str) -> <Worker::Model as InferenceModel>::LoadConfig + Send + Sync,
{
    /// 创建一个新的[SimpleComputeNode]
    ///
    /// # Parameters
    /// - `worker`: [Worker](worker::Worker)实例
    /// - `model_provider`: 模型提供者，接受一个模型名称，返回模型加载配置
    pub fn new(worker: Worker, model_provider: ModelProvider) -> Self {
        SimpleComputeNode {
            worker: RwLock::new(worker),
            model_provider,
            model_name: Mutex::new(None),
        }
    }
}

/// [SimpleComputeNode]可以执行的的命令
#[derive(Debug, Encode, Decode)]
pub enum SimpleComputeNodeCommand {
    /// 加载模型（模型名称）
    Load(String),
    /// 卸载模型
    UnLoad,
}

/// [SimpleComputeNode]可以执行的推理工作
#[derive(Debug, Encode, Decode)]
pub struct SimpleComputeNodeWork<State: WorkState> {
    data: DynTensor,
    state: PhantomData<fn() -> State>,
}

impl SimpleComputeNodeWork<Input> {
    /// 创建一个新的推理工作
    pub fn new(input: DynTensor) -> Self {
        SimpleComputeNodeWork {
            data: input,
            state: PhantomData,
        }
    }
}

impl From<DynTensor> for SimpleComputeNodeWork<Input> {
    fn from(value: DynTensor) -> Self {
        Self::new(value)
    }
}

impl From<SimpleComputeNodeWork<Input>> for DynTensor {
    fn from(value: SimpleComputeNodeWork<Input>) -> Self {
        value.data
    }
}

impl From<SimpleComputeNodeWork<Output>> for DynTensor {
    fn from(value: SimpleComputeNodeWork<Output>) -> Self {
        value.data
    }
}

impl From<DynTensor> for SimpleComputeNodeWork<Output> {
    fn from(value: DynTensor) -> Self {
        Self {
            data: value,
            state: PhantomData,
        }
    }
}

impl<Worker, ModelProvider> ComputeNode for SimpleComputeNode<Worker, ModelProvider>
where
    Worker: worker::Worker,
    ModelProvider: Fn(&str) -> <Worker::Model as InferenceModel>::LoadConfig + Send + Sync,
{
    type State = Option<String>;
    type Status = ();
    type Command = SimpleComputeNodeCommand;

    type InputData = SimpleComputeNodeWork<Input>;

    type OutputData = SimpleComputeNodeWork<Output>;

    async fn handle(&self, command: Self::Command) -> Result<Self::State, String> {
        trace!("command: {command:?}");
        let mut model_name = self.model_name.lock().await;
        match command {
            SimpleComputeNodeCommand::Load(name) => {
                if model_name.as_ref() == Some(&name) {
                    return Ok(Some(name));
                }
                {
                    let mut worker = self.worker.write().await;
                    if model_name.is_some() {
                        worker.unload_model().await.map_err(|e| format!("{e:?}"))?;
                        model_name.take();
                    }
                    worker
                        .load_model((self.model_provider)(&name))
                        .await
                        .map_err(|e| {
                            let res = format!("{e:?}");
                            error!("Failed to load model: {res}");
                            res
                        })?;
                }
                {
                    model_name.replace(name.clone());
                }
                trace!("model loaded: {name}");
                Ok(Some(name))
            }
            SimpleComputeNodeCommand::UnLoad => {
                if model_name.is_none() {
                    return Ok(None);
                }
                {
                    let mut worker = self.worker.write().await;
                    worker.unload_model().await.map_err(|e| {
                        let res = format!("{e:?}");
                        error!("Failed to unload model: {res}");
                        res
                    })?;
                }
                {
                    model_name.take();
                }
                trace!("model unloaded");
                Ok(None)
            }
        }
    }

    async fn state(&self) -> Result<Self::State, String> {
        Ok(self.model_name.lock().await.clone())
    }

    async fn status(&self) -> Result<Self::Status, String> {
        Ok(())
    }

    async fn compute(&self, work: Self::InputData) -> Result<Self::OutputData, String> {
        trace!("receive work");
        let input = work.into();

        let output = {
            let worker = self.worker.read().await;
            worker.execute(input).await
        }
        .map_err(|e| {
            error!("compute error: {e:?}");
            format!("{e:?}")
        })?;

        Ok(output.into())
    }
}

impl<Worker, ModelProvider> BinComputeNode for SimpleComputeNode<Worker, ModelProvider>
where
    Worker: worker::Worker + 'static,
    ModelProvider:
        Fn(&str) -> <Worker::Model as InferenceModel>::LoadConfig + Send + Sync + 'static,
{
    async fn shutdown(&self) -> Result<(), String> {
        self.handle(SimpleComputeNodeCommand::UnLoad)
            .await
            .map(|_| ())
    }
}

/// [BinComputeNode]包裹[SimpleComputeNode]时的请求类型
pub type BinSimpleComputeNodeRequest =
    BinComputeNodeRequest<SimpleComputeNodeCommand, SimpleComputeNodeWork<Input>>;

/// [BinComputeNode]包裹[SimpleComputeNode]时的响应类型
pub type BinSimpleComputeNodeResponse =
    BinComputeNodeResponse<Option<String>, (), SimpleComputeNodeWork<Output>>;
