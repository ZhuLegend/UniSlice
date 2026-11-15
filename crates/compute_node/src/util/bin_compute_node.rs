//! 可作为二进制文件运行的计算节点
//!
//! 对[ComputeNode]添加进一步约束和包装，提供[BinComputeNode]和[BinComputeNodeWrapper]，后者实现了[Listener]，从而运行直接监听本地端口运行

use std::{fmt::Debug, future::Future, sync::Arc, time::Duration};

use bincode::{Decode, Encode};
use bon::bon;
use comm::{
    ChannelCreator, CommError, Listener, SimpleChannelCreator, TypedConnection,
    service::{ChunkLayer, Service},
};
use serde::{Deserialize, Serialize};
use tokio::net::{TcpListener, ToSocketAddrs};
use tracing::warn;
use utils::sync::Task;

use crate::ComputeNode;

/// 可独立执行的计算节点
pub trait BinComputeNode:
    ComputeNode<
        State: Encode + Decode<()>,
        Status: Encode + Decode<()>,
        Command: Encode + Decode<()>,
        InputData: Encode + Decode<()>,
        OutputData: Encode + Decode<()>,
    > + Sync
    + 'static
{
    /// Await model/worker cleanup after incoming requests have drained.
    fn shutdown(&self) -> impl Future<Output = Result<(), String>> + Send;
}

/// connect to a bin compute node
pub async fn connect_bin_compute_node<Command, State, Status, InputData, OutputData>(
    addr: impl ToSocketAddrs,
) -> Result<
    TypedConnection<
        impl Service,
        BinComputeNodeRequest<Command, InputData>,
        BinComputeNodeResponse<State, Status, OutputData>,
    >,
    CommError,
>
where
    Command: Encode + Decode<()> + Send,
    State: Encode + Decode<()> + Send + 'static,
    Status: Encode + Decode<()> + Send + 'static,
    InputData: Encode + Decode<()> + Send + 'static,
    OutputData: Encode + Decode<()> + Send + 'static,
{
    Ok(TypedConnection::connect(addr)
        .await?
        .layer(ChunkLayer::default()))
}

/// 计算节点请求
///
/// - [BinComputeNodeRequest::State]返回内部计算节点的可配置状态
/// - [BinComputeNodeRequest::Status]返回内部计算节点的不可配置状态
/// - [BinComputeNodeRequest::Command]向内部计算节点发送命令
/// - [BinComputeNodeRequest::Compute]向内部计算节点发送计算任务
#[derive(Serialize, Deserialize, Encode, Decode, Debug)]
pub enum BinComputeNodeRequest<Command: Encode + Decode<()>, Work: Encode + Decode<()>> {
    /// 获取可配置状态
    State,
    /// 获取运行时状态（不可配置）
    Status,
    /// 执行命令
    Command(Command),
    /// 执行推理
    Compute(Work),
}

impl<Command: Encode + Decode<()>, Work: Encode + Decode<()>> BinComputeNodeRequest<Command, Work> {
    /// init command request
    pub fn command(cmd: Command) -> Self {
        BinComputeNodeRequest::Command(cmd)
    }

    /// init compute request
    pub fn compute(work: Work) -> Self {
        BinComputeNodeRequest::Compute(work)
    }
}

/// 计算节点响应
///
/// - [BinComputeNodeResponse::State]内部计算节点的状态
/// - [BinComputeNodeResponse::Compute]内部计算节点计算任务的执行结果
/// - [BinComputeNodeResponse::Error]内部计算节点指令执行错误信息
#[derive(Serialize, Deserialize, Encode, Decode, Debug)]
pub enum BinComputeNodeResponse<
    State: Encode + Decode<()>,
    Status: Encode + Decode<()>,
    Work: Encode + Decode<()>,
> {
    /// 状态
    State(State),
    /// 运行时状态
    Status(Status),
    /// 执行结果
    Compute(Work),
    /// 错误
    Error(String),
}

/// 计算节点工作负载转换器
pub type WorkTransformer<T> = Arc<dyn Fn(T) -> T + Send + Sync + 'static>;

/// 计算节点包装器
///
/// 该包装器实现了[Listener] trait，用于处理计算节点请求
///
/// 监听指定的端口，接受[BinComputeNodeRequest]，返回[BinComputeNodeResponse]
#[derive(Clone)]
pub struct BinComputeNodeWrapper<T: ComputeNode> {
    inner: Arc<T>,
    max_frame_size: usize,
    buffer_capacity: usize,
    shutdown_timeout: Duration,
    input_transformer: Vec<WorkTransformer<T::InputData>>,
    output_transformer: Vec<WorkTransformer<T::OutputData>>,
}

impl<T: ComputeNode> Debug for BinComputeNodeWrapper<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BinComputeNodeWrapper")
            .field("inner", &"ComputeNode")
            .field("max_frame_size", &self.max_frame_size)
            .field("buffer_capacity", &self.buffer_capacity)
            .field("shutdown_timeout", &self.shutdown_timeout)
            .finish()
    }
}

#[bon]
impl<T: ComputeNode> BinComputeNodeWrapper<T> {
    /// 创建计算节点包装器
    #[builder]
    pub fn new(
        #[builder(field)] input_transformer: Vec<WorkTransformer<T::InputData>>,
        #[builder(field)] output_transformer: Vec<WorkTransformer<T::OutputData>>,
        #[builder(name = wrap)] inner: T,
        #[builder(default = 8 * 1024 * 1024)] max_frame_size: usize,
        #[builder(default = 10)] buffer_capacity: usize,
        #[builder(default = Duration::from_secs(300))] shutdown_timeout: Duration,
    ) -> Self {
        BinComputeNodeWrapper {
            inner: Arc::new(inner),
            max_frame_size,
            buffer_capacity,
            shutdown_timeout,
            input_transformer,
            output_transformer,
        }
    }
}

impl<T: ComputeNode, S: bin_compute_node_wrapper_builder::State>
    BinComputeNodeWrapperBuilder<T, S>
{
    /// 添加工作负载转换器
    ///
    /// 该转换器在每次收到计算请求时应用于输入数据，多次添加时，会按照添加顺序应用
    pub fn input_transformer<F: Fn(T::InputData) -> T::InputData + Send + Sync + 'static>(
        mut self,
        transformer: F,
    ) -> Self {
        self.input_transformer.push(Arc::new(transformer));
        self
    }

    /// 如果提供了工作负载转换器，则添加该转换器
    ///
    /// 该转换器在每次收到计算请求时应用于输入数据，多次添加时，会按照添加顺序应用
    pub fn maybe_input_transformer<F: Fn(T::InputData) -> T::InputData + Send + Sync + 'static>(
        mut self,
        transformer: Option<F>,
    ) -> Self {
        if let Some(transformer) = transformer {
            self.input_transformer.push(Arc::new(transformer));
        }
        self
    }

    /// 添加输出转换器
    ///
    /// 该转换器在每次计算完成后应用于输出数据，多次添加时，会按照添加顺序应用
    pub fn output_transformer<F: Fn(T::OutputData) -> T::OutputData + Send + Sync + 'static>(
        mut self,
        transformer: F,
    ) -> Self {
        self.output_transformer.push(Arc::new(transformer));
        self
    }

    /// 如果提供了输出转换器，则添加该转换器
    ///
    /// 该转换器在每次计算完成后应用于输出数据，多次添加时，会按照添加顺序应用
    pub fn maybe_output_transformer<
        F: Fn(T::OutputData) -> T::OutputData + Send + Sync + 'static,
    >(
        mut self,
        transformer: Option<F>,
    ) -> Self {
        if let Some(transformer) = transformer {
            self.output_transformer.push(Arc::new(transformer));
        }
        self
    }
}

impl<T: BinComputeNode> Listener for BinComputeNodeWrapper<T> {
    type Request =
        BinComputeNodeRequest<<T as ComputeNode>::Command, <T as ComputeNode>::InputData>;
    type Response = BinComputeNodeResponse<
        <T as ComputeNode>::State,
        <T as ComputeNode>::Status,
        <T as ComputeNode>::OutputData,
    >;

    async fn on_recv_frame(&self, frame: Self::Request) -> Self::Response {
        match frame {
            BinComputeNodeRequest::State => {
                self.inner.state().await.map(BinComputeNodeResponse::State)
            }
            BinComputeNodeRequest::Status => self
                .inner
                .status()
                .await
                .map(BinComputeNodeResponse::Status),
            BinComputeNodeRequest::Command(command) => self
                .inner
                .handle(command)
                .await
                .map(BinComputeNodeResponse::State),
            BinComputeNodeRequest::Compute(input) => {
                let work = self
                    .input_transformer
                    .iter()
                    .fold(input, |work, transformer| (*transformer)(work));
                self.inner.compute(work).await.map(|res| {
                    let res = self
                        .output_transformer
                        .iter()
                        .fold(res, |work, transformer| (*transformer)(work));
                    BinComputeNodeResponse::Compute(res)
                })
            }
        }
        .unwrap_or_else(|e| {
            warn!("BinComputeNodeWrapper found inner error: {}", e);
            BinComputeNodeResponse::Error(e)
        })
    }
}

impl<T: BinComputeNode> BinComputeNodeWrapper<T> {
    /// 获取该包装器对应的通道创建器
    #[allow(clippy::type_complexity)]
    pub fn channel_creator(
        &self,
    ) -> impl ChannelCreator<
        BinComputeNodeRequest<<T as ComputeNode>::Command, <T as ComputeNode>::InputData>,
        BinComputeNodeResponse<
            <T as ComputeNode>::State,
            <T as ComputeNode>::Status,
            <T as ComputeNode>::OutputData,
        >,
    > + use<T> {
        SimpleChannelCreator::<TypedConnection<_, _, _>>::new(
            self.max_frame_size,
            self.buffer_capacity,
        )
    }

    /// Serve until the stop future completes, drain accepted requests, then await
    /// node cleanup. Await this method to completion to release resources safely.
    /// On timeout, the process must exit with an error. Cleanup is not started
    /// while dispatched requests may still be running.
    pub async fn serve_until(
        self,
        tcp_listener: TcpListener,
        stop: impl Future<Output = ()> + Send,
    ) -> Result<(), String> {
        let inner = self.inner.clone();
        let shutdown_timeout = self.shutdown_timeout;
        let task = Task::new();
        let channel_creator = self.channel_creator();
        {
            let serving =
                Listener::listen_gracefully(self, task.clone(), tcp_listener, channel_creator);
            tokio::pin!(serving);
            tokio::select! {
                _ = &mut serving => {},
                _ = stop => {},
            }
        }
        task.cancel();
        let deadline = tokio::time::Instant::now() + shutdown_timeout;
        tokio::time::timeout_at(deadline, task.wait())
            .await
            .map_err(|_| "request drain timed out; cleanup skipped while requests may still be running; process must exit".to_owned())?;
        tokio::time::timeout_at(deadline, inner.shutdown())
            .await
            .map_err(|_| "node cleanup timed out; process must exit".to_owned())?
    }

    /// Serve until `task` is cancelled, including request draining and cleanup.
    pub async fn listen(self, task: Task, tcp_listener: TcpListener) -> Result<(), String> {
        self.serve_until(tcp_listener, task.cancelled()).await
    }
}

/// Wait for the platform termination signal used by all node entry points.
pub async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("signal handler");
        tokio::select! { _ = tokio::signal::ctrl_c() => {}, _ = term.recv() => {} }
    }
    #[cfg(not(unix))]
    tokio::signal::ctrl_c().await.expect("signal handler");
}
