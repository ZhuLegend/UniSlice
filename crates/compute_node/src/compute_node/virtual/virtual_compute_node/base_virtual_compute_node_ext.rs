//! some extensions for [BaseVirtualComputeNode] which used by [VirtualComputeNode]

use std::{fmt::Debug, future::Future, marker::PhantomData, pin::Pin, sync::Arc, time::Duration};

use bincode::{Decode, Encode};
use comm::rpc_pool::RpcPool;
use serde::{Serialize, de::DeserializeOwned};

use crate::{
    ComputeNode,
    compute_device::{
        ComputeDevice,
        vgpu::{ShareableComputeDevice, share_target::ShareTarget},
    },
    compute_node::r#virtual::base::{
        self, BaseVirtualComputeNode, Command, DeviceManageHelper, DeviceState, State, Status, Work,
    },
    runnable_on::RunnableOn,
    util::{
        Input, Output,
        bin_compute_node::{BinComputeNodeRequest, BinComputeNodeResponse},
        worker_proxy::{
            WorkerProxy,
            bin_simple_worker_proxy::{BinSimpleWorkerProxyImplInit, WorkerProxyCapable},
        },
    },
};

/// 动态虚拟计算节点 trait，提供类型擦除的接口
pub trait DynBaseVirtualComputeNode: Send + Sync {
    /// Device sharing parameter.
    type ShareParam: ShareTarget + Serialize + DeserializeOwned + Debug + Send + Sync;

    /// Device status representation.
    type DeviceState: DeviceState + Encode + Decode<()> + Send + Sync;

    /// 执行命令
    fn command(&self, command: Command<Self::ShareParam>) -> CommandFuture<'_, Self::ShareParam>;

    /// 获取状态
    fn state(&self) -> StateFuture<'_, Self::ShareParam>;

    /// 获取状态统计
    fn status(&self) -> StatusFuture<'_, Self::DeviceState>;

    /// 执行计算任务
    fn compute(&self, work: Work<Input>) -> ComputeFuture<'_>;
}

/// 命令执行的 Future 类型别名
pub type CommandFuture<'a, ShareParam> =
    Pin<Box<dyn Future<Output = Result<State<ShareParam>, String>> + Send + 'a>>;

/// 状态查询的 Future 类型别名
pub type StateFuture<'a, ShareParam> =
    Pin<Box<dyn Future<Output = Result<State<ShareParam>, String>> + Send + 'a>>;

/// 状态统计的 Future 类型别名
pub type StatusFuture<'a, Response> =
    Pin<Box<dyn Future<Output = Result<Status<Response>, String>> + Send + 'a>>;

/// 计算任务的 Future 类型别名
pub type ComputeFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Work<Output>, String>> + Send + 'a>>;

impl<Worker, Device, Proxy> DynBaseVirtualComputeNode
    for BaseVirtualComputeNode<Worker, Device, Proxy>
where
    Worker: worker::Worker
        + RunnableOn<<Device as ShareableComputeDevice>::SharedComputeDevice>
        + WorkerProxyCapable<<Device as ShareableComputeDevice>::SharedComputeDevice>
        + 'static,
    Device: ShareableComputeDevice + Send + Sync + 'static,
    <Device as ShareableComputeDevice>::RawComputeDevice:
        DeviceManageHelper<<Device as ShareableComputeDevice>::RawComputeDevice>,
    <<Device as ShareableComputeDevice>::RawComputeDevice as ComputeDevice>::Response:
        Encode + Decode<()> + Send + Sync + DeviceState,
    <Device as ShareableComputeDevice>::ShareParam:
        Serialize + DeserializeOwned + Debug + Send + Sync,
    Proxy: WorkerProxy
        + BinSimpleWorkerProxyImplInit<<Device as ShareableComputeDevice>::SharedComputeDevice>
        + Send
        + Sync
        + 'static,
{
    type ShareParam = <Device as ShareableComputeDevice>::ShareParam;
    type DeviceState =
        <<Device as ShareableComputeDevice>::RawComputeDevice as ComputeDevice>::Response;

    fn command(&self, command: Command<Self::ShareParam>) -> CommandFuture<'_, Self::ShareParam> {
        Box::pin(<Self as ComputeNode>::handle(self, command))
    }

    fn state(&self) -> StateFuture<'_, Self::ShareParam> {
        Box::pin(<Self as ComputeNode>::state(self))
    }

    fn status(&self) -> StatusFuture<'_, Self::DeviceState> {
        Box::pin(<Self as ComputeNode>::status(self))
    }

    fn compute(&self, work: Work<Input>) -> ComputeFuture<'_> {
        Box::pin(<Self as ComputeNode>::compute(self, work))
    }
}

impl<Worker, Device, Proxy> BaseVirtualComputeNode<Worker, Device, Proxy>
where
    Worker: worker::Worker
        + RunnableOn<<Device as ShareableComputeDevice>::SharedComputeDevice>
        + WorkerProxyCapable<<Device as ShareableComputeDevice>::SharedComputeDevice>
        + 'static,
    Device: ShareableComputeDevice + Send + Sync + 'static,
    <Device as ShareableComputeDevice>::RawComputeDevice:
        DeviceManageHelper<<Device as ShareableComputeDevice>::RawComputeDevice>,
    <<Device as ShareableComputeDevice>::RawComputeDevice as ComputeDevice>::Response:
        Encode + Decode<()> + Send + Sync + DeviceState,
    <Device as ShareableComputeDevice>::ShareParam:
        Serialize + DeserializeOwned + Debug + Send + Sync,
    Proxy: WorkerProxy
        + BinSimpleWorkerProxyImplInit<<Device as ShareableComputeDevice>::SharedComputeDevice>
        + Send
        + Sync
        + 'static,
{
    /// Convert the BaseVirtualComputeNode into a dynamic trait object
    pub fn into_dyn(
        self,
    ) -> Arc<
        dyn DynBaseVirtualComputeNode<
                ShareParam = <Device as ShareableComputeDevice>::ShareParam,
                DeviceState = <<Device as ShareableComputeDevice>::RawComputeDevice as ComputeDevice>::Response,
            >,
    >{
        Arc::new(self)
    }
}

/// 远程BaseVirtualComputeNode代理
///
/// 该结构体通过网络连接与远程的BinComputeNode通信，
/// 实现了DynBaseVirtualComputeNode trait，提供与本地节点相同的接口
pub struct RemoteBaseVirtualComputeNode<
    ShareParam: ShareTarget + Serialize + DeserializeOwned + Debug + Send + Sync + Encode + Decode<()>,
    DeviceState: base::DeviceState + Encode + Decode<()> + Send + Sync,
> {
    /// 远程节点地址（用于连接和标识）
    address: String,
    connections: RpcPool<
        BinComputeNodeRequest<Command<ShareParam>, Work<Input>>,
        BinComputeNodeResponse<State<ShareParam>, Status<DeviceState>, Work<Output>>,
    >,
    control_timeout: Duration,
    inference_timeout: Duration,
    /// 设备类型标记
    _share_param: PhantomData<ShareParam>,
    _device_state: PhantomData<DeviceState>,
}

impl<
    ShareParam: ShareTarget
        + Serialize
        + DeserializeOwned
        + Debug
        + Send
        + Sync
        + Encode
        + Decode<()>
        + 'static,
    DeviceState: base::DeviceState + Encode + Decode<()> + Send + Sync + 'static,
> RemoteBaseVirtualComputeNode<ShareParam, DeviceState>
{
    /// 创建新的远程BaseVirtualComputeNode
    pub fn new(addr: impl std::fmt::Display) -> Self {
        Self {
            address: addr.to_string(),
            connections: RpcPool::new(addr.to_string()),
            control_timeout: Duration::from_secs(30),
            inference_timeout: Duration::from_secs(30),
            _share_param: PhantomData,
            _device_state: PhantomData,
        }
    }

    /// Configure separate response budgets for lifecycle commands and inference.
    pub fn with_timeouts(mut self, control: Duration, inference: Duration) -> Self {
        self.control_timeout = control;
        self.inference_timeout = inference;
        self
    }

    /// 获取远程节点地址
    pub fn address(&self) -> &str {
        &self.address
    }

    /// 通用的请求-响应处理方法
    ///
    /// 该方法提取了四个命令的公共模式：
    /// 1. 创建连接
    /// 2. 发送请求
    /// 3. 接收响应
    /// 4. 处理响应并返回结果
    async fn execute_request<T>(
        &self,
        request: BinComputeNodeRequest<Command<ShareParam>, Work<Input>>,
        response_matcher: impl Fn(
            BinComputeNodeResponse<State<ShareParam>, Status<DeviceState>, Work<Output>>,
        ) -> Result<T, String>,
    ) -> Result<T, String> {
        let response_timeout = if matches!(&request, BinComputeNodeRequest::Compute(_)) {
            self.inference_timeout
        } else {
            self.control_timeout
        };
        let response = self
            .connections
            .request(
                &request,
                Duration::from_secs(10),
                self.control_timeout,
                response_timeout,
            )
            .await
            .map_err(|e| format!("RPC to {}: {e}", self.address))?;
        response_matcher(response)
    }
}

impl<
    ShareParam: ShareTarget
        + Serialize
        + DeserializeOwned
        + Debug
        + Send
        + Sync
        + Encode
        + Decode<()>
        + 'static,
    DeviceState: base::DeviceState + Encode + Decode<()> + Send + Sync + 'static,
> DynBaseVirtualComputeNode for RemoteBaseVirtualComputeNode<ShareParam, DeviceState>
{
    type ShareParam = ShareParam;
    type DeviceState = DeviceState;

    fn command(&self, command: Command<Self::ShareParam>) -> CommandFuture<'_, Self::ShareParam> {
        Box::pin(async move {
            let request = BinComputeNodeRequest::Command(command);
            self.execute_request(request, |response| match response {
                BinComputeNodeResponse::State(state) => Ok(state),
                BinComputeNodeResponse::Error(err) => Err(err),
                _ => Err("意外的响应类型，期望命令响应".to_string()),
            })
            .await
        })
    }

    fn state(&self) -> StateFuture<'_, Self::ShareParam> {
        Box::pin(async move {
            let request = BinComputeNodeRequest::State;
            self.execute_request(request, |response| match response {
                BinComputeNodeResponse::State(state) => Ok(state),
                BinComputeNodeResponse::Error(err) => Err(err),
                _ => Err("意外的响应类型，期望状态响应".to_string()),
            })
            .await
        })
    }

    fn status(&self) -> StatusFuture<'_, Self::DeviceState> {
        Box::pin(async move {
            let request = BinComputeNodeRequest::Status;
            self.execute_request(request, |response| match response {
                BinComputeNodeResponse::Status(status) => Ok(status),
                BinComputeNodeResponse::Error(err) => Err(err),
                _ => Err("意外的响应类型，期望状态统计响应".to_string()),
            })
            .await
        })
    }

    fn compute(&self, work: Work<Input>) -> ComputeFuture<'_> {
        Box::pin(async move {
            let request = BinComputeNodeRequest::Compute(work);
            self.execute_request(request, |response| match response {
                BinComputeNodeResponse::Compute(result) => Ok(result),
                BinComputeNodeResponse::Error(err) => Err(err),
                _ => Err("意外的响应类型，期望计算响应".to_string()),
            })
            .await
        })
    }
}

impl<
    ShareParam: ShareTarget + Serialize + DeserializeOwned + Debug + Send + Sync + Encode + Decode<()>,
    DeviceState: base::DeviceState + Encode + Decode<()> + Send + Sync,
> Debug for RemoteBaseVirtualComputeNode<ShareParam, DeviceState>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteBaseVirtualComputeNode")
            .field("address", &self.address)
            .field("share_param", &self._share_param)
            .field("device_state", &self._device_state)
            .finish()
    }
}
