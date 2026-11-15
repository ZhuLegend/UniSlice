//! 虚拟GPU模块

use std::{
    fmt::{Debug, Display},
    future::Future,
    ops::Deref,
};

use super::ComputeDevice;

mod full_device_wrapper;

#[cfg(any(feature = "nvidia", feature = "huawei", feature = "resource_spec"))]
pub mod resource_spec;

/// 共享目标
pub mod share_target;
/// 可共享的计算设备
pub mod shareable;

pub use full_device_wrapper::*;
pub use shareable::{
    ShareableComputeDevice, ShareableComputeDeviceResource, ShareableResource, SharedResource,
};

/// 虚拟GPU
#[trait_variant::make(Send)]
pub trait VirtualGpu: Sized + Sync + std::fmt::Debug {
    /// 宿主机设备
    type HostDevice: ComputeDevice;

    /// VGpu状态
    type State: Send + Clone;

    /// 错误类型
    type Error: Display;

    /// 宿主机设备
    fn host_device(&self) -> impl Deref<Target = Self::HostDevice> + '_;

    /// 获取VGpu状态
    async fn state(&self) -> Self::State;

    /// 向宿主机发出请求
    fn request_host(
        &self,
        request: <Self::HostDevice as ComputeDevice>::Request,
    ) -> impl Future<Output = Result<<Self::HostDevice as ComputeDevice>::Response, Self::Error>> + Send
    {
        async move { Ok(self.host_device().request(request).await) }
    }

    /// Release this virtual device, retaining its lease on failure.
    fn release(self) -> Result<(), String> {
        Ok(())
    }

    /// 获取设备逻辑索引
    fn logic_index(&self) -> Option<u32> {
        None
    }
}

/// VGpu请求
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, bincode::Encode, bincode::Decode)]
pub enum VirtualGpuRequest<HostRequest> {
    /// 宿主机请求
    HostRequest(HostRequest),
    /// 获取VGpu状态
    State,
}

/// VGpu响应
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, bincode::Encode, bincode::Decode)]
pub enum VirtualGpuResponse<HostResponse, State> {
    /// 宿主机响应
    HostResponse(HostResponse),
    /// VGpu状态
    State(State),
    /// 请求执行错误
    Error(String),
}

impl<T> ComputeDevice for T
where
    T: VirtualGpu,
    <<T as VirtualGpu>::HostDevice as ComputeDevice>::Request: Send,
    <<T as VirtualGpu>::HostDevice as ComputeDevice>::Response: Send,
{
    type Request = VirtualGpuRequest<<<T as VirtualGpu>::HostDevice as ComputeDevice>::Request>;
    type Response = VirtualGpuResponse<
        <<T as VirtualGpu>::HostDevice as ComputeDevice>::Response,
        <T as VirtualGpu>::State,
    >;

    async fn request(&self, request: Self::Request) -> Self::Response {
        match request {
            VirtualGpuRequest::HostRequest(request) => match self.request_host(request).await {
                Ok(response) => VirtualGpuResponse::HostResponse(response),
                Err(err) => VirtualGpuResponse::Error(err.to_string()),
            },
            VirtualGpuRequest::State => VirtualGpuResponse::State(self.state().await),
        }
    }

    fn release(self) -> Result<(), String> {
        VirtualGpu::release(self)
    }

    fn logic_index(&self) -> Option<u32> {
        self.logic_index()
    }
}
