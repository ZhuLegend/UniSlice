//! 计算设备抽象

#![warn(missing_docs)]

#[cfg(feature = "vgpu")]
pub use vgpu::VirtualGpu;

#[cfg(feature = "vgpu")]
pub mod vgpu;

pub mod cpu;

#[cfg(any(feature = "resource_spec", feature = "nvidia", feature = "huawei"))]
pub mod resource_spec;

#[cfg(feature = "nvidia")]
pub mod nv;

#[cfg(feature = "huawei")]
pub mod hw;

#[cfg(feature = "mock")]
pub mod mock;

pub mod config;

/// 计算设备抽象
///
/// 实现该`trait`的类型，应当提供自身接受的请求种类和所有可能的响应类型
/// 通过异步的方式完成对请求的响应
///
/// 通常[ComputeDevice::Request]和[ComputeDevice::Response]可以选取enum实现
#[trait_variant::make(Send)]
pub trait ComputeDevice: std::fmt::Debug {
    /// 向计算设备发出的请求
    type Request: Send;
    /// GPU接受请求后返回的响应
    type Response: Send;

    /// 向计算设备发出请求，接收相应响应
    async fn request(&self, request: Self::Request) -> Self::Response;

    /// Consume the device after its worker stops. Implementations must retain
    /// the lease if physical destruction fails, including during `Drop`.
    ///
    /// ```compile_fail
    /// fn release_twice<D: compute_node::compute_device::ComputeDevice>(device: D) {
    ///     device.release().unwrap();
    ///     device.release().unwrap(); // ownership was consumed
    /// }
    /// ```
    fn release(self) -> Result<(), String>
    where
        Self: Sized,
    {
        Ok(())
    }

    /// 计算设备逻辑索引
    fn logic_index(&self) -> Option<u32>;
}
