//! NVIDIA GPU 支持

use std::sync::OnceLock;

use nvml_wrapper::Nvml;

pub(crate) mod gpu;

#[cfg(feature = "vgpu")]
pub mod mig;
#[cfg(feature = "vgpu")]
pub mod vgpu;

pub use gpu::NvidiaGpu;
#[cfg(feature = "vgpu")]
pub use vgpu::NvidiaMpsVirtualGpu;

/// 获取Nvml的多线程安全唯一示例
pub fn nvml() -> &'static Nvml {
    static INNER: OnceLock<Nvml> = OnceLock::new();
    INNER.get_or_init(|| Nvml::init().unwrap())
}
