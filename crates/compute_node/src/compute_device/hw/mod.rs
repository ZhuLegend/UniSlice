//! Huawei GPU 支持

use std::sync::OnceLock;

use hw_dcmi_wrapper::DCMI;

pub mod gpu;
#[cfg(feature = "vgpu")]
pub(crate) mod mig;

#[cfg(feature = "vgpu")]
pub mod vgpu;

pub use gpu::HuaweiGpu;
#[cfg(feature = "vgpu")]
pub use vgpu::HuaweiVirtualGpu;

#[cfg(feature = "vgpu")]
pub use crate::compute_device::vgpu::resource_spec::hw::{
    VChipScale, VChipTemplateB1B2, VChipTemplateB3, VChipTemplateB4,
};

/// 获取DCMI的多线程安全唯一示例
pub fn dcmi() -> &'static DCMI {
    static INNER: OnceLock<DCMI> = OnceLock::new();
    INNER.get_or_init(|| DCMI::init().unwrap())
}
