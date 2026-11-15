//! Shareable resource specifications for vGPU devices.

#[cfg(any(feature = "nvidia", feature = "resource_spec"))]
pub mod nv;

#[cfg(any(feature = "huawei", feature = "resource_spec"))]
pub mod hw;
