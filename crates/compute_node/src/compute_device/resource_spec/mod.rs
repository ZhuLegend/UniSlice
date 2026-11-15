//! Resource specific definitions for compute devices.

#[cfg(any(feature = "resource_spec", feature = "nvidia"))]
pub mod nv;

#[cfg(any(feature = "resource_spec", feature = "huawei"))]
pub mod hw;
