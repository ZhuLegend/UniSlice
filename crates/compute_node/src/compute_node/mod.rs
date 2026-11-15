//! Compute Node module

#[cfg(feature = "vgpu")]
pub mod r#virtual;

#[cfg(feature = "nvidia")]
pub mod nv;

#[cfg(feature = "huawei")]
pub mod hw;

#[cfg(feature = "mock")]
pub mod mock;
