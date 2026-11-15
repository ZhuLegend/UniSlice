//! Resource implementations for virtual compute node scheduling.

#[cfg(feature = "huawei")]
pub mod huawei_mig;

#[cfg(feature = "nvidia")]
pub mod nvidia_mig;
