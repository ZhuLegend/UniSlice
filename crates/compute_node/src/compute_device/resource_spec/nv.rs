//! Nvidia resource specific definition

use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};

use crate::compute_node::r#virtual::base::DeviceState;

/// Nvidia GPU 请求
#[derive(Serialize, Deserialize, Encode, Decode)]
pub enum NvGpuRequest {
    /// 查询内存信息
    QueryMemoryInfo,
}

/// Nvidia GPU 响应
#[derive(Debug, Serialize, Deserialize, Encode, Decode)]
pub enum NvGpuResponse {
    /// 内存信息
    MemoryInfo {
        /// 总内存
        total: u64,
        /// 空闲内存
        free: u64,
        /// 已使用内存
        used: u64,
    },
    /// Nvml错误信息
    ///
    /// # 参数
    ///
    /// * `message` - 错误信息
    NvmlError(String),
}

#[cfg(feature = "vgpu")]
impl DeviceState for NvGpuResponse {}
