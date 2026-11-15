//! Huawei resource specific definition

use bincode::{Decode, Encode};
#[cfg(feature = "huawei")]
use hw_dcmi_wrapper::structs::{HBMInfo, MemoryInfo};
use serde::{Deserialize, Serialize};

use crate::compute_node::r#virtual::base::DeviceState;

/// Huawei GPU 请求
#[derive(Serialize, Deserialize, Encode, Decode)]
pub enum HwGpuRequest {
    /// 查询内存信息
    QueryMemoryInfo,
    /// 查询HBM内存信息
    QueryHBMInfo,
}

/// Huawei NPU DDR 内存信息
#[derive(Debug, Serialize, Deserialize, Encode, Decode)]
pub struct HwMemoryInfo {
    /// 总内存，MB
    pub memory_size: u64,
    /// 可用内存，MB，free + huge_pages_free * huge_page_size
    pub memory_available: u64,
    /// 频率
    pub freq: u32,
    /// 大页大小，KB
    pub huge_page_size: u64,
    /// 大页总数
    pub huge_pages_total: u64,
    /// 空闲大页数
    pub huge_pages_free: u64,
    /// 利用率，DDR内存信息使用情况
    pub utilization: u32,
}

#[cfg(feature = "huawei")]
impl From<MemoryInfo> for HwMemoryInfo {
    fn from(info: MemoryInfo) -> Self {
        HwMemoryInfo {
            memory_size: info.memory_size,
            memory_available: info.memory_available,
            freq: info.freq,
            huge_page_size: info.huge_page_size,
            huge_pages_total: info.huge_pages_total,
            huge_pages_free: info.huge_pages_free,
            utilization: info.utilization,
        }
    }
}

/// Huawei HBM 内存信息
#[derive(Debug, Serialize, Deserialize, Encode, Decode)]
pub struct HwHBMInfo {
    /// HBM 总内存，MB
    pub memory_size: u64,
    /// HBM 频率，MHz
    pub frequency: u32,
    /// HBM 已使用内存，MB
    pub memory_usage: u64,
    /// HBM 温度
    pub temperature: i32,
    /// HBM 带宽利用率
    pub bandwidth_util_rate: u32,
}

#[cfg(feature = "huawei")]
impl From<HBMInfo> for HwHBMInfo {
    fn from(info: HBMInfo) -> Self {
        HwHBMInfo {
            memory_size: info.memory_size,
            frequency: info.frequency,
            memory_usage: info.memory_usage,
            temperature: info.temperature,
            bandwidth_util_rate: info.bandwidth_util_rate,
        }
    }
}

/// Huawei GPU 响应
#[derive(Debug, Serialize, Deserialize, Encode, Decode)]
pub enum HwGpuResponse {
    /// DDR内存信息
    /// u32 - Chip ID
    MemoryInfos(Vec<(u32, Result<HwMemoryInfo, String>)>),
    /// HBM内存信息
    /// u32 - Chip ID
    HBMInfos(Vec<(u32, Result<HwHBMInfo, String>)>),
    /// DCMI错误信息
    ///
    /// # 参数
    ///
    /// * `message` - 错误信息
    DCMIError(String),
}

#[cfg(feature = "vgpu")]
impl DeviceState for HwGpuResponse {}
