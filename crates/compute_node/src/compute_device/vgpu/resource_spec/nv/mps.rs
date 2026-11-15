//! MPS Resource Specification for Nvidia GPUs

use std::fmt::Display;

use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};

use crate::compute_device::vgpu::{ShareableComputeDeviceResource, share_target::ShareTarget};

/// Nvidia Mps Resource
///
/// a wrapper of u32, which represent the percentage of the total resource used.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct NvidiaMpsResource(pub(crate) u32);

impl NvidiaMpsResource {
    /// 获取内部的百分比值（引用）
    pub fn percentage(&self) -> u32 {
        self.0
    }
}

impl ShareableComputeDeviceResource for NvidiaMpsResource {
    fn can_share(&self, _other: &Self) -> bool {
        true
    }

    fn borrow(&mut self, target: &Self) -> Option<Self> {
        self.0 -= target.0;
        Some(*target)
    }

    fn refund(&mut self, target: Self) {
        self.0 += target.0;
    }
}

impl Display for NvidiaMpsResource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("MPS Resource: {}%", self.0))
    }
}

impl From<&'_ NvidiaMpsShareTarget> for NvidiaMpsResource {
    fn from(percentage: &'_ NvidiaMpsShareTarget) -> Self {
        Self(percentage.0)
    }
}

/// Nvidia Mps Share Target
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize, Encode, Decode)]
pub struct NvidiaMpsShareTarget(pub u32, String);

impl From<u32> for NvidiaMpsShareTarget {
    fn from(percentage: u32) -> Self {
        NvidiaMpsShareTarget(percentage, percentage.to_string())
    }
}

impl ShareTarget for NvidiaMpsShareTarget {
    fn name(&self) -> &str {
        self.1.as_str()
    }

    fn from_name(name: &str) -> Option<Self> {
        let percentage = name.parse().ok()?;
        if percentage > 100 {
            None
        } else {
            Some(NvidiaMpsShareTarget(percentage, name.to_string()))
        }
    }
}
