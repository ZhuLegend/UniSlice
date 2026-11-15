//! MIG Resource Specification for Nvidia GPUs

use std::{fmt::Display, str::FromStr};

use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};

use crate::compute_device::vgpu::{
    ShareableComputeDeviceResource,
    share_target::{ChoosableShareTargetContainer, ShareTarget},
};

/// Nvidia A100 GPU 支持的 MIG 模板
#[derive(
    Clone, Copy, Debug, Eq, PartialEq, Hash, PartialOrd, Ord, Serialize, Deserialize, Encode, Decode,
)]
pub enum A100Template {
    /// MIG 1g.10gb Total 7
    Tiny,
    /// MIG 1g.20gb Total 4
    Small,
    /// MIG 2g.20gb Total 3
    Medium,
    /// MIG 3g.40gb Total 2
    Big,
    /// MIG 4g.40gb Total 1
    Large,
    /// MIG 7g.80gb Total 1
    Huge,
}

impl A100Template {
    /// 获取模板的 id
    pub fn id(&self) -> u32 {
        match self {
            A100Template::Tiny => 19,
            A100Template::Small => 15,
            A100Template::Medium => 14,
            A100Template::Big => 9,
            A100Template::Large => 5,
            A100Template::Huge => 0,
        }
    }

    /// 获取模板的名称
    pub fn name(&self) -> &'static str {
        match self {
            A100Template::Tiny => "1g.10gb",
            A100Template::Small => "1g.20gb",
            A100Template::Medium => "2g.20gb",
            A100Template::Big => "3g.40gb",
            A100Template::Large => "4g.40gb",
            A100Template::Huge => "7g.80gb",
        }
    }
}

impl FromStr for A100Template {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "1g.10gb" => Ok(A100Template::Tiny),
            "1g.20gb" => Ok(A100Template::Small),
            "2g.20gb" => Ok(A100Template::Medium),
            "3g.40gb" => Ok(A100Template::Big),
            "4g.40gb" => Ok(A100Template::Large),
            "7g.80gb" => Ok(A100Template::Huge),
            _ => Err("Unknown template, currently supports 1g.10gb, 1g.20gb, 2g.20gb, 3g.40gb, 4g.40gb, 7g.80gb only".to_string()),
        }
    }
}

/// 算力切分规模
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Encode, Decode,
)]
pub enum VDeviceScale {
    /// NVIDIA A100 80GB
    A100(A100Template),
}

impl VDeviceScale {
    /// 获取模板的名称
    pub fn name(&self) -> &'static str {
        match self {
            VDeviceScale::A100(scale) => scale.name(),
        }
    }
}

impl ShareTarget for VDeviceScale {
    fn name(&self) -> &'static str {
        VDeviceScale::name(self)
    }

    fn from_name(name: &str) -> Option<Self> {
        A100Template::from_str(name).ok().map(VDeviceScale::A100)
    }
}

impl ChoosableShareTargetContainer for VDeviceScale {
    type Target = Self;

    fn all_names(&self) -> Vec<&'static str> {
        vec![
            "1g.10gb", "1g.20gb", "2g.20gb", "3g.40gb", "4g.40gb", "7g.80gb",
        ]
    }
}

impl From<&VDeviceScale> for VDeviceScale {
    fn from(value: &VDeviceScale) -> Self {
        *value
    }
}

/// Nvidia MIG 资源
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde::Serialize,
    serde::Deserialize,
    bincode::Encode,
    bincode::Decode,
)]
pub struct NvidiaMigResource {
    /// Compute Engine 数量
    pub ce: u8,
    /// 内存大小（GB）
    pub memory: u16,
}

impl Display for NvidiaMigResource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "Nvidia MIG Resource: CE: {}, Memory: {}GB",
            self.ce, self.memory
        ))
    }
}

impl ShareableComputeDeviceResource for NvidiaMigResource {
    fn can_share(&self, other: &Self) -> bool {
        self.ce >= other.ce && self.memory >= other.memory
    }

    fn borrow(&mut self, target: &Self) -> Option<Self> {
        if self.can_share(target) {
            self.ce -= target.ce;
            self.memory -= target.memory;
            Some(Self {
                ce: target.ce,
                memory: target.memory,
            })
        } else {
            None
        }
    }

    fn refund(&mut self, resource: Self) {
        self.ce += resource.ce;
        self.memory += resource.memory;
    }
}

impl From<&'_ A100Template> for NvidiaMigResource {
    fn from(template: &'_ A100Template) -> Self {
        let (ce, memory) = match template {
            A100Template::Tiny => (1, 10),
            A100Template::Small => (1, 20),
            A100Template::Medium => (2, 20),
            A100Template::Big => (3, 40),
            A100Template::Large => (4, 40),
            A100Template::Huge => (7, 80),
        };
        Self { ce, memory }
    }
}

impl From<&'_ VDeviceScale> for NvidiaMigResource {
    fn from(value: &'_ VDeviceScale) -> Self {
        match value {
            VDeviceScale::A100(scale) => NvidiaMigResource::from(scale),
        }
    }
}

impl From<VDeviceScale> for NvidiaMigResource {
    fn from(value: VDeviceScale) -> Self {
        (&value).into()
    }
}
