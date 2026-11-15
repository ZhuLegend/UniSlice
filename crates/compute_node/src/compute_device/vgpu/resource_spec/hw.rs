//! Shareable Resource Specification for Huawei NPUs

use std::fmt::Display;

use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};

use crate::compute_device::vgpu::{
    ShareableComputeDeviceResource,
    share_target::{ChoosableShareTargetContainer, ShareTarget},
};

/// Huawei MIG-like 资源
#[derive(
    Debug,
    Clone,
    Copy,
    Ord,
    PartialOrd,
    Eq,
    PartialEq,
    Hash,
    serde::Serialize,
    serde::Deserialize,
    bincode::Encode,
    bincode::Decode,
)]
pub struct HuaweiMigResource {
    /// AI核心数量
    pub ai_core: u8,
    /// AI CPU数量
    pub ai_cpu: u8,
    /// 内存大小（GB）
    pub memory: u16,
}

impl Display for HuaweiMigResource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "Ai Core: {}, Ai CPU: {}, Memory: {} GB",
            self.ai_core, self.ai_cpu, self.memory
        ))
    }
}

impl ShareableComputeDeviceResource for HuaweiMigResource {
    fn can_share(&self, other: &Self) -> bool {
        self.ai_core >= other.ai_core && self.ai_cpu >= other.ai_cpu && self.memory >= other.memory
    }

    fn borrow(&mut self, target: &Self) -> Option<Self> {
        if self.can_share(target) {
            self.ai_core -= target.ai_core;
            self.ai_cpu -= target.ai_cpu;
            self.memory -= target.memory;
            Some(*target)
        } else {
            None
        }
    }

    fn refund(&mut self, target: Self) {
        self.ai_core += target.ai_core;
        self.ai_cpu += target.ai_cpu;
        self.memory += target.memory;
    }
}

impl From<&'_ VChipTemplateB1B2> for HuaweiMigResource {
    fn from(template: &'_ VChipTemplateB1B2) -> Self {
        match template {
            VChipTemplateB1B2::Big => Self {
                ai_core: 12,
                ai_cpu: 3,
                memory: 32,
            },
            VChipTemplateB1B2::Small => Self {
                ai_core: 6,
                ai_cpu: 1,
                memory: 16,
            },
            VChipTemplateB1B2::Tiny => Self {
                ai_core: 3,
                ai_cpu: 1,
                memory: 8,
            },
        }
    }
}

impl From<&'_ VChipTemplateB3> for HuaweiMigResource {
    fn from(value: &'_ VChipTemplateB3) -> Self {
        match value {
            VChipTemplateB3::Big => Self {
                ai_core: 10,
                ai_cpu: 3,
                memory: 32,
            },
            VChipTemplateB3::Small => Self {
                ai_core: 5,
                ai_cpu: 1,
                memory: 16,
            },
        }
    }
}

impl From<&'_ VChipTemplateB4> for HuaweiMigResource {
    fn from(value: &'_ VChipTemplateB4) -> Self {
        match value {
            VChipTemplateB4::Huge => Self {
                ai_core: 10,
                ai_cpu: 4,
                memory: 16,
            },
            VChipTemplateB4::Big => Self {
                ai_core: 10,
                ai_cpu: 3,
                memory: 16,
            },
            VChipTemplateB4::BigNoMemory => Self {
                ai_core: 10,
                ai_cpu: 3,
                memory: 16,
            },
            VChipTemplateB4::Small => Self {
                ai_core: 5,
                ai_cpu: 1,
                memory: 8,
            },
        }
    }
}

impl From<&'_ VChipScale> for HuaweiMigResource {
    fn from(value: &'_ VChipScale) -> Self {
        match value {
            VChipScale::B1(template) => template.into(),
            VChipScale::B2(template) => template.into(),
            VChipScale::B3(template) => template.into(),
            VChipScale::B4(template) => template.into(),
        }
    }
}

impl From<VChipScale> for HuaweiMigResource {
    fn from(value: VChipScale) -> Self {
        (&value).into()
    }
}

/// B1、B2设备算力切分模板
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Encode, Decode,
)]
pub enum VChipTemplateB1B2 {
    /// 小型配置实例：3个AICORE，8 GB内存，1个AICPU核心，较少的资源
    Tiny,
    /// 中型配置实例：6个AICORE，16 GB内存，1个AICPU核心，适量的VPC和JPEG处理资源
    Small,
    /// 大型配置实例：12个AICORE，32 GB内存，3个AICPU核心，适量的VPC和JPEG处理资源
    Big,
}

impl VChipTemplateB1B2 {
    /// 获取模板名称
    pub fn name(&self) -> &'static str {
        match self {
            VChipTemplateB1B2::Big => "vir12_3c_32g",
            VChipTemplateB1B2::Small => "vir06_1c_16g",
            VChipTemplateB1B2::Tiny => "vir03_1c_8g",
        }
    }
}

/// B3设备算力切分模板
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Encode, Decode,
)]
pub enum VChipTemplateB3 {
    /// 小型配置实例：5个AICORE，16 GB内存，1个AICPU核心，以及少量的VPC和JPEG处理资源
    Small,
    /// 大型配置实例：10个AICORE，32 GB内存，3个AICPU核心，以及适量的VPC和JPEG处理资源
    Big,
}

impl VChipTemplateB3 {
    /// 获取模板名称
    pub fn name(&self) -> &'static str {
        match self {
            VChipTemplateB3::Big => "vir10_3c_32g",
            VChipTemplateB3::Small => "vir05_1c_16g",
        }
    }
}

/// B4设备算力切分模板
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Encode, Decode,
)]
pub enum VChipTemplateB4 {
    /// 小配实例：5 GB AICORE，8 GB内存，以及少量的其他资源（适合轻量级任务）
    Small,
    /// 中配精简实例：10 GB AICORE，16 GB内存，但没有额外的视频或JPEG处理资源
    BigNoMemory,
    /// 中配实例：10 GB AICORE，16 GB内存，3个AICPU核心，以及适量的其他资源
    Big,
    /// 高配实例：10 GB AICORE，16 GB内存，4个AICPU核心，以及较多的其他资源（如JPEG处理、视频解码等）
    Huge,
}

impl VChipTemplateB4 {
    /// 获取模板名称
    pub fn name(&self) -> &'static str {
        match self {
            VChipTemplateB4::Huge => "vir10_3c_16g_m",
            VChipTemplateB4::Big => "vir10_3c_16g",
            VChipTemplateB4::BigNoMemory => "vir10_3c_16g_nm",
            VChipTemplateB4::Small => "vir05_1c_8g",
        }
    }
}

/// 算力切分规模
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Encode, Decode,
)]
pub enum VChipScale {
    /// B1设备
    B1(VChipTemplateB1B2),
    /// B2设备
    B2(VChipTemplateB1B2),
    /// B3设备
    B3(VChipTemplateB3),
    /// B4设备
    B4(VChipTemplateB4),
}

impl VChipScale {
    /// 获取模板名称
    pub fn name(&self) -> &'static str {
        match self {
            VChipScale::B1(template) => template.name(),
            VChipScale::B2(template) => template.name(),
            VChipScale::B3(template) => template.name(),
            VChipScale::B4(template) => template.name(),
        }
    }
}

impl From<&str> for VChipScale {
    fn from(value: &str) -> Self {
        match value {
            "vir12_3c_32g" => VChipScale::B1(VChipTemplateB1B2::Big),
            "vir06_1c_16g" => VChipScale::B1(VChipTemplateB1B2::Small),
            "vir03_1c_8g" => VChipScale::B1(VChipTemplateB1B2::Tiny),
            "vir10_3c_32g" => VChipScale::B3(VChipTemplateB3::Big),
            "vir05_1c_16g" => VChipScale::B3(VChipTemplateB3::Small),
            "vir10_3c_16g_m" => VChipScale::B4(VChipTemplateB4::Huge),
            "vir10_3c_16g" => VChipScale::B4(VChipTemplateB4::Big),
            "vir10_3c_16g_nm" => VChipScale::B4(VChipTemplateB4::BigNoMemory),
            "vir05_1c_8g" => VChipScale::B4(VChipTemplateB4::Small),
            _ => panic!("Unknown template name"),
        }
    }
}

impl ShareTarget for VChipScale {
    fn name(&self) -> &'static str {
        VChipScale::name(self)
    }

    fn from_name(name: &str) -> Option<Self> {
        match name {
            "vir12_3c_32g" => Some(VChipScale::B1(VChipTemplateB1B2::Big)),
            "vir06_1c_16g" => Some(VChipScale::B1(VChipTemplateB1B2::Small)),
            "vir03_1c_8g" => Some(VChipScale::B1(VChipTemplateB1B2::Tiny)),
            "vir10_3c_32g" => Some(VChipScale::B3(VChipTemplateB3::Big)),
            "vir05_1c_16g" => Some(VChipScale::B3(VChipTemplateB3::Small)),
            "vir10_3c_16g_m" => Some(VChipScale::B4(VChipTemplateB4::Huge)),
            "vir10_3c_16g" => Some(VChipScale::B4(VChipTemplateB4::Big)),
            "vir10_3c_16g_nm" => Some(VChipScale::B4(VChipTemplateB4::BigNoMemory)),
            "vir05_1c_8g" => Some(VChipScale::B4(VChipTemplateB4::Small)),
            _ => None,
        }
    }
}

impl ChoosableShareTargetContainer for VChipScale {
    type Target = Self;

    fn all_names(&self) -> Vec<&'static str> {
        vec![
            "vir12_3c_32g",
            "vir06_1c_16g",
            "vir03_1c_8g",
            "vir10_3c_32g",
            "vir05_1c_16g",
            "vir10_3c_16g_m",
            "vir10_3c_16g",
            "vir10_3c_16g_nm",
            "vir05_1c_8g",
        ]
    }
}

impl From<&VChipScale> for VChipScale {
    fn from(value: &VChipScale) -> Self {
        *value
    }
}
