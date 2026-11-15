//! 定义了与资源共享目标相关的 Trait 和结构体

use std::{fmt::Debug, hash::Hash};

/// 定义了一个可共享的计算资源目标（如 MIG/vChip 模板）
pub trait ShareTarget: Sized + Clone + Eq + Hash + Debug {
    /// 获取目标的静态名称
    fn name(&self) -> &str;
    /// 从字符串名称创建目标实例
    fn from_name(name: &str) -> Option<Self>;
}

/// 管理一组共享目标
pub trait ChoosableShareTargetContainer {
    /// 共享目标的类型
    type Target: ShareTarget;

    /// 获取所有支持的目标名称列表
    fn all_names(&self) -> Vec<&str>;

    /// 获取所有支持的目标实例列表
    fn all_cases(&self) -> Vec<Self::Target> {
        self.all_names()
            .iter()
            .filter_map(|&name| Self::Target::from_name(name))
            .collect()
    }

    /// 验证一组名称是否都是合法的目标名称
    fn validate_names(names: impl IntoIterator<Item = impl AsRef<str>>) -> Result<(), String> {
        for name in names {
            let name = name.as_ref();
            if Self::Target::from_name(name).is_none() {
                return Err(format!("Invalid share target name: '{}'.", name));
            }
        }
        Ok(())
    }

    /// 从一组名称创建目标实例列表
    fn from_names(names: impl IntoIterator<Item = impl AsRef<str>>) -> Vec<Self::Target> {
        names
            .into_iter()
            .filter_map(|name| Self::Target::from_name(name.as_ref()))
            .collect()
    }

    /// 获取所有支持的目标实例列表中最小的实例（可能多个）
    fn min(&self) -> Vec<Self::Target>
    where
        Self::Target: Ord,
    {
        let cases = self.all_cases();
        if cases.is_empty() {
            vec![]
        } else {
            let min_val = cases.iter().min().unwrap().clone();
            cases.into_iter().filter(|t| t == &min_val).collect()
        }
    }

    /// 获取所有支持的目标实例列表中最大的实例（可能多个）
    fn max(&self) -> Vec<Self::Target>
    where
        Self::Target: Ord,
    {
        let cases = self.all_cases();
        if cases.is_empty() {
            vec![]
        } else {
            let max_val = cases.iter().max().unwrap().clone();
            cases.into_iter().filter(|t| t == &max_val).collect()
        }
    }
}

/// 管理一组可配置的、有序的共享目标
#[derive(Debug, Clone)]
pub struct ConfigurableShareTargets<T> {
    enabled_targets: Vec<T>,
    min_targets: Vec<T>,
    max_targets: Vec<T>,
}

impl<T: ShareTarget + Ord> ConfigurableShareTargets<T> {
    /// 从一组启用的目标创建一个新的可配置共享目标集合
    pub fn new(mut enabled_targets: Vec<T>) -> Self {
        // Sort to easily find min/max and handle duplicates
        enabled_targets.sort();
        enabled_targets.dedup();

        let min_targets = if enabled_targets.is_empty() {
            vec![]
        } else {
            let min_val = enabled_targets.first().unwrap();
            enabled_targets
                .iter()
                .take_while(|&t| t == min_val)
                .cloned()
                .collect()
        };

        let max_targets = if enabled_targets.is_empty() {
            vec![]
        } else {
            let max_val = enabled_targets.last().unwrap();
            enabled_targets
                .iter()
                .rev()
                .take_while(|&t| t == max_val)
                .cloned()
                .collect()
        };

        Self {
            enabled_targets,
            min_targets,
            max_targets,
        }
    }
}

impl<T: ShareTarget + Ord> ChoosableShareTargetContainer for ConfigurableShareTargets<T> {
    type Target = T;

    fn all_names(&self) -> Vec<&str> {
        self.enabled_targets.iter().map(|t| t.name()).collect()
    }

    /// 返回所有启用的目标
    fn all_cases(&self) -> Vec<T> {
        self.enabled_targets.clone()
    }

    /// 返回最小的目标（可能多个）
    fn min(&self) -> Vec<T> {
        self.min_targets.clone()
    }

    /// 返回最大的目标（可能多个）
    fn max(&self) -> Vec<T> {
        self.max_targets.clone()
    }
}

#[allow(missing_docs)]
#[cfg(any(test, feature = "test_utils"))]
pub mod test {
    use crate::compute_device::vgpu::{
        share_target::{ChoosableShareTargetContainer, ShareTarget},
        shareable::test::TestShareTarget,
    };

    impl ShareTarget for TestShareTarget {
        fn name(&self) -> &str {
            match self {
                TestShareTarget::G7Gb80 => "7G80Gb",
                TestShareTarget::G4Gb40 => "4G40Gb",
                TestShareTarget::G3Gb40 => "3G40Gb",
                TestShareTarget::G2Gb20 => "2G20Gb",
                TestShareTarget::G1Gb10 => "1G10Gb",
            }
        }

        fn from_name(name: &str) -> Option<Self> {
            match name {
                "7G80Gb" => Some(TestShareTarget::G7Gb80),
                "4G40Gb" => Some(TestShareTarget::G4Gb40),
                "3G40Gb" => Some(TestShareTarget::G3Gb40),
                "2G20Gb" => Some(TestShareTarget::G2Gb20),
                "1G10Gb" => Some(TestShareTarget::G1Gb10),
                _ => None,
            }
        }
    }

    impl ChoosableShareTargetContainer for TestShareTarget {
        type Target = Self;

        fn all_names(&self) -> Vec<&str> {
            vec!["7G80Gb", "4G40Gb", "3G40Gb", "2G20Gb", "1G10Gb"]
        }

        fn all_cases(&self) -> Vec<Self> {
            vec![
                TestShareTarget::G7Gb80,
                TestShareTarget::G4Gb40,
                TestShareTarget::G3Gb40,
                TestShareTarget::G2Gb20,
                TestShareTarget::G1Gb10,
            ]
        }

        fn min(&self) -> Vec<Self> {
            vec![TestShareTarget::G1Gb10]
        }

        fn max(&self) -> Vec<Self> {
            vec![TestShareTarget::G7Gb80]
        }
    }
}
