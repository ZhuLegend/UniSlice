//! 华为硬件的 DaemonSpec 实现

#[cfg(feature = "mock")]
use compute_node::compute_device::mock::MockComputeDeviceResponse;
use compute_node::compute_device::{
    resource_spec::hw::HwGpuResponse,
    vgpu::{
        resource_spec::hw::{HuaweiMigResource, VChipScale},
        share_target::{ChoosableShareTargetContainer, ConfigurableShareTargets},
    },
};

use crate::{
    daemon::{
        spec::{DaemonSpec, VScale},
        types::DeviceConfig,
    },
    schedule_system::virtual_compute_node::resources::huawei_mig::ConstrainedHuaweiMigResource,
};

/// VScale Trait 实现
impl VScale for VChipScale {}

/// Huawei MIG Daemon
pub struct HuaweiMigDaemon;

/// DaemonSpec Trait 实现
impl DaemonSpec for HuaweiMigDaemon {
    type ShareParam = VChipScale;

    type Resource = HuaweiMigResource;

    type DeviceState = HwGpuResponse;

    type SchedulerResource = ConstrainedHuaweiMigResource;

    fn service_name() -> &'static str {
        "hw_scheduler_daemon"
    }

    fn create_resource_template(device_config: &DeviceConfig) -> Result<Self::Resource, String> {
        match device_config {
            #[cfg(feature = "huawei")]
            DeviceConfig::Huawei {
                resource_template, ..
            } => Ok(*resource_template),
            _ => Err("Expected Huawei device configuration".to_string()),
        }
    }

    fn create_configurable_targets(
        device_config: &DeviceConfig,
    ) -> Result<ConfigurableShareTargets<Self::ShareParam>, String> {
        match device_config {
            #[cfg(feature = "huawei")]
            DeviceConfig::Huawei { share_targets, .. } => {
                let names = if share_targets.is_empty() {
                    // 如果在配置中未指定，则默认使用所有可用的目标
                    let dummy_instance =
                        VChipScale::B1(compute_node::compute_device::vgpu::resource_spec::hw::VChipTemplateB1B2::Tiny);
                    dummy_instance
                        .all_names()
                        .iter()
                        .map(|s| s.to_string())
                        .collect()
                } else {
                    share_targets.clone()
                };
                let targets = VChipScale::from_names(names);
                Ok(ConfigurableShareTargets::new(targets))
            }
            _ => Err("Expected Huawei device configuration".to_string()),
        }
    }
}

/// Mock Huawei MIG Daemon
#[cfg(feature = "mock")]
pub struct MockHuaweiMigDaemon;

/// DaemonSpec Trait 实现
#[cfg(feature = "mock")]
impl DaemonSpec for MockHuaweiMigDaemon {
    type ShareParam = VChipScale;

    type Resource = HuaweiMigResource;

    type DeviceState = MockComputeDeviceResponse;

    type SchedulerResource = ConstrainedHuaweiMigResource;

    fn service_name() -> &'static str {
        "mock_hw_scheduler_daemon"
    }

    fn create_resource_template(device_config: &DeviceConfig) -> Result<Self::Resource, String> {
        match device_config {
            DeviceConfig::MockHuawei {
                resource_template, ..
            } => Ok(*resource_template),
            _ => Err("Expected Huawei device configuration".to_string()),
        }
    }

    fn create_configurable_targets(
        device_config: &DeviceConfig,
    ) -> Result<ConfigurableShareTargets<Self::ShareParam>, String> {
        match device_config {
            DeviceConfig::MockHuawei { share_targets, .. } => {
                let names = if share_targets.is_empty() {
                    // 如果在配置中未指定，则默认使用所有可用的目标
                    let dummy_instance =
                        VChipScale::B1(compute_node::compute_device::vgpu::resource_spec::hw::VChipTemplateB1B2::Tiny);
                    dummy_instance
                        .all_names()
                        .iter()
                        .map(|s| s.to_string())
                        .collect()
                } else {
                    share_targets.clone()
                };
                let targets = VChipScale::from_names(names);
                Ok(ConfigurableShareTargets::new(targets))
            }
            _ => Err("Expected Huawei device configuration".to_string()),
        }
    }
}
