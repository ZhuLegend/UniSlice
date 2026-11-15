//! Nvidia 硬件的 DaemonSpec 实现

#[cfg(feature = "mock")]
use compute_node::compute_device::mock::MockComputeDeviceResponse;
use compute_node::compute_device::{
    resource_spec::nv::NvGpuResponse,
    vgpu::{
        resource_spec::nv::mig::{A100Template, NvidiaMigResource, VDeviceScale},
        share_target::{ChoosableShareTargetContainer, ConfigurableShareTargets},
    },
};

use crate::{
    daemon::{
        spec::{DaemonSpec, VScale},
        types::DeviceConfig,
    },
    schedule_system::virtual_compute_node::resources::nvidia_mig::ConstrainedNvidiaMigResource,
};

/// VScale Trait 实现
impl VScale for VDeviceScale {}

/// Nvidia MIG Daemon
pub struct NvidiaMigDaemon;

/// DaemonSpec Trait 实现
impl DaemonSpec for NvidiaMigDaemon {
    type ShareParam = VDeviceScale;

    type Resource = NvidiaMigResource;

    type DeviceState = NvGpuResponse;

    type SchedulerResource = ConstrainedNvidiaMigResource;

    fn service_name() -> &'static str {
        "nv_scheduler_daemon"
    }

    fn create_resource_template(device_config: &DeviceConfig) -> Result<Self::Resource, String> {
        match device_config {
            #[cfg(feature = "nvidia")]
            DeviceConfig::Nvidia {
                resource_template, ..
            } => Ok(*resource_template),
            _ => Err("Expected Nvidia device configuration".to_string()),
        }
    }

    fn create_configurable_targets(
        device_config: &DeviceConfig,
    ) -> Result<ConfigurableShareTargets<Self::ShareParam>, String> {
        match device_config {
            #[cfg(feature = "nvidia")]
            DeviceConfig::Nvidia { share_targets, .. } => {
                let names = if share_targets.is_empty() {
                    // 如果在配置中未指定，则默认使用所有可用的目标
                    let dummy_instance = VDeviceScale::A100(A100Template::Tiny);
                    dummy_instance
                        .all_names()
                        .iter()
                        .map(|s| s.to_string())
                        .collect()
                } else {
                    share_targets.clone()
                };
                let targets = VDeviceScale::from_names(names);
                Ok(ConfigurableShareTargets::new(targets))
            }
            _ => Err("Expected Nvidia device configuration".to_string()),
        }
    }
}

/// Mock Nvidia MIG Daemon
#[cfg(feature = "mock")]
pub struct MockNvidiaMigDaemon;

/// DaemonSpec Trait 实现
#[cfg(feature = "mock")]
impl DaemonSpec for MockNvidiaMigDaemon {
    type ShareParam = VDeviceScale;

    type Resource = NvidiaMigResource;

    type DeviceState = MockComputeDeviceResponse;

    type SchedulerResource = ConstrainedNvidiaMigResource;

    fn service_name() -> &'static str {
        "mock_nv_scheduler_daemon"
    }

    fn create_resource_template(device_config: &DeviceConfig) -> Result<Self::Resource, String> {
        match device_config {
            DeviceConfig::MockNvidia {
                resource_template, ..
            } => Ok(*resource_template),
            _ => Err("Expected Nvidia device configuration".to_string()),
        }
    }

    fn create_configurable_targets(
        device_config: &DeviceConfig,
    ) -> Result<ConfigurableShareTargets<Self::ShareParam>, String> {
        match device_config {
            DeviceConfig::MockNvidia { share_targets, .. } => {
                let names = if share_targets.is_empty() {
                    // 如果在配置中未指定，则默认使用所有可用的目标
                    let dummy_instance = VDeviceScale::A100(A100Template::Tiny);
                    dummy_instance
                        .all_names()
                        .iter()
                        .map(|s| s.to_string())
                        .collect()
                } else {
                    share_targets.clone()
                };
                let targets = VDeviceScale::from_names(names);
                Ok(ConfigurableShareTargets::new(targets))
            }
            _ => Err("Expected Nvidia device configuration".to_string()),
        }
    }
}
