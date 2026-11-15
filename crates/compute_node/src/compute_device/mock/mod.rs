//! A mock implementation of a compute device for testing purposes.

use std::{ops::Deref, sync::Arc, time::Duration};

use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use worker::mock::MockWorker;

use crate::{
    compute_device::{ComputeDevice, VirtualGpu, config::MockWorkerModelLoadConfig},
    compute_node::r#virtual::base::{DeviceManageHelper, DeviceState},
    runnable_on::{IndependentModelLoadConfig, RunnableOn},
};

#[cfg(all(feature = "vgpu", any(feature = "nvidia", feature = "resource_spec")))]
pub mod nv;

#[cfg(all(feature = "vgpu", any(feature = "huawei", feature = "resource_spec")))]
pub mod hw;

/// Mock compute device for testing purposes.
#[derive(Debug, Clone)]
pub struct MockComputeDevice {
    config_name: String,
    simulation: Option<Arc<MockDeviceSettings>>,
}

impl MockComputeDevice {
    /// Creates a new mock compute device with the given configuration name.
    pub fn new(config_name: String) -> Self {
        Self {
            config_name,
            simulation: None,
        }
    }

    /// Construct a device with explicit lifecycle and profile settings.
    pub fn with_simulation(config_name: String, settings: MockDeviceSettings) -> Self {
        Self {
            config_name,
            simulation: Some(Arc::new(settings)),
        }
    }

    /// Returns the configuration name of the mock compute device.
    pub fn config_name(&self) -> &str {
        &self.config_name
    }
}

/// Request type for the mock compute device.
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct MockComputeDeviceRequest;

/// Response type for the mock compute device.
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct MockComputeDeviceResponse {
    /// The configuration name of [MockComputeDevice].
    pub config_name: String,
}

impl DeviceState for MockComputeDeviceResponse {}

impl ComputeDevice for MockComputeDevice {
    type Request = MockComputeDeviceRequest;

    type Response = MockComputeDeviceResponse;

    async fn request(&self, _request: Self::Request) -> Self::Response {
        let config_name = self.config_name.clone();
        // Simulate processing the request and returning a response.
        MockComputeDeviceResponse { config_name }
    }

    fn logic_index(&self) -> Option<u32> {
        None
    }
}

impl RunnableOn<MockComputeDevice> for MockWorker {
    type DeviceIndependentConfig = MockWorkerModelLoadConfig;

    fn device_config_getter(
        device: &MockComputeDevice,
    ) -> impl FnOnce() -> <Self::DeviceIndependentConfig as IndependentModelLoadConfig>::RequireDevice
    {
        || device.config_name.clone()
    }
}

/// Virtual GPU mock for testing purposes.
#[derive(Debug, Clone)]
pub struct MockVirtualComputeDevice {
    host_device: MockComputeDevice,
    config: String,
    _resource: Arc<dyn std::fmt::Debug + Send + Sync>,
}

impl MockVirtualComputeDevice {
    /// Simulated creation time after reserving resources.
    pub fn create_delay(&self) -> Duration {
        self.host_device
            .simulation
            .as_ref()
            .map_or(Duration::ZERO, |s| s.create_delay)
    }
    /// Simulated destruction time before releasing resources.
    pub fn destroy_delay(&self) -> Duration {
        self.host_device
            .simulation
            .as_ref()
            .map_or(Duration::ZERO, |s| s.destroy_delay)
    }
    /// Response budget passed through to the child worker's inference RPC.
    pub fn inference_timeout(&self) -> Duration {
        self.host_device
            .simulation
            .as_ref()
            .map_or(Duration::from_secs(60), |s| s.inference_timeout)
    }
    /// Control budget for model load/unload in the child.
    pub fn control_timeout(&self) -> Duration {
        self.host_device
            .simulation
            .as_ref()
            .map_or(Duration::from_secs(60), |s| s.control_timeout)
    }
    /// Resolved costs for this slice only.
    pub fn model_lifecycle(
        &self,
    ) -> std::collections::BTreeMap<String, worker::mock::ModelLifecycle> {
        self.host_device
            .simulation
            .as_ref()
            .and_then(|s| s.model_lifecycle.get(&self.config))
            .cloned()
            .unwrap_or_default()
    }
    /// Resolve execution variation without exposing the node topology to workers.
    pub fn execution_noise(&self) -> worker::mock::ExecutionNoise {
        self.host_device
            .simulation
            .as_ref()
            .map_or(Default::default(), |s| s.execution_noise)
    }
    /// Final model-generation seeds from the runner's external noise table.
    #[cfg(feature = "vgpu")]
    pub fn model_noise_seeds(
        &self,
        worker_id: Uuid,
    ) -> Result<std::collections::BTreeMap<String, Vec<u64>>, String> {
        match self
            .host_device
            .simulation
            .as_ref()
            .and_then(|s| s.random_namespace.as_deref())
        {
            Some(namespace) => {
                crate::experiment_random::worker_noise(namespace, &self.config, worker_id)
            }
            None => Ok(Default::default()),
        }
    }
    /// Exact CSV key selected by the explicit alias map.
    pub fn profile_name(&self) -> &str {
        self.host_device
            .simulation
            .as_ref()
            .and_then(|s| s.aliases.get(&self.config))
            .map(String::as_str)
            .unwrap_or(&self.config)
    }
    /// get the config of [MockVirtualComputeDevice]
    pub fn config(&self) -> &str {
        &self.config
    }
}

impl VirtualGpu for MockVirtualComputeDevice {
    type HostDevice = MockComputeDevice;
    type State = String;
    type Error = String;

    fn host_device(&self) -> impl Deref<Target = Self::HostDevice> + '_ + Send {
        &self.host_device
    }

    async fn state(&self) -> Self::State {
        self.config.clone()
    }
}

impl RunnableOn<MockVirtualComputeDevice> for MockWorker {
    type DeviceIndependentConfig = MockWorkerModelLoadConfig;

    fn device_config_getter(
        device: &MockVirtualComputeDevice,
    ) -> impl FnOnce() -> <Self::DeviceIndependentConfig as IndependentModelLoadConfig>::RequireDevice
    {
        let config = device.config.clone();
        || config
    }
}

impl DeviceManageHelper for MockComputeDevice {
    fn get_device(id: usize) -> (Uuid, Self) {
        let device = MockComputeDevice::new(format!("mock_device_{}", id));
        (Uuid::new_v4(), device)
    }

    fn get_device_state(device: impl Deref<Target = Self>) -> <Self as ComputeDevice>::Response {
        let config_name = device.config_name.clone();
        MockComputeDeviceResponse { config_name }
    }
}

/// Applied only to explicitly configured simulation devices; ordinary unit mocks stay instantaneous.
#[derive(Debug, Clone)]
pub struct MockDeviceSettings {
    /// External noise-table namespace: node name / physical device index.
    pub random_namespace: Option<String>,
    /// Inference variation; ordinary mocks default to deterministic execution.
    pub execution_noise: worker::mock::ExecutionNoise,
    /// Creation time per slice.
    pub create_delay: Duration,
    /// Destruction time per slice.
    pub destroy_delay: Duration,
    /// Response budget for the child worker's inference RPC.
    pub inference_timeout: Duration,
    /// Control RPC budget.
    pub control_timeout: Duration,
    /// Slice -> model -> resolved lifecycle costs.
    pub model_lifecycle: std::collections::BTreeMap<
        String,
        std::collections::BTreeMap<String, worker::mock::ModelLifecycle>,
    >,
    /// Allowed slice configurations.
    pub share_targets: Vec<String>,
    /// Resource name to measured profile name.
    pub aliases: std::collections::HashMap<String, String>,
}

/// One reservation shared by all clones of a mock slice.
#[derive(Debug)]
pub(super) struct MockResourceLease<
    R: super::vgpu::ShareableComputeDeviceResource + std::fmt::Debug,
> {
    reservation: Option<super::vgpu::SharedResource<R>>,
    id: Uuid,
    amount: R,
}
impl<R: super::vgpu::ShareableComputeDeviceResource + std::fmt::Debug> MockResourceLease<R> {
    fn new(reservation: super::vgpu::SharedResource<R>, amount: R) -> Self {
        let id = Uuid::new_v4();
        tracing::info!(target: "simulation", event = "resource_reserved", reservation_id = %id, amount = ?amount);
        Self {
            reservation: Some(reservation),
            id,
            amount,
        }
    }
}
impl<R: super::vgpu::ShareableComputeDeviceResource + std::fmt::Debug> Drop
    for MockResourceLease<R>
{
    fn drop(&mut self) {
        drop(self.reservation.take());
        tracing::info!(target: "simulation", event = "resource_released", reservation_id = %self.id, amount = ?self.amount);
    }
}
