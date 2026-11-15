//! `WorkerProxy` related implementation of [MockWorker] for testing purposes.

use std::{path::Path, process::Stdio};

use tokio::process::Command;
use uuid::Uuid;
use worker::mock::MockWorker;

use crate::{
    compute_device::mock::MockVirtualComputeDevice,
    util::worker_proxy::bin_simple_worker_proxy::{
        BinSimpleWorkerProxyImpl, BinSimpleWorkerProxyImplInit, WorkerProxyCapable, WorkerProxyType,
    },
};

#[cfg(feature = "vgpu")]
impl WorkerProxyCapable<MockVirtualComputeDevice> for MockWorker {}

/// A worker proxy for a mock virtual GPU device.
#[cfg(feature = "vgpu")]
pub type MockVirtualGpuWorkerProxy = BinSimpleWorkerProxyImpl<MockVirtualComputeDevice>;

impl BinSimpleWorkerProxyImplInit<MockVirtualComputeDevice> for MockVirtualGpuWorkerProxy {
    const PROXY_TYPE: WorkerProxyType = WorkerProxyType::Mock;

    async fn new<W: WorkerProxyCapable<MockVirtualComputeDevice>>(
        id: Uuid,
        device: MockVirtualComputeDevice,
        model_dir: impl AsRef<Path> + Send,
        _model_config_dir: impl AsRef<Path> + Send,
        bin_worker_path: impl AsRef<Path> + Send,
    ) -> Result<Self, String> {
        let started = std::time::Instant::now();
        tracing::info!(target: "simulation", event = "create_started", %id, slice = device.config(), delay_ms = device.create_delay().as_millis() as u64);
        tokio::time::sleep(device.create_delay()).await;
        let destroy_delay = device.destroy_delay();
        let inference_timeout = device.inference_timeout();
        let control_timeout = device.control_timeout();
        let model_lifecycle =
            serde_json::to_string(&device.model_lifecycle()).map_err(|e| e.to_string())?;
        let execution_noise =
            serde_json::to_string(&device.execution_noise()).map_err(|e| e.to_string())?;
        let model_noise_seeds = device.model_noise_seeds(id)?;
        let noise_slot = crate::experiment_random::worker_noise_slot(id);
        tracing::info!(target: "simulation", event="worker_noise_injected", worker_id=%id,
            noise_slot=?noise_slot, model_noise_seeds=?model_noise_seeds);
        let model_noise_seeds =
            serde_json::to_string(&model_noise_seeds).map_err(|e| e.to_string())?;

        let mut command = Command::new(bin_worker_path.as_ref().to_str().unwrap());
        let process = command
            .args([
                "--id",
                id.to_string().as_str(),
                "--config",
                device.profile_name(),
                "--latency-profile",
                model_dir.as_ref().to_str().unwrap(),
                "--model-lifecycle",
                &model_lifecycle,
                "--execution-noise",
                &execution_noise,
                "--model-noise-seeds",
                &model_noise_seeds,
                "--port",
                "0",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| format!("Failed to spawn worker process: {e}"))?;

        let on_clean_up = Box::new(move || {
            Box::into_pin(Box::new(async move {
                let started = std::time::Instant::now();
                tracing::info!(target: "simulation", event = "destroy_started", %id, delay_ms = destroy_delay.as_millis() as u64);
                tokio::time::sleep(destroy_delay).await;
                tracing::info!(target: "simulation", event = "destroy_finished", %id, elapsed_seconds = started.elapsed().as_secs_f64());
                Ok(())
            })
                as Box<dyn std::future::Future<Output = Result<(), String>> + Send>)
        });
        let proxy = Self::new(0, device, None, process, on_clean_up)
            .with_inference_timeout(inference_timeout)
            .with_control_timeout(control_timeout);
        let proxy = proxy.finish_startup(id).await?;
        tracing::info!(target: "simulation", event = "create_finished", %id, elapsed_seconds = started.elapsed().as_secs_f64());

        Ok(proxy)
    }
}
