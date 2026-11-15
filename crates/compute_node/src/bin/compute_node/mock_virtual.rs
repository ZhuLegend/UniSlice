//! CSV-driven virtual GPU node; no hardware runtime is required.
use std::{collections::HashMap, path::PathBuf, time::Duration};

use clap::Parser;
use compute_node::{
    compute_device::{
        mock::{MockComputeDevice, MockDeviceSettings},
        vgpu::{ShareableComputeDeviceResource, share_target::ChoosableShareTargetContainer},
    },
    compute_node::{mock::MockVirtualGpuWorkerProxy, r#virtual::base::BaseVirtualComputeNode},
    simulation::SimulationConfig,
    util::bin_compute_node::{BinComputeNodeWrapper, shutdown_signal},
};
use tokio::net::TcpListener;
use worker::mock::MockWorker;

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// Shared simulation JSON; paths inside it are relative to that file.
    #[arg(long)]
    simulation_config: PathBuf,
    /// Name of the node in the shared simulation JSON.
    #[arg(long)]
    node: String,
    #[arg(long)]
    bin_worker_path: PathBuf,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulation_config_and_node_are_required() {
        let required = ["mock_virtual", "--bin-worker-path", "worker"];
        assert!(Cli::try_parse_from(required).is_err());
        assert!(
            Cli::try_parse_from(
                required
                    .into_iter()
                    .chain(["--simulation-config", "simulation.json",])
            )
            .is_err()
        );
        assert!(
            Cli::try_parse_from(required.into_iter().chain([
                "--simulation-config",
                "simulation.json",
                "--node",
                "nvidia",
            ]))
            .is_ok()
        );
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .json()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let cli = Cli::parse();
    let config = SimulationConfig::load(&cli.simulation_config)?;
    config.profiles()?;
    let node = config
        .nodes
        .iter()
        .find(|node| node.name == cli.node)
        .ok_or("--node must name a configured simulation node")?
        .clone();
    let settings = MockDeviceSettings {
        random_namespace: None,
        create_delay: Duration::from_millis(config.create_delay_ms),
        destroy_delay: Duration::from_millis(config.destroy_delay_ms),
        inference_timeout: Duration::from_millis(config.inference_timeout_ms),
        control_timeout: Duration::from_millis(config.control_timeout_ms),
        model_lifecycle: config.model_costs(&node)?,
        execution_noise: config.execution_noise,
        share_targets: node.share_targets.clone(),
        aliases: config.profile_aliases.clone(),
    };
    let identity = config.node_identity(&node)?;
    let listener = TcpListener::bind(("127.0.0.1", node.port)).await?;
    let resource_json = serde_json::to_value(&node.resource)?;
    macro_rules! serve {
        ($device:ty, $resource:ty, $scale:ty, $proxy:ty) => {{
            let capacity: $resource = serde_json::from_value(resource_json)?;
            <$scale>::validate_names(&node.share_targets)?;
            for target in <$scale>::from_names(&node.share_targets) {
                if !capacity.can_share(&(&target).into()) {
                    return Err("whitelisted slice exceeds node capacity".into());
                }
            }
            let devices: HashMap<_, _> = (0..node.gpus)
                .map(|i| {
                    let mut settings = settings.clone();
                    settings.random_namespace = Some(format!("{}/{}", node.name, i));
                    (
                        uuid::Uuid::new_v4(),
                        (
                            i as u32,
                            <$device>::with_capacity(
                                MockComputeDevice::with_simulation(identity.clone(), settings),
                                capacity,
                            ),
                        ),
                    )
                })
                .collect();
            let wrapper = BinComputeNodeWrapper::builder()
                .wrap(
                    BaseVirtualComputeNode::<MockWorker, $device, $proxy>::from_devices(
                        devices,
                        config.latency_csv.clone(),
                        config.latency_csv,
                        cli.bin_worker_path,
                    ),
                )
                .max_frame_size(64 * 4 * 3 * 224 * 224 + 1024)
                .build();
            wrapper.serve_until(listener, shutdown_signal()).await?;
        }};
    }
    match node.kind.as_str() {
        "nvidia" => serve!(
            compute_node::compute_device::mock::nv::MockNvidiaMigShareableGpu,
            compute_node::compute_device::vgpu::resource_spec::nv::mig::NvidiaMigResource,
            compute_node::compute_device::vgpu::resource_spec::nv::mig::VDeviceScale,
            MockVirtualGpuWorkerProxy
        ),
        "huawei" => serve!(
            compute_node::compute_device::mock::hw::MockHuaweiMigShareableGpu,
            compute_node::compute_device::vgpu::resource_spec::hw::HuaweiMigResource,
            compute_node::compute_device::vgpu::resource_spec::hw::VChipScale,
            MockVirtualGpuWorkerProxy
        ),
        _ => unreachable!(),
    }
    Ok(())
}
