//! 配置管理模块

#[cfg(feature = "huawei")]
use compute_node::compute_device::vgpu::resource_spec::hw::VChipScale;
#[cfg(feature = "nvidia")]
use compute_node::compute_device::vgpu::resource_spec::nv::mig::VDeviceScale;
use tokio::fs;

use crate::daemon::types::Config;

/// 加载配置文件
pub async fn load_config(
    config_path: &std::path::Path,
) -> Result<Config, Box<dyn std::error::Error>> {
    let config_content = fs::read_to_string(config_path).await?;
    let config: Config = toml::from_str(&config_content)?;

    tracing::info!("已加载 {} 个远程节点", config.remote_nodes.len());

    Ok(config)
}

/// 验证配置有效性
pub fn validate_config(config: &Config) -> Result<(), String> {
    if config.remote_nodes.is_empty() {
        return Err("配置文件中必须至少包含一个远程节点".to_string());
    }

    // 验证设备组合配置
    if config.device_combinations.is_empty() {
        return Err("配置文件中必须至少包含一个设备组合".to_string());
    }

    for device_config in config.device_combinations.values() {
        use compute_node::compute_device::vgpu::share_target::ChoosableShareTargetContainer;

        // 验证设备特定配置
        match device_config {
            #[cfg(feature = "huawei")]
            crate::daemon::types::DeviceConfig::Huawei { share_targets, .. } => {
                if !share_targets.is_empty() {
                    VChipScale::validate_names(share_targets)?;
                }
            }
            #[cfg(feature = "nvidia")]
            crate::daemon::types::DeviceConfig::Nvidia { share_targets, .. } => {
                if !share_targets.is_empty() {
                    VDeviceScale::validate_names(share_targets)?;
                }
            }
            #[cfg(all(feature = "mock", feature = "huawei"))]
            crate::daemon::types::DeviceConfig::MockHuawei { share_targets, .. } => {
                if !share_targets.is_empty() {
                    VChipScale::validate_names(share_targets)?;
                }
            }
            #[cfg(all(feature = "mock", feature = "nvidia"))]
            crate::daemon::types::DeviceConfig::MockNvidia { share_targets, .. } => {
                if !share_targets.is_empty() {
                    VDeviceScale::validate_names(share_targets)?;
                }
            }
        }
    }

    // 验证远程节点配置
    for (name, node_config) in &config.remote_nodes {
        if node_config.ip.is_empty() {
            return Err(format!("节点 '{}' 的IP地址不能为空", name));
        }

        if node_config.port == 0 {
            return Err(format!("节点 '{}' 的端口号无效", name));
        }

        // 验证模板名称是否存在于设备组合中
        if !config
            .device_combinations
            .keys()
            .any(|name| name == &node_config.template_name)
        {
            return Err(format!(
                "节点 '{}' 引用的模板 '{}' 不存在",
                name, node_config.template_name
            ));
        }
    }

    // 验证批次大小范围
    if config.batch_size_range.0 == 0 || config.batch_size_range.1 == 0 {
        return Err("批次大小范围必须大于0".to_string());
    }

    if config.batch_size_range.0 > config.batch_size_range.1 {
        return Err("批次大小范围的最小值不能大于最大值".to_string());
    }

    Ok(())
}

#[cfg(all(test, feature = "mock", feature = "nvidia", feature = "huawei"))]
mod simulation_tests {
    use serde_json::json;

    use super::*;

    fn configuration() -> (Config, compute_node::simulation::SimulationConfig) {
        let simulation = compute_node::simulation::SimulationConfig::load(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../resource/simulation/heterogeneous.json"),
        )
        .unwrap();
        let mut devices = serde_json::Map::new();
        let mut remotes = serde_json::Map::new();
        for node in &simulation.nodes {
            devices.insert(node.template_name.clone(), json!({
                "type": if node.kind == "nvidia" { "MockNvidia" } else { "MockHuawei" },
                "config": { "share_targets": node.share_targets, "resource_template": node.resource }
            }));
            remotes.insert(
                node.name.clone(),
                json!({
                    "ip": "127.0.0.1", "port": node.port, "template_name": node.template_name
                }),
            );
        }
        let rates: std::collections::HashMap<_, _> =
            simulation.models.iter().map(|m| (m, 100.0)).collect();
        let source: toml::Value = toml::from_str(
            &std::fs::read_to_string(simulation.batching_config.as_ref().unwrap()).unwrap(),
        )
        .unwrap();
        let config = serde_json::from_value(json!({
            "batch_extractor_strategy": source["batch_extractor_strategy"],
            "device_combinations": devices, "remote_nodes": remotes,
            "batch_size_range": [1,32], "init_arrival_rate": rates
        }))
        .unwrap();
        (config, simulation)
    }

    #[test]
    fn simulation_budgets_and_identity_reach_every_remote() {
        let (mut config, simulation) = configuration();
        configure_simulation(&mut config, &simulation).unwrap();
        for node in &simulation.nodes {
            let remote = &config.remote_nodes[&node.name];
            assert_eq!(remote.control_timeout_ms, 300_000);
            assert_eq!(remote.inference_timeout_ms, 300_000);
            assert_eq!(
                remote.expected_simulation_identity.as_ref(),
                Some(&simulation.node_identity(node).unwrap())
            );
        }
    }

    #[test]
    fn simulation_rejects_scheduler_capacity_or_model_mismatches() {
        let (mut config, simulation) = configuration();
        if let crate::daemon::types::DeviceConfig::MockNvidia {
            resource_template, ..
        } = config.device_combinations.get_mut("a100").unwrap()
        {
            resource_template.ce = 3;
        }
        assert!(
            configure_simulation(&mut config, &simulation)
                .unwrap_err()
                .contains("device configuration")
        );
        let (mut config, simulation) = configuration();
        config.init_arrival_rate.remove("resnet50");
        assert!(
            configure_simulation(&mut config, &simulation)
                .unwrap_err()
                .contains("served models")
        );
    }

    #[test]
    fn simulation_rejects_pool_or_batching_drift() {
        let (mut config, simulation) = configuration();
        let device = config.device_combinations["a100"].clone();
        config
            .device_combinations
            .insert("another_pool".into(), device);
        config.remote_nodes.get_mut("a100").unwrap().template_name = "another_pool".into();
        assert!(
            configure_simulation(&mut config, &simulation)
                .unwrap_err()
                .contains("scheduling pool")
        );

        let (mut config, simulation) = configuration();
        config.batch_extractor_strategy = compute_node::compute_node::r#virtual::virtual_compute_node::components::request_scheduler::batch_size_extractor::BatchExtractorStrategy::split_and_satisfy();
        assert!(
            configure_simulation(&mut config, &simulation)
                .unwrap_err()
                .contains("batching policy")
        );
    }
}

/// Validate the scheduler against the same topology consumed by the virtual nodes.
#[cfg(feature = "mock")]
pub fn configure_simulation(
    config: &mut Config,
    simulation: &compute_node::simulation::SimulationConfig,
) -> Result<(), String> {
    if let Some(source) = &simulation.batching_config {
        let experiment: toml::Value = toml::from_str(
            &std::fs::read_to_string(source).map_err(|e| format!("batching source: {e}"))?,
        )
        .map_err(|e| format!("batching source: {e}"))?;
        let policy = experiment
            .get("batch_extractor_strategy")
            .ok_or("batching source has no batch_extractor_strategy")?;
        let policy: compute_node::compute_node::r#virtual::virtual_compute_node::components::request_scheduler::batch_size_extractor::BatchExtractorStrategy = policy.clone().try_into()
            .map_err(|e| format!("batching source policy: {e}"))?;
        if serde_json::to_value(&config.batch_extractor_strategy).map_err(|e| e.to_string())?
            != serde_json::to_value(&policy).map_err(|e| e.to_string())?
        {
            return Err("scheduler batching policy differs from experiment source".into());
        }
    }
    if config.remote_nodes.len() != simulation.nodes.len()
        || [config.batch_size_range.0, config.batch_size_range.1] != simulation.batch_size_range
    {
        return Err("scheduler topology/batch range differs from simulation JSON".into());
    }
    if config.init_arrival_rate.len() != simulation.models.len()
        || simulation.models.iter().any(|m| {
            config
                .init_arrival_rate
                .get(m)
                .is_none_or(|r| !r.is_finite() || *r <= 0.0)
        })
    {
        return Err("simulation needs positive initial rates for exactly its served models".into());
    }
    for node in &simulation.nodes {
        let remote = config
            .remote_nodes
            .get_mut(&node.name)
            .ok_or("simulation node missing from scheduler")?;
        if remote.template_name != node.template_name {
            return Err(format!(
                "scheduler scheduling pool differs for {}",
                node.name
            ));
        }
        let device = config
            .device_combinations
            .get(&remote.template_name)
            .ok_or("missing device template")?;
        let value = serde_json::to_value(device).map_err(|e| e.to_string())?;
        let expected_type = if node.kind == "nvidia" {
            "MockNvidia"
        } else {
            "MockHuawei"
        };
        if value["type"] != expected_type
            || value["config"]["share_targets"]
                != serde_json::to_value(&node.share_targets).unwrap()
            || value["config"]["resource_template"] != serde_json::to_value(&node.resource).unwrap()
            || remote.port != node.port
        {
            return Err(format!(
                "scheduler device configuration differs for {}",
                node.name
            ));
        }
        remote.control_timeout_ms = simulation.control_timeout_ms;
        remote.inference_timeout_ms = simulation.inference_timeout_ms;
        remote.expected_simulation_identity = Some(simulation.node_identity(node)?);
    }
    Ok(())
}
