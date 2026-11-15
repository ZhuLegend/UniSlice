//! Shared configuration for CSV simulation; independent of command-line parsing.
use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
};

use latency_profile::{LatencyData, apply_aliases, curve, interpolate, load_csv};
use serde::{Deserialize, Serialize};

fn default_delay() -> u64 {
    15_000
}
fn default_timeout() -> u64 {
    300_000
}
fn default_batch_range() -> [u32; 2] {
    [1, 32]
}

/// One source of truth for the simulated topology and profiler-name mapping.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SimulationConfig {
    /// CSV path, relative to the configuration file.
    pub latency_csv: std::path::PathBuf,
    /// Experiment TOML supplying the batching policy; snapshotted by the runner.
    #[serde(default)]
    pub batching_config: Option<std::path::PathBuf>,
    #[serde(default)]
    /// Resource-profile names mapped to historical CSV keys.
    pub profile_aliases: HashMap<String, String>,
    #[serde(default = "default_delay")]
    /// Reservation to worker-process creation delay in milliseconds.
    pub create_delay_ms: u64,
    #[serde(default = "default_delay")]
    /// Process exit to resource-release delay in milliseconds.
    pub destroy_delay_ms: u64,
    #[serde(default)]
    /// Default and overridden model costs, independent of inference latency.
    pub model_lifecycle: ModelLifecycleConfig,
    #[serde(default)]
    /// Mean-preserving lognormal variation, applied only by mock inference.
    pub execution_noise: worker::mock::ExecutionNoise,
    #[serde(default = "default_timeout")]
    /// Per-control-RPC response budget, including lifecycle operations.
    pub control_timeout_ms: u64,
    #[serde(default = "default_timeout")]
    /// Per-inference-RPC response budget.
    pub inference_timeout_ms: u64,
    #[serde(default = "default_batch_range")]
    /// Supported internal batch interval; the current AIMD requires [1, 32].
    pub batch_size_range: [u32; 2],
    /// Models served by this simulation.
    pub models: Vec<String>,
    /// Physical resource nodes, grouped into logical pools by template_name.
    pub nodes: Vec<NodeConfig>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
/// One named group of devices with the same capacity and allocation whitelist.
pub struct NodeConfig {
    /// Unique node name.
    pub name: String,
    /// Shared logical scheduling pool, using the real backend's template grouping.
    pub template_name: String,
    /// nvidia or huawei; resource units match the repository's resource_spec module.
    pub kind: String,
    /// Number of resource pools with this capacity.
    pub gpus: usize,
    /// Loopback TCP listener port.
    pub port: u16,
    /// Capacity in the device resource specification units.
    pub resource: BTreeMap<String, u32>,
    /// Explicit allocation whitelist, independent of CSV coverage.
    pub share_targets: Vec<String>,
}

/// Uncalibrated defaults; override them with measurements when available.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelLifecycleConfig {
    /// Loading and warmup time until ready, in milliseconds.
    pub load_ms: u64,
    /// Unloading time after draining inference, in milliseconds.
    pub unload_ms: u64,
    #[serde(default)]
    /// More-specific costs selected per configured device/profile/model.
    pub overrides: Vec<ModelLifecycleOverride>,
}
impl Default for ModelLifecycleConfig {
    fn default() -> Self {
        Self {
            load_ms: 1_000,
            unload_ms: 100,
            overrides: Vec::new(),
        }
    }
}
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
/// One optional selector tuple and one or both overridden costs.
pub struct ModelLifecycleOverride {
    /// Optional device kind selector (nvidia or huawei).
    pub device: Option<String>,
    /// Optional resource profile name selector, before alias resolution.
    pub profile: Option<String>,
    /// Optional model-name selector.
    pub model: Option<String>,
    /// Loading and warmup time until ready, in milliseconds.
    pub load_ms: Option<u64>,
    /// Unloading time after draining inference, in milliseconds.
    pub unload_ms: Option<u64>,
}
impl ModelLifecycleConfig {
    /// Most-specific matching selector wins separately for each cost. Equal
    /// specificity is rejected, so array order cannot silently alter semantics.
    pub fn resolve(
        &self,
        device: &str,
        profile: &str,
        model: &str,
    ) -> Result<worker::mock::ModelLifecycle, String> {
        let matching: Vec<_> = self
            .overrides
            .iter()
            .filter(|item| {
                item.device.as_deref().is_none_or(|s| s == device)
                    && item.profile.as_deref().is_none_or(|s| s == profile)
                    && item.model.as_deref().is_none_or(|s| s == model)
            })
            .collect();
        let select = |default, load| -> Result<u64, String> {
            let mut values: Vec<_> = matching
                .iter()
                .filter_map(|item| {
                    let rank = usize::from(item.device.is_some())
                        + usize::from(item.profile.is_some())
                        + usize::from(item.model.is_some());
                    (if load { item.load_ms } else { item.unload_ms }).map(|v| (rank, v))
                })
                .collect();
            values.sort_by_key(|v| v.0);
            if values.len() >= 2 && values[values.len() - 1].0 == values[values.len() - 2].0 {
                return Err(format!(
                    "ambiguous lifecycle overrides for {device}/{profile}/{model}"
                ));
            }
            Ok(values.last().map_or(default, |v| v.1))
        };
        Ok(worker::mock::ModelLifecycle {
            load_ms: select(self.load_ms, true)?,
            unload_ms: select(self.unload_ms, false)?,
        })
    }
}

impl SimulationConfig {
    /// Relative data/configuration paths are relative to this JSON file.
    pub fn load(path: &Path) -> Result<Self, String> {
        let mut value: Self =
            serde_json::from_str(&std::fs::read_to_string(path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        if value.latency_csv.is_relative() {
            value.latency_csv = path
                .parent()
                .unwrap_or(Path::new("."))
                .join(&value.latency_csv);
        }
        if let Some(source) = &mut value.batching_config {
            if source.is_relative() {
                *source = path.parent().unwrap_or(Path::new(".")).join(&*source);
            }
        }
        value.validate()?;
        Ok(value)
    }
    /// Validate capacities, selectors and per-operation time budgets.
    pub fn validate(&self) -> Result<(), String> {
        self.execution_noise.validate()?;
        if self.nodes.is_empty()
            || self.models.is_empty()
            || self.batch_size_range[0] == 0
            || self.batch_size_range[0] > self.batch_size_range[1]
        {
            return Err("simulation needs nodes, models and a positive batch range".into());
        }
        if self.control_timeout_ms
            <= self
                .create_delay_ms
                .max(self.destroy_delay_ms)
                .saturating_add(10_000)
            || self.inference_timeout_ms == 0
        {
            return Err("control timeout must exceed each lifecycle delay by more than 10 seconds; inference timeout must be positive".into());
        }
        if self.batch_size_range != [1, 32] {
            return Err(
                "current AIMD controller requires simulation batch_size_range [1,32]".into(),
            );
        }
        let mut models = std::collections::HashSet::new();
        if self
            .models
            .iter()
            .any(|m| m.is_empty() || *m != m.to_ascii_lowercase() || !models.insert(m))
        {
            return Err("simulation model names must be unique lowercase CSV keys".into());
        }
        let mut names = std::collections::HashSet::new();
        let mut ports = std::collections::HashSet::new();
        let mut templates = HashMap::<&str, &NodeConfig>::new();
        for node in &self.nodes {
            if node.gpus == 0
                || node.name.is_empty()
                || node.template_name.is_empty()
                || node.port < 1024
                || !names.insert(&node.name)
                || !ports.insert(node.port)
                || node.share_targets.is_empty()
                || !["nvidia", "huawei"].contains(&node.kind.as_str())
                || node.resource.values().any(|&v| v == 0)
            {
                return Err(format!(
                    "invalid or duplicate simulation node {}",
                    node.name
                ));
            }
            if let Some(other) = templates.insert(&node.template_name, node) {
                if node.kind != other.kind
                    || node.resource != other.resource
                    || node.share_targets != other.share_targets
                {
                    return Err(format!(
                        "inconsistent device template {}",
                        node.template_name
                    ));
                }
            }
            let expected: Vec<&str> = if node.kind == "nvidia" {
                vec!["ce", "memory"]
            } else {
                vec!["ai_core", "ai_cpu", "memory"]
            };
            if node.resource.len() != expected.len()
                || expected.iter().any(|k| !node.resource.contains_key(*k))
            {
                return Err(format!("invalid capacity fields on node {}", node.name));
            }
            if node
                .share_targets
                .iter()
                .any(|s| s == "full" || s == "7g.80gb")
            {
                return Err(
                    "whole-card allocation is disabled; full profiles are reference data only"
                        .into(),
                );
            }
        }
        for item in &self.model_lifecycle.overrides {
            if (item.load_ms.is_none() && item.unload_ms.is_none())
                || !self.nodes.iter().any(|node| {
                    item.device.as_deref().is_none_or(|s| s == node.kind)
                        && node
                            .share_targets
                            .iter()
                            .any(|p| item.profile.as_ref().is_none_or(|s| s == p))
                        && self
                            .models
                            .iter()
                            .any(|m| item.model.as_ref().is_none_or(|s| s == m))
                })
            {
                return Err(
                    "lifecycle override has no cost or matches no configured device/profile/model"
                        .into(),
                );
            }
        }
        for node in &self.nodes {
            for profile in &node.share_targets {
                for model in &self.models {
                    let costs = self.model_lifecycle.resolve(&node.kind, profile, model)?;
                    if self.control_timeout_ms
                        <= costs
                            .load_ms
                            .max(self.destroy_delay_ms)
                            .saturating_add(costs.unload_ms)
                            .saturating_add(10_000)
                    {
                        return Err(format!(
                            "control timeout too short for model lifecycle {model}/{profile}"
                        ));
                    }
                }
            }
        }
        Ok(())
    }
    /// Resolve only this node's worker settings; children never receive the topology.
    pub fn model_costs(
        &self,
        node: &NodeConfig,
    ) -> Result<BTreeMap<String, BTreeMap<String, worker::mock::ModelLifecycle>>, String> {
        node.share_targets
            .iter()
            .map(|profile| {
                let models = self
                    .models
                    .iter()
                    .map(|model| {
                        Ok((
                            model.clone(),
                            self.model_lifecycle.resolve(&node.kind, profile, model)?,
                        ))
                    })
                    .collect::<Result<_, String>>()?;
                Ok((profile.clone(), models))
            })
            .collect()
    }
    /// Load and validate all configured model/profile/batch combinations.
    pub fn profiles(&self) -> Result<LatencyData, String> {
        let mut data = load_csv(&self.latency_csv)?;
        apply_aliases(&mut data, &self.profile_aliases);
        for node in &self.nodes {
            for profile in &node.share_targets {
                for model in &self.models {
                    let c = curve(&data, model, profile)?;
                    interpolate(c, self.batch_size_range[0])?;
                    interpolate(c, self.batch_size_range[1])?;
                    if c.values()
                        .any(|&v| f64::from(v) * 1000.0 >= self.inference_timeout_ms as f64)
                    {
                        return Err(format!("inference timeout too short for {model}/{profile}"));
                    }
                }
            }
        }
        Ok(data)
    }
    fn csv_identity(&self) -> Result<u64, String> {
        // Deterministic content identity for accidental mismatch detection; the runner
        // separately records a cryptographic SHA-256 in its run manifest.
        Ok(std::fs::read(&self.latency_csv)
            .map_err(|e| format!("{}: {e}", self.latency_csv.display()))?
            .iter()
            .fold(0xcbf29ce484222325_u64, |hash, b| {
                (hash ^ u64::from(*b)).wrapping_mul(0x100000001b3)
            }))
    }
    /// Identity checked at scheduler connection; excludes unrelated nodes.
    pub fn node_identity(&self, node: &NodeConfig) -> Result<String, String> {
        let aliases: BTreeMap<_, _> = self.profile_aliases.iter().collect();
        serde_json::to_string(&(
            node,
            aliases,
            self.create_delay_ms,
            self.destroy_delay_ms,
            self.model_costs(node)?,
            self.execution_noise,
            &self.models,
            self.batch_size_range,
            self.csv_identity()?,
        ))
        .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn simulation_validates_coverage_and_lifecycle_budgets() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../resource/simulation/heterogeneous.json");
        let mut config = SimulationConfig::load(&path).unwrap();
        let original = config.node_identity(&config.nodes[0]).unwrap();
        config.execution_noise = worker::mock::ExecutionNoise { cv: 0.1, seed: 7 };
        config.validate().unwrap();
        assert_ne!(original, config.node_identity(&config.nodes[0]).unwrap());
        config.execution_noise.cv = f64::NAN;
        assert!(config.validate().is_err());
        config.execution_noise.cv = 0.1;
        assert!(config.profiles().is_ok());
        config.profile_aliases.clear();
        assert!(config.profiles().is_err());
        config.control_timeout_ms = config.create_delay_ms;
        assert!(config.validate().is_err());
        config.control_timeout_ms = 300_000;
        let mut combined_cost = config.clone();
        combined_cost.destroy_delay_ms = 200_000;
        combined_cost.model_lifecycle.unload_ms = 100_000;
        assert!(
            combined_cost.validate().is_err(),
            "destroy budget must include model unload"
        );
        config.nodes[0].share_targets.push("7g.80gb".into());
        assert!(config.validate().is_err());
    }

    #[test]
    fn model_overrides_are_order_independent_and_part_of_identity() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../resource/simulation/paper-a100.json");
        let mut config = SimulationConfig::load(&path).unwrap();
        let original = config.node_identity(&config.nodes[0]).unwrap();
        config.model_lifecycle.overrides = serde_json::from_str(
            r#"[
            {"device":"nvidia","load_ms":2000},
            {"profile":"1g.10gb","model":"vgg16","load_ms":3000}
        ]"#,
        )
        .unwrap();
        config.validate().unwrap();
        assert_eq!(
            config
                .model_lifecycle
                .resolve("nvidia", "1g.10gb", "vgg16")
                .unwrap()
                .load_ms,
            3000
        );
        let identity = config.node_identity(&config.nodes[0]).unwrap();
        assert_ne!(identity, original);
        config.model_lifecycle.overrides.reverse();
        assert_eq!(identity, config.node_identity(&config.nodes[0]).unwrap());
        config
            .model_lifecycle
            .overrides
            .push(config.model_lifecycle.overrides[0].clone());
        assert!(config.validate().is_err());
    }
}
