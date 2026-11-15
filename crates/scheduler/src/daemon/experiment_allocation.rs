//! Simulation-only static spatial plans. Application/hold belongs to the daemon loop.

use std::collections::{BTreeMap, HashMap, HashSet};

use compute_node::compute_device::vgpu::{
    resource_spec::nv::mig::{NvidiaMigResource, VDeviceScale},
    share_target::ShareTarget,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    Scheduler,
    daemon::{
        config::validate_config,
        config_builder::template_pool_ids,
        types::{Config, DeviceConfig, LatencyData},
    },
    schedule_system::{
        allocator::ResourceAllocator,
        virtual_compute_node::system::heterogeneous::{
            allocator::BidAllocator, bid::Bid, system::HeterogeneousSystemContext,
        },
    },
};

pub(crate) type Allocation = HashMap<Uuid, HashMap<String, Vec<Bid>>>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AllocationPlan {
    pub schema_version: u32,
    pub placements: Vec<AllocationPlacement>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AllocationPlacement {
    pub template_name: String,
    pub model: String,
    pub slice: String,
    pub count: usize,
}

#[derive(Debug, Clone)]
struct ResolvedPlacement {
    pool_id: Uuid,
    placement: AllocationPlacement,
    scale: VDeviceScale,
}

#[derive(Debug, Clone)]
pub(crate) struct FixedAllocationScheduler {
    placements: Vec<ResolvedPlacement>,
}

impl FixedAllocationScheduler {
    /// Validate portable names/profiles now; actual aggregate budgets are checked
    /// with fresh runtime allocators before any plan can be applied.
    pub(crate) fn from_plan(
        mut plan: AllocationPlan,
        config: &Config,
        latencies: &LatencyData,
    ) -> Result<Self, String> {
        validate_config(config)?;
        if plan.schema_version != 1 {
            return Err(format!(
                "unsupported allocation schema {}",
                plan.schema_version
            ));
        }
        if config.init_arrival_rate.is_empty() {
            return Err("fixed allocation requires configured models".into());
        }
        let pool_ids = template_pool_ids(config);
        for template in pool_ids.keys() {
            let Some(DeviceConfig::MockNvidia {
                share_targets,
                resource_template,
            }) = config.device_combinations.get(template)
            else {
                return Err(format!(
                    "fixed allocation requires MockNvidia pool {template}"
                ));
            };
            if resource_template.ce == 0
                || resource_template.memory == 0
                || share_targets.is_empty()
            {
                return Err(format!(
                    "pool {template} has no usable resource/slice whitelist"
                ));
            }
            for slice in share_targets {
                let scale = VDeviceScale::from_name(slice)
                    .ok_or_else(|| format!("unknown slice {template}/{slice}"))?;
                let cost = NvidiaMigResource::from(scale);
                if cost.ce > resource_template.ce || cost.memory > resource_template.memory {
                    return Err(format!(
                        "slice {template}/{slice} exceeds its parent CE/GB budget"
                    ));
                }
                for model in config.init_arrival_rate.keys() {
                    let curve = latency_profile::curve(latencies, model, slice)?;
                    // Endpoint checks fail quickly for an out-of-profile configured range.
                    latency_profile::interpolate(curve, config.batch_size_range.0)?;
                    latency_profile::interpolate(curve, config.batch_size_range.1)?;
                    for batch in
                        (config.batch_size_range.0..=config.batch_size_range.1).chain([1, 4, 8, 16])
                    {
                        let latency = latency_profile::interpolate(curve, batch)?;
                        if !latency.is_finite() || latency <= 0.0 {
                            return Err(format!("invalid latency {model}/{slice}/batch{batch}"));
                        }
                    }
                }
            }
        }
        plan.placements.sort_by(|a, b| {
            (&a.template_name, &a.model, &a.slice).cmp(&(&b.template_name, &b.model, &b.slice))
        });
        let mut seen = HashSet::new();
        let mut covered_models = HashSet::new();
        let mut placements = Vec::with_capacity(plan.placements.len());
        for placement in plan.placements {
            if placement.count == 0 {
                return Err("allocation counts must be positive integers".into());
            }
            if !seen.insert((
                placement.template_name.clone(),
                placement.model.clone(),
                placement.slice.clone(),
            )) {
                return Err("duplicate template/model/slice placement".into());
            }
            if !config.init_arrival_rate.contains_key(&placement.model) {
                return Err(format!("unknown allocation model {}", placement.model));
            }
            let pool_id = *pool_ids
                .get(&placement.template_name)
                .ok_or_else(|| format!("unknown allocation pool {}", placement.template_name))?;
            let DeviceConfig::MockNvidia { share_targets, .. } =
                &config.device_combinations[&placement.template_name]
            else {
                unreachable!("active pool validated above")
            };
            if !share_targets.contains(&placement.slice) {
                return Err(format!(
                    "slice {} is not allowed in pool {}",
                    placement.slice, placement.template_name
                ));
            }
            let scale = VDeviceScale::from_name(&placement.slice)
                .ok_or_else(|| format!("unknown slice {}", placement.slice))?;
            covered_models.insert(placement.model.clone());
            placements.push(ResolvedPlacement {
                pool_id,
                placement,
                scale,
            });
        }
        if config
            .init_arrival_rate
            .keys()
            .any(|model| !covered_models.contains(model))
        {
            return Err("allocation must serve every configured model".into());
        }
        // Match deployment's resource-first ordering; small slices first can
        // fragment a multi-parent pool even when the same counts fit.
        placements.sort_by(|a, b| {
            a.pool_id
                .cmp(&b.pool_id)
                .then_with(|| {
                    NvidiaMigResource::from(b.scale).cmp(&NvidiaMigResource::from(a.scale))
                })
                .then_with(|| {
                    (&a.placement.model, &a.placement.slice)
                        .cmp(&(&b.placement.model, &b.placement.slice))
                })
        });
        Ok(Self { placements })
    }

    fn materialize(&self, mut allocator: BidAllocator) -> Result<Allocation, String> {
        let mut allocation = Allocation::new();
        for item in &self.placements {
            for _ in 0..item.placement.count {
                let bid = allocator.choosable_share_targets().into_iter().find(|bid| {
                    bid.device_id() == item.pool_id
                        && bid.data.downcast_ref::<VDeviceScale>() == Some(&item.scale)
                }).ok_or_else(|| format!(
                    "allocation infeasible in runtime allocator: {}/{}/{} (CE/GB or packing)",
                    item.placement.template_name, item.placement.model, item.placement.slice
                ))?;
                if allocator.share(bid.clone()).is_none() {
                    return Err("runtime allocator rejected advertised allocation bid".into());
                }
                allocation
                    .entry(item.pool_id)
                    .or_default()
                    .entry(item.placement.model.clone())
                    .or_default()
                    .push(bid);
            }
        }
        Ok(allocation)
    }
}

impl Scheduler<HeterogeneousSystemContext> for FixedAllocationScheduler {
    type Error = String;

    async fn schedule(
        &self,
        context: impl AsRef<HeterogeneousSystemContext>,
    ) -> Result<Allocation, String> {
        self.materialize(context.as_ref().fresh_allocator())
    }
}

/// Canonical portable snapshot; preserves each actual Bid's typed slice identity.
pub(crate) fn export_allocation(
    state: &Allocation,
    pool_names: &HashMap<Uuid, String>,
) -> Result<AllocationPlan, String> {
    if pool_names.values().any(|name| name.is_empty())
        || pool_names.values().collect::<HashSet<_>>().len() != pool_names.len()
    {
        return Err("pool-name map must contain unique nonempty template names".into());
    }
    let mut counts = BTreeMap::<(String, String, String), usize>::new();
    for (pool_id, models) in state {
        let template = pool_names
            .get(pool_id)
            .ok_or_else(|| format!("cannot export unknown pool {pool_id}"))?;
        for (model, bids) in models {
            if model.is_empty() {
                return Err("cannot export an empty model name".into());
            }
            for bid in bids {
                if bid.device_id() != *pool_id {
                    return Err("cannot export a bid belonging to another pool".into());
                }
                let scale = bid
                    .data
                    .downcast_ref::<VDeviceScale>()
                    .ok_or("cannot export a non-Nvidia slice payload")?;
                let count = counts
                    .entry((template.clone(), model.clone(), scale.name().into()))
                    .or_default();
                *count = count.checked_add(1).ok_or("allocation count overflow")?;
            }
        }
    }
    if counts.is_empty() {
        return Err("cannot export an empty allocation".into());
    }
    Ok(AllocationPlan {
        schema_version: 1,
        placements: counts
            .into_iter()
            .map(
                |((template_name, model, slice), count)| AllocationPlacement {
                    template_name,
                    model,
                    slice,
                    count,
                },
            )
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use compute_node::compute_node::r#virtual::virtual_compute_node::components::request_scheduler::batch_size_extractor::BatchExtractorStrategy;

    use super::*;
    use crate::{
        daemon::types::RemoteNodeConfig,
        schedule_system::{
            allocator::RankableShareTarget,
            virtual_compute_node::{
                algorithms::buddy_cell::{BuddyCellAllocator, constrained::ConstrainedResource},
                system::heterogeneous::device::ShareableResourceAllocator,
            },
        },
    };

    fn inputs() -> (Config, LatencyData, AllocationPlan) {
        let config = Config {
            batch_extractor_strategy: BatchExtractorStrategy::split_and_satisfy(),
            device_combinations: HashMap::from([
                (
                    "full".into(),
                    DeviceConfig::MockNvidia {
                        share_targets: [
                            "1g.10gb", "1g.20gb", "2g.20gb", "3g.40gb", "4g.40gb", "7g.80gb",
                        ]
                        .into_iter()
                        .map(str::to_owned)
                        .collect(),
                        resource_template: NvidiaMigResource { ce: 7, memory: 80 },
                    },
                ),
                (
                    "half".into(),
                    DeviceConfig::MockNvidia {
                        share_targets: ["1g.10gb", "2g.20gb", "3g.40gb"]
                            .into_iter()
                            .map(str::to_owned)
                            .collect(),
                        resource_template: NvidiaMigResource { ce: 3, memory: 40 },
                    },
                ),
            ]),
            remote_nodes: ["half", "full"]
                .into_iter()
                .enumerate()
                .map(|(i, template)| {
                    (
                        template.to_owned(),
                        RemoteNodeConfig {
                            ip: "127.0.0.1".into(),
                            port: 5401 + i as u16,
                            template_name: template.into(),
                            control_timeout_ms: 300_000,
                            inference_timeout_ms: 300_000,
                            expected_simulation_identity: None,
                        },
                    )
                })
                .collect(),
            batch_size_range: (1, 32),
            init_arrival_rate: HashMap::from([("m1".into(), 100.0), ("m2".into(), 100.0)]),
        };
        let latencies = config
            .init_arrival_rate
            .keys()
            .map(|model| {
                let profiles = [
                    "1g.10gb", "1g.20gb", "2g.20gb", "3g.40gb", "4g.40gb", "7g.80gb",
                ]
                .into_iter()
                .map(|slice| {
                    (
                        slice.into(),
                        [1, 4, 8, 16, 32]
                            .into_iter()
                            .map(|batch| (batch, batch as f32 * 0.01))
                            .collect(),
                    )
                })
                .collect();
                (model.clone(), profiles)
            })
            .collect();
        let plan = AllocationPlan {
            schema_version: 1,
            placements: vec![
                AllocationPlacement {
                    template_name: "half".into(),
                    model: "m1".into(),
                    slice: "1g.10gb".into(),
                    count: 1,
                },
                AllocationPlacement {
                    template_name: "full".into(),
                    model: "m2".into(),
                    slice: "1g.10gb".into(),
                    count: 2,
                },
            ],
        };
        (config, latencies, plan)
    }

    #[test]
    fn fixed_allocation_rejects_schema_names_counts_and_model_omissions() {
        let (config, latencies, plan) = inputs();
        assert!(FixedAllocationScheduler::from_plan(plan.clone(), &config, &latencies).is_ok());
        let mut bad = plan.clone();
        bad.schema_version = 2;
        assert!(FixedAllocationScheduler::from_plan(bad, &config, &latencies).is_err());
        let mut bad = plan.clone();
        bad.placements.push(bad.placements[0].clone());
        assert!(FixedAllocationScheduler::from_plan(bad, &config, &latencies).is_err());
        let mut bad = plan.clone();
        bad.placements[0].count = 0;
        assert!(FixedAllocationScheduler::from_plan(bad, &config, &latencies).is_err());
        for (field, value) in [
            ("template_name", "missing"),
            ("model", "missing"),
            ("slice", "4g.40gb"),
        ] {
            let mut bad = serde_json::to_value(&plan).unwrap();
            bad["placements"][0][field] = value.into();
            assert!(
                FixedAllocationScheduler::from_plan(
                    serde_json::from_value(bad).unwrap(),
                    &config,
                    &latencies
                )
                .is_err()
            );
        }
        let mut bad = plan.clone();
        bad.placements.pop();
        assert!(FixedAllocationScheduler::from_plan(bad, &config, &latencies).is_err());
        for count in [serde_json::json!(-1), serde_json::json!(1.5)] {
            let mut bad = serde_json::to_value(&plan).unwrap();
            bad["placements"][0]["count"] = count;
            assert!(serde_json::from_value::<AllocationPlan>(bad).is_err());
        }
        let mut bad = serde_json::to_value(&plan).unwrap();
        bad["unexpected"] = true.into();
        assert!(serde_json::from_value::<AllocationPlan>(bad).is_err());
    }

    #[test]
    fn fixed_allocation_rejects_hardware_parent_budgets_and_incomplete_profiles() {
        let (mut config, mut latencies, plan) = inputs();
        let original = config.device_combinations["full"].clone();
        config.device_combinations.insert(
            "full".into(),
            DeviceConfig::Nvidia {
                share_targets: vec!["1g.10gb".into()],
                resource_template: NvidiaMigResource { ce: 7, memory: 80 },
            },
        );
        assert!(FixedAllocationScheduler::from_plan(plan.clone(), &config, &latencies).is_err());
        for resource_template in [
            NvidiaMigResource { ce: 6, memory: 80 },
            NvidiaMigResource { ce: 7, memory: 79 },
        ] {
            config.device_combinations.insert(
                "full".into(),
                DeviceConfig::MockNvidia {
                    share_targets: vec!["1g.10gb".into(), "7g.80gb".into()],
                    resource_template,
                },
            );
            assert!(
                FixedAllocationScheduler::from_plan(plan.clone(), &config, &latencies).is_err()
            );
        }
        config.device_combinations.insert("full".into(), original);
        let original_curve = latencies["m1"]["1g.10gb"].clone();
        // Routing coverage is required even when execution is configured as batch one.
        config.batch_size_range = (1, 1);
        latencies
            .get_mut("m1")
            .unwrap()
            .insert("1g.10gb".into(), BTreeMap::from([(1, 0.01)]));
        assert!(FixedAllocationScheduler::from_plan(plan.clone(), &config, &latencies).is_err());
        config.batch_size_range = (1, 32);
        latencies
            .get_mut("m1")
            .unwrap()
            .insert("1g.10gb".into(), original_curve);
        latencies
            .get_mut("m1")
            .unwrap()
            .get_mut("1g.10gb")
            .unwrap()
            .insert(5, f32::NAN);
        assert!(FixedAllocationScheduler::from_plan(plan.clone(), &config, &latencies).is_err());
        latencies
            .get_mut("m1")
            .unwrap()
            .get_mut("1g.10gb")
            .unwrap()
            .remove(&5);
        latencies
            .get_mut("m2")
            .unwrap()
            .get_mut("2g.20gb")
            .unwrap()
            .remove(&32);
        assert!(FixedAllocationScheduler::from_plan(plan, &config, &latencies).is_err());
    }

    #[derive(Debug)]
    struct BudgetAllocator {
        inner: BuddyCellAllocator<ConstrainedResource<NvidiaMigResource>>,
        bids: Vec<Bid>,
    }
    impl BudgetAllocator {
        fn new(pool_id: Uuid, parents: Vec<NvidiaMigResource>) -> Self {
            let scales = [
                "1g.10gb", "1g.20gb", "2g.20gb", "3g.40gb", "4g.40gb", "7g.80gb",
            ]
            .into_iter()
            .map(|name| VDeviceScale::from_name(name).unwrap())
            .collect::<Vec<_>>();
            let tiny = VDeviceScale::from_name("1g.10gb").unwrap();
            let limit = parents[0];
            let inner = BuddyCellAllocator::new(
                parents
                    .into_iter()
                    .map(|parent| ConstrainedResource::new(parent, parent))
                    .collect(),
                scales.clone(),
                vec![tiny],
                limit,
            );
            let bids = scales
                .into_iter()
                .map(|scale| {
                    Bid::new(
                        pool_id,
                        format!("opaque-test-id-{}", scale.name()),
                        Arc::new(scale),
                        Arc::new(|_, batch| batch as f32 * 0.007),
                        0.125,
                    )
                })
                .collect();
            Self { inner, bids }
        }
    }
    impl ShareableResourceAllocator for BudgetAllocator {
        type Target = Bid;
        fn available_targets(&self) -> Vec<Bid> {
            let available = self.inner.choosable_share_targets();
            self.bids
                .iter()
                .filter(|bid| available.contains(bid.data.downcast_ref::<VDeviceScale>().unwrap()))
                .cloned()
                .collect()
        }
        fn share(&mut self, target: &Bid) -> bool {
            self.inner
                .share(*target.data.downcast_ref::<VDeviceScale>().unwrap())
                .is_some()
        }
        fn refund(&mut self, target: &Bid) -> bool {
            self.inner
                .refund(*target.data.downcast_ref::<VDeviceScale>().unwrap())
        }
        fn min_targets(&self) -> Vec<Bid> {
            vec![self.bids[0].clone()]
        }
    }

    fn budgets(config: &Config) -> BidAllocator {
        let ids = template_pool_ids(config);
        ["full", "half"]
            .into_iter()
            .map(|template| {
                let DeviceConfig::MockNvidia {
                    resource_template, ..
                } = &config.device_combinations[template]
                else {
                    unreachable!()
                };
                Box::new(BudgetAllocator::new(
                    ids[template],
                    vec![*resource_template],
                )) as Box<dyn ShareableResourceAllocator<Target = Bid>>
            })
            .collect()
    }

    /// Offline preflight only: one full parent, optionally with one restricted parent.
    #[tokio::test]
    #[ignore = "requires explicit E2 fixture and candidate file paths"]
    async fn preflight_full_parent_candidates() {
        use std::{fs, io::Write, path::PathBuf};

        let input = PathBuf::from(std::env::var("E2_PREFLIGHT_INPUT").unwrap());
        let config: Config =
            toml::from_str(&fs::read_to_string(input.join("scheduler.toml")).unwrap()).unwrap();
        let ids = template_pool_ids(&config);
        assert_eq!(config.remote_nodes.len(), ids.len());
        assert!(ids.len() == 1 || ids.len() == 2);
        assert!(ids.contains_key("full"));
        let mut resources = Vec::new();
        for (name, id) in ids {
            let DeviceConfig::MockNvidia {
                resource_template, ..
            } = config.device_combinations[&name]
            else {
                panic!("mock parent required")
            };
            let expected = match name.as_str() {
                "full" => NvidiaMigResource { ce: 7, memory: 80 },
                "half" => NvidiaMigResource { ce: 3, memory: 40 },
                _ => panic!("unexpected E2 parent template"),
            };
            assert_eq!(resource_template, expected);
            resources.push((id, resource_template));
        }
        let csv = PathBuf::from(std::env::var("E2_PREFLIGHT_CSV").unwrap());
        let latencies = crate::daemon::predictor::load_latency_csv(&csv)
            .await
            .unwrap();
        let mut records = Vec::new();
        for line in fs::read_to_string(input.join("nominal-jobs.jsonl"))
            .unwrap()
            .lines()
        {
            let row: serde_json::Value = serde_json::from_str(line).unwrap();
            let plan: AllocationPlan = serde_json::from_value(row["allocation"].clone()).unwrap();
            let scheduler = FixedAllocationScheduler::from_plan(plan, &config, &latencies).unwrap();
            let allocator: BidAllocator = resources
                .iter()
                .map(|(id, resource)| {
                    Box::new(BudgetAllocator::new(*id, vec![*resource]))
                        as Box<dyn ShareableResourceAllocator<Target = Bid>>
                })
                .collect();
            let result = scheduler.materialize(allocator);
            records.push(
                serde_json::json!({"id":row["id"],"accepted":result.is_ok(),"error":result.err()}),
            );
        }
        let mut output = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(input.join("allocator-preflight.json"))
            .unwrap();
        writeln!(
            output,
            "{}",
            serde_json::to_string_pretty(&records).unwrap()
        )
        .unwrap();
        assert!(!records.is_empty());
    }

    #[test]
    fn fixed_allocation_materializes_real_bids_and_exports_canonical_counts() {
        let (config, latencies, plan) = inputs();
        let ids = template_pool_ids(&config);
        let allocator = budgets(&config);
        let original_bid = allocator
            .choosable_share_targets()
            .into_iter()
            .find(|bid| {
                bid.device_id() == ids["full"]
                    && bid.data.downcast_ref::<VDeviceScale>().unwrap().name() == "1g.10gb"
            })
            .unwrap();
        let scheduler =
            FixedAllocationScheduler::from_plan(plan.clone(), &config, &latencies).unwrap();
        let state = scheduler.materialize(allocator).unwrap();
        let selected = &state[&ids["full"]]["m2"][0];
        assert!(Arc::ptr_eq(&original_bid.data, &selected.data));
        assert_eq!(selected.target_id(), original_bid.target_id());
        assert_eq!(selected.evaluate_latency("m2", 16), 16.0 * 0.007);
        assert_eq!(selected.resource_usage_rate(), 0.125);
        let names = ids.into_iter().map(|(name, id)| (id, name)).collect();
        let exported = export_allocation(&state, &names).unwrap();
        let mut expected = plan;
        expected
            .placements
            .sort_by(|a, b| a.template_name.cmp(&b.template_name));
        assert_eq!(exported, expected);
        assert_eq!(
            serde_json::from_str::<AllocationPlan>(&serde_json::to_string(&exported).unwrap())
                .unwrap(),
            exported
        );
        let repeated = scheduler.materialize(budgets(&config)).unwrap();
        assert_eq!(export_allocation(&repeated, &names).unwrap(), exported);
        assert!(FixedAllocationScheduler::from_plan(exported, &config, &latencies).is_ok());
    }

    #[test]
    fn fixed_allocation_uses_real_ce_gb_and_per_parent_packing_limits() {
        let (config, latencies, plan) = inputs();
        for (slice, count) in [("1g.10gb", 8), ("1g.20gb", 5)] {
            let mut bad = plan.clone();
            bad.placements[1].slice = slice.into();
            bad.placements[1].count = count;
            let scheduler = FixedAllocationScheduler::from_plan(bad, &config, &latencies).unwrap();
            assert!(scheduler.materialize(budgets(&config)).is_err());
        }
        // Aggregate (6 CE, 60 GB) fits two (3 CE, 40 GB) parents,
        // but three 2g.20gb replicas cannot be packed into them.
        let mut packing = plan;
        packing.placements[1].slice = "2g.20gb".into();
        packing.placements[1].count = 3;
        let scheduler = FixedAllocationScheduler::from_plan(packing, &config, &latencies).unwrap();
        let ids = template_pool_ids(&config);
        let allocator = [
            Box::new(BudgetAllocator::new(
                ids["full"],
                vec![NvidiaMigResource { ce: 3, memory: 40 }; 2],
            )),
            Box::new(BudgetAllocator::new(
                ids["half"],
                vec![NvidiaMigResource { ce: 3, memory: 40 }],
            )),
        ]
        .into_iter()
        .map(|allocator| allocator as Box<dyn ShareableResourceAllocator<Target = Bid>>)
        .collect();
        assert!(scheduler.materialize(allocator).is_err());
    }

    #[test]
    fn fixed_allocation_shares_mixed_slice_sizes_before_smaller_fragments() {
        let (config, latencies, mut plan) = inputs();
        plan.placements[1].count = 5;
        for (slice, count) in [("1g.20gb", 1), ("4g.40gb", 2)] {
            plan.placements.push(AllocationPlacement {
                template_name: "full".into(),
                model: "m2".into(),
                slice: slice.into(),
                count,
            });
        }
        let scheduler = FixedAllocationScheduler::from_plan(plan, &config, &latencies).unwrap();
        let ids = template_pool_ids(&config);
        assert_eq!(scheduler.placements[0].scale.name(), "4g.40gb");
        let allocator = [
            Box::new(BudgetAllocator::new(
                ids["full"],
                vec![NvidiaMigResource { ce: 7, memory: 80 }; 2],
            )),
            Box::new(BudgetAllocator::new(
                ids["half"],
                vec![NvidiaMigResource { ce: 3, memory: 40 }],
            )),
        ]
        .into_iter()
        .map(|allocator| allocator as Box<dyn ShareableResourceAllocator<Target = Bid>>)
        .collect();
        let state = scheduler.materialize(allocator).unwrap();
        let names = ids.into_iter().map(|(name, id)| (id, name)).collect();
        let exported = export_allocation(&state, &names).unwrap();
        assert_eq!(
            exported
                .placements
                .iter()
                .filter(|p| p.template_name == "full")
                .map(|p| (p.slice.as_str(), p.count))
                .collect::<Vec<_>>(),
            vec![("1g.10gb", 5), ("1g.20gb", 1), ("4g.40gb", 2)]
        );
    }

    #[test]
    fn fixed_allocation_export_rejects_wrong_pool_and_payload() {
        let (config, latencies, plan) = inputs();
        let ids = template_pool_ids(&config);
        let names = ids.iter().map(|(name, id)| (*id, name.clone())).collect();
        let state = FixedAllocationScheduler::from_plan(plan, &config, &latencies)
            .unwrap()
            .materialize(budgets(&config))
            .unwrap();
        assert!(export_allocation(&state, &HashMap::new()).is_err());
        assert!(export_allocation(&Allocation::new(), &names).is_err());
        let mut bad = state.clone();
        bad.get_mut(&ids["full"]).unwrap().get_mut("m2").unwrap()[0] = Bid::new(
            ids["full"],
            "wrong-type".into(),
            Arc::new(()),
            Arc::new(|_, _| 0.1),
            1.0,
        );
        assert!(export_allocation(&bad, &names).is_err());
        let mut bad = state;
        let foreign = bad[&ids["half"]]["m1"][0].clone();
        bad.get_mut(&ids["full"]).unwrap().get_mut("m2").unwrap()[0] = foreign;
        assert!(export_allocation(&bad, &names).is_err());
    }

    #[test]
    fn fixed_allocation_pool_ids_preserve_builder_order_and_group_identity() {
        let (mut config, _, _) = inputs();
        let expected = HashMap::from([
            (
                "full".into(),
                Uuid::from_u128(0x48534753000000000000000000000000),
            ),
            (
                "half".into(),
                Uuid::from_u128(0x48534753000000000000000000000001),
            ),
        ]);
        assert_eq!(template_pool_ids(&config), expected);
        let second = config.remote_nodes["full"].clone();
        config
            .remote_nodes
            .insert("second-full-node".into(), second);
        assert_eq!(template_pool_ids(&config), expected);
    }
}
