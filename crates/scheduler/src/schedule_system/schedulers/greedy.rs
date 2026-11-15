//! A Greedy Scheduler for the Virtual Compute Node.

use std::{collections::HashMap, ops::RangeInclusive};

#[cfg(feature = "tracing")]
use tracing::{error, trace};

use crate::schedule_system::{
    allocator::{RankableShareTarget, ResourceAllocator},
    schedulers::utils::{saturated_capacity, search_best_capacity, search_best_partition},
};

/// A greedy scheduler that schedules tasks based on the greedy algorithm.
#[derive(Default)]
pub struct GreedyScheduler;

impl GreedyScheduler {
    /// Schedule the best resource partition for the given target using a greedy approach.
    ///
    /// allocator: The ResourceAllocator which contains system total resources.
    /// model_request_rate: Model names mapped to input sample arrival rates (samples/s).
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(allocator))
    )]
    pub fn schedule<A: ResourceAllocator>(
        allocator: A,
        model_request_rate: HashMap<String, f32>,
        batch_size_range: RangeInclusive<u32>,
    ) -> HashMap<String, Vec<A::Target>> {
        Self::try_schedule(allocator, model_request_rate, batch_size_range)
            .expect("Invalid or infeasible allocation")
    }

    /// Return infeasible demand as an error so a daemon can keep its current plan.
    pub fn try_schedule<A: ResourceAllocator>(
        mut allocator: A,
        model_request_rate: HashMap<String, f32>,
        batch_size_range: RangeInclusive<u32>,
    ) -> Result<HashMap<String, Vec<A::Target>>, String> {
        let mut result = HashMap::with_capacity(model_request_rate.len());

        if !model_request_rate
            .values()
            .all(|v| v.is_finite() && *v >= 0.0)
            || batch_size_range.is_empty()
            || *batch_size_range.start() == 0
        {
            return Err(
                "Arrival rates must be finite and nonnegative; batch range must be positive".into(),
            );
        }
        let mut models = model_request_rate.keys().cloned().collect::<Vec<_>>();
        models.sort();
        let mut remaining = model_request_rate.clone();
        let mut min_targets = allocator.min_targets();
        min_targets.sort_by(|a, b| a.resource_usage_rate().total_cmp(&b.resource_usage_rate()));
        for model_name in &models {
            let min_target = min_targets
                .iter()
                .find(|target| allocator.can_share((*target).clone()))
                .ok_or_else(|| format!("Insufficient resources for model {model_name}"))?;
            allocator.share(min_target.clone()).ok_or_else(|| {
                format!("Allocator rejected minimum target for model {model_name}")
            })?;
            // Minimum slices ensure service; demand is reduced only by incremental allocations.
            result.insert(model_name.clone(), vec![min_target.clone()]);
        }
        if models.is_empty() {
            return Ok(result);
        }

        while let available_targets = allocator.choosable_share_targets()
            && !available_targets.is_empty()
        {
            #[cfg(feature = "tracing")]
            trace!(?available_targets, "Available targets");
            let model = models
                .iter()
                .max_by(|a, b| {
                    remaining[*a]
                        .total_cmp(&remaining[*b])
                        .then_with(|| b.cmp(a))
                })
                .unwrap();
            let target = if remaining[model] > 0.0 {
                search_best_partition(
                    &available_targets,
                    model,
                    remaining[model],
                    batch_size_range.clone(),
                )
                .0
            } else {
                search_best_capacity(&available_targets, model, batch_size_range.clone())
            };
            let capacity = saturated_capacity(&target, model, batch_size_range.clone());
            #[cfg(feature = "tracing")]
            trace!(?model, ?target, ?capacity, "Selected target and capacity");

            if allocator.share(target.clone()).is_none() {
                #[cfg(feature = "tracing")]
                error!("Failed to share target for model: {model}");
                return Err("Allocator rejected an advertised available target".into());
            }

            #[cfg(feature = "tracing")]
            trace!(?allocator, "After share target");

            let reqs = remaining
                .get_mut(model)
                .expect("Model request rate should exist");
            *reqs -= capacity;

            result
                .entry(model.to_string())
                .or_insert_with(Vec::new)
                .push(target);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Clone, Debug)]
    struct Slice;
    impl RankableShareTarget for Slice {
        fn evaluate_latency(&self, model: &str, batch: u32) -> f32 {
            batch as f32 / if model == "fast" { 100. } else { 50. }
        }
        fn resource_usage_rate(&self) -> f32 {
            1. / 8.
        }
    }
    #[derive(Debug)]
    struct Pool(usize);
    impl ResourceAllocator for Pool {
        type Target = Slice;
        fn choosable_share_targets(&self) -> Vec<Slice> {
            if self.0 > 0 { vec![Slice] } else { vec![] }
        }
        fn share(&mut self, _: Slice) -> Option<()> {
            if self.0 == 0 {
                None
            } else {
                self.0 -= 1;
                Some(())
            }
        }
        fn refund(&mut self, _: Slice) -> bool {
            self.0 += 1;
            true
        }
        fn min_targets(&self) -> Vec<Slice> {
            vec![Slice]
        }
        fn can_share(&self, _: Slice) -> bool {
            self.0 > 0
        }
    }
    #[test]
    fn minimum_allocations_reserve_resources_without_reducing_demand() {
        let result = GreedyScheduler::schedule(
            Pool(3),
            HashMap::from([("fast".into(), 100.), ("slow".into(), 90.)]),
            1..=1,
        );
        assert_eq!((result["fast"].len(), result["slow"].len()), (2, 1));
    }

    #[test]
    fn incremental_scoring_uses_remaining_demand_after_saturation_deduction() {
        #[derive(Clone, Copy, Debug, PartialEq)]
        enum Candidate {
            Minimum,
            Seed,
            SmallBatch,
            LargeBatch,
        }
        impl RankableShareTarget for Candidate {
            fn evaluate_latency(&self, _: &str, batch: u32) -> f32 {
                match self {
                    Self::Minimum => 0.001,
                    Self::Seed if batch == 1 => 0.02,
                    Self::Seed => batch as f32 / 80.,
                    Self::SmallBatch => 0.04,
                    Self::LargeBatch if batch == 4 => 0.005,
                    Self::LargeBatch => 0.2,
                }
            }
            fn resource_usage_rate(&self) -> f32 {
                1.
            }
        }
        #[derive(Debug)]
        struct StagedPool(usize);
        impl ResourceAllocator for StagedPool {
            type Target = Candidate;
            fn choosable_share_targets(&self) -> Vec<Candidate> {
                match self.0 {
                    1 => vec![Candidate::Seed],
                    2 => vec![Candidate::SmallBatch, Candidate::LargeBatch],
                    _ => vec![],
                }
            }
            fn share(&mut self, _: Candidate) -> Option<()> {
                if self.0 == 3 {
                    return None;
                }
                self.0 += 1;
                Some(())
            }
            fn refund(&mut self, _: Candidate) -> bool {
                false
            }
            fn min_targets(&self) -> Vec<Candidate> {
                vec![Candidate::Minimum]
            }
            fn can_share(&self, _: Candidate) -> bool {
                self.0 < 3
            }
        }
        // Seed scores best at batch 1 (capacity 50), but saturation deducts 80.
        // Remaining demand 20 selects SmallBatch; original demand 100 selects LargeBatch.
        let result = GreedyScheduler::schedule(
            StagedPool(0),
            HashMap::from([("model".into(), 100.)]),
            1..=4,
        );
        assert_eq!(
            result["model"],
            vec![Candidate::Minimum, Candidate::Seed, Candidate::SmallBatch]
        );
    }

    #[test]
    fn zero_demand_balances_capacity_instead_of_instance_counts() {
        let result = GreedyScheduler::schedule(
            Pool(8),
            HashMap::from([("fast".into(), 0.), ("slow".into(), 0.)]),
            1..=32,
        );
        assert_eq!((result["fast"].len(), result["slow"].len()), (3, 5));
    }
    #[test]
    fn signed_surplus_keeps_allocating_and_updates_the_next_model() {
        let result = GreedyScheduler::schedule(
            Pool(8),
            HashMap::from([("fast".into(), 350.), ("slow".into(), 50.)]),
            1..=32,
        );
        assert_eq!(result.values().map(Vec::len).sum::<usize>(), 8);
        let fast_residual = 350. - result["fast"].len() as f32 * 100.;
        let slow_residual = 50. - result["slow"].len() as f32 * 50.;
        assert!((fast_residual - slow_residual).abs() <= 100.);
        assert!(fast_residual < 0. && slow_residual < 0.);
    }
    #[test]
    fn no_models_does_not_consume_resources() {
        assert!(GreedyScheduler::schedule(Pool(8), HashMap::new(), 1..=32).is_empty());
    }
    #[test]
    #[should_panic(expected = "finite and nonnegative")]
    fn rejects_invalid_actual_demand() {
        GreedyScheduler::schedule(Pool(8), HashMap::from([("fast".into(), -1.)]), 1..=32);
    }
    #[test]
    fn infeasible_allocations_return_error_without_panicking() {
        assert!(
            GreedyScheduler::try_schedule(
                Pool(1),
                HashMap::from([("fast".into(), 1.), ("slow".into(), 1.)]),
                1..=32
            )
            .is_err()
        );
    }
}
