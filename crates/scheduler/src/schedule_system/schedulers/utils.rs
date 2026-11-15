//! Shared scheduling logic for both homogeneous and heterogeneous schedulers.

use std::ops::RangeInclusive;

use crate::schedule_system::allocator::RankableShareTarget;

/// Minimize whole-batch execution latency plus batch-formation waiting cost.
///
/// - request_rate: Remaining sample demand used by allocation scoring (samples/s).
/// - latency_predictor: Inference latency in seconds for a given batch size.
/// - range: batch size search range.
pub(crate) fn calculate_inference_latency(
    request_rate: f32,
    latency_predictor: impl Fn(u32) -> f32,
    range: RangeInclusive<u32>,
) -> f32 {
    assert!(
        request_rate.is_finite() && request_rate > 0.0,
        "waiting-time scoring requires a finite positive arrival rate"
    );
    if range.is_empty() {
        panic!("Range should not be empty");
    }
    range
        .into_iter()
        .map(|batch_size| {
            let latency = latency_predictor(batch_size);
            let wait_time = (batch_size - 1) as f32 / (2.0 * request_rate);
            latency + wait_time
        })
        .reduce(f32::min)
        .expect("best latency should exist")
}

/// Capacity is the maximum sample service rate (samples/s), independent of demand.
pub(crate) fn saturated_capacity<T: RankableShareTarget>(
    target: &T,
    model: &str,
    range: RangeInclusive<u32>,
) -> f32 {
    range
        .map(|batch| {
            let latency = target.evaluate_latency(model, batch);
            assert!(
                batch > 0 && latency.is_finite() && latency > 0.0,
                "capacity estimation requires positive finite inference latency"
            );
            batch as f32 / latency
        })
        .reduce(f32::max)
        .expect("batch range must not be empty")
}

/// Spend spare resources on capacity per resource unit, without a fictitious
/// arrival process when demand is zero or already covered.
pub(crate) fn search_best_capacity<T: RankableShareTarget + Clone>(
    targets: &[T],
    model: &str,
    range: RangeInclusive<u32>,
) -> T {
    targets
        .iter()
        .max_by(|a, b| {
            let efficiency = |t: &T| {
                let usage = t.resource_usage_rate();
                assert!(usage.is_finite() && usage > 0.0);
                saturated_capacity(t, model, range.clone()) / usage
            };
            efficiency(a)
                .total_cmp(&efficiency(b))
                .then_with(|| b.resource_usage_rate().total_cmp(&a.resource_usage_rate()))
        })
        .expect("available targets must not be empty")
        .clone()
}

pub(crate) fn search_best_partition<Target: RankableShareTarget + Clone>(
    available_targets: &[Target],
    model_id: &str,
    request_rate: f32,
    range: RangeInclusive<u32>,
) -> (Target, f32) {
    available_targets
        .iter()
        .map(|target| {
            let latency = calculate_inference_latency(
                request_rate,
                |batch_size| target.evaluate_latency(model_id, batch_size),
                range.clone(),
            );
            // Scale remaining demand by this slice's share of its parent resource.
            let equivalent_latency = calculate_inference_latency(
                request_rate * target.resource_usage_rate(),
                |batch_size| target.evaluate_latency(model_id, batch_size),
                range.clone(),
            );

            (target.clone(), latency, equivalent_latency)
        })
        .min_by(
            |(_, _, equivalent_latency_a), (_, _, equivalent_latency_b)| {
                equivalent_latency_a
                    .partial_cmp(equivalent_latency_b)
                    .expect("All of latencies should not be NaN")
            },
        )
        .map(|(target, latency, _)| (target, latency))
        .expect("As all targets's latency should not be NaN, there should be at least one target")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoring_uses_whole_batch_latency_without_amortizing_execution() {
        let score = calculate_inference_latency(
            1000.,
            |batch| match batch {
                1 => 0.01,
                8 => 0.04,
                _ => 0.2,
            },
            1..=8,
        );
        assert!((score - 0.01).abs() < 1e-6);
    }
}
