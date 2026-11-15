//! 预测器模块，包含延迟预测和请求速率预测

use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, RwLock},
    time::{Duration, SystemTime},
};

use tokio::fs;

use crate::daemon::types::ModelRequestRatePredictor;

/// Calibrate one pool's profiles from actual inference duration at the executed batch.
pub(crate) fn with_online_correction<T>(
    pool_id: uuid::Uuid,
    base: crate::schedule_system::virtual_compute_node::system::InferenceLatencyPredictor<T>,
) -> (
    crate::schedule_system::virtual_compute_node::system::InferenceLatencyPredictor<T>,
    compute_node::compute_node::r#virtual::virtual_compute_node::components::worker_registry::PerformanceObserver<T>,
)
where
    T: compute_node::compute_device::vgpu::share_target::ShareTarget + Send + Sync + 'static,
{
    let factors = Arc::new(dashmap::DashMap::<(String, String), f32>::new());
    let observed_factors = factors.clone();
    let observed_base = base.clone();
    let observer = Arc::new(
        move |model: &str, resource: &T, batch: usize, seconds: f32| {
            let Ok(batch) = u32::try_from(batch) else {
                return;
            };
            if batch == 0 || !seconds.is_finite() || seconds <= 0.0 {
                return;
            }
            let reference = observed_base(model, resource, batch);
            let ratio = seconds / reference;
            if !reference.is_finite() || reference <= 0.0 || !ratio.is_finite() || ratio <= 0.0 {
                return;
            }
            // ponytail: one factor per model/slice; use per-batch calibration if bias varies by batch.
            let mut factor = observed_factors
                .entry((model.to_owned(), resource.name().to_owned()))
                .or_insert(1.0);
            let previous = *factor;
            *factor = 0.3 * ratio + 0.7 * previous;
            tracing::info!(target: "allocation", event = "online_prediction_feedback",
                pool_id = %pool_id, model, profile = resource.name(), batch,
                base_seconds = reference, observed_processing_seconds = seconds,
                prediction_before_seconds = reference * previous,
                correction_before = previous, correction_after = *factor);
        },
    );
    let predictor = Arc::new(move |model: &str, resource: &T, batch: u32| {
        let reference = base(model, resource, batch);
        let factor = factors
            .get(&(model.to_owned(), resource.name().to_owned()))
            .map_or(1.0, |factor| *factor);
        let corrected = reference * factor;
        if corrected.is_finite() && corrected > 0.0 {
            corrected
        } else {
            reference
        }
    });
    (predictor, observer)
}

impl Default for ModelRequestRatePredictor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn online_correction_converges_and_keeps_pools_models_and_slices_separate() {
        use compute_node::compute_device::vgpu::shareable::test::TestShareTarget;
        let base: crate::schedule_system::virtual_compute_node::system::InferenceLatencyPredictor<
            TestShareTarget,
        > = Arc::new(|_, _, batch| 0.12 * batch as f32);
        let (predict, observe) = with_online_correction(uuid::Uuid::nil(), base.clone());
        let (other_pool, _) = with_online_correction(uuid::Uuid::nil(), base);
        let full = TestShareTarget::G7Gb80;
        let restricted = TestShareTarget::G3Gb40;
        assert_eq!(predict("m", &full, 8), 0.96);
        for _ in 0..10 {
            observe("m", &full, 8, 0.8);
        }
        assert!((predict("m", &full, 8) - 0.8).abs() < 0.005);
        assert!((predict("m", &full, 16) - 1.6).abs() < 0.01);
        assert_eq!(predict("other", &full, 8), 0.96);
        assert_eq!(predict("m", &restricted, 8), 0.96);
        assert_eq!(other_pool("m", &full, 8), 0.96);
        let before = predict("m", &full, 8);
        for (batch, seconds) in [
            (0, 0.8),
            (8, 0.0),
            (8, -1.0),
            (8, f32::NAN),
            (8, f32::INFINITY),
        ] {
            observe("m", &full, batch, seconds);
        }
        assert_eq!(predict("m", &full, 8), before);
    }
    #[cfg(feature = "mock")]
    #[test]
    fn prediction_csvs_match_previous_predictions_without_changing_workers() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let simulation = compute_node::simulation::SimulationConfig::load(
            &root.join("resource/simulation/a100-multi-slice.json"),
        )
        .unwrap();
        let truth = simulation.profiles().unwrap();
        for scale in [0.6_f32, 0.8, 1.2, 1.4] {
            let mut prediction = latency_profile::load_csv(&root.join(format!(
                "resource/simulation/experiments/e1/latency/latency-{scale:.1}.csv"
            )))
            .unwrap();
            latency_profile::apply_aliases(&mut prediction, &simulation.profile_aliases);
            validate_latency_coverage(&truth, &prediction).unwrap();
            for (model, profiles) in &truth {
                for (profile, curve) in profiles {
                    let old_scaled = curve
                        .iter()
                        .map(|(&batch, &latency)| (batch, latency * scale))
                        .collect();
                    for batch in simulation.batch_size_range[0]..=simulation.batch_size_range[1] {
                        assert_eq!(
                            latency_profile::interpolate(&prediction[model][profile], batch),
                            latency_profile::interpolate(&old_scaled, batch)
                        );
                    }
                }
            }
        }
        assert_eq!(simulation.profiles().unwrap(), truth);
        let mut incomplete = truth.clone();
        incomplete.remove(truth.keys().next().unwrap());
        assert!(validate_latency_coverage(&truth, &incomplete).is_err());
    }
    #[tokio::test]
    async fn windows_keep_samples_and_incrementally_remove_old_arrivals() {
        let predictor = ModelRequestRatePredictor::new();
        let now = SystemTime::now();
        predictor.arrivals.write().unwrap().insert(
            "m".into(),
            VecDeque::from([
                (now - Duration::from_secs(301), 99),
                (now - Duration::from_secs(10), 7),
                (now, 3),
            ]),
        );
        predictor.record_arrival("m", 2);
        assert_eq!(predictor.arrivals.read().unwrap()["m"].len(), 3);
        assert_eq!(predictor.get_model_traffic_volume("m", 5), 5);
        assert_eq!(predictor.get_model_traffic_volume("m", 300), 12);
        assert_eq!(predictor.predict().await["m"], 1.);
    }
    #[test]
    fn concurrent_arrivals_remain_ordered_for_reverse_window_queries() {
        let predictor = ModelRequestRatePredictor::new();
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let predictor = &predictor;
                scope.spawn(move || {
                    for _ in 0..1000 {
                        predictor.record_arrival("m", 1);
                    }
                });
            }
        });
        let arrivals = predictor.arrivals.read().unwrap();
        assert_eq!(arrivals["m"].len(), 8000);
        assert!(
            arrivals["m"]
                .iter()
                .zip(arrivals["m"].iter().skip(1))
                .all(|(a, b)| a.0 <= b.0)
        );
    }
}

impl ModelRequestRatePredictor {
    /// 创建新的模型请求速率预测器
    pub fn new() -> Self {
        Self {
            arrivals: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// 使用初始速率创建预测器
    pub fn with_initial_rates(initial_rates: &HashMap<String, f32>) -> Self {
        let mut arrivals = HashMap::new();
        let now = SystemTime::now();

        for (model_name, rate) in initial_rates {
            // 假设每个模型在过去5秒内均匀到达
            let batch_size = (rate * 5.0).ceil() as usize;
            arrivals.insert(model_name.clone(), VecDeque::from([(now, batch_size)]));
        }

        Self {
            arrivals: Arc::new(RwLock::new(arrivals)),
        }
    }

    /// 记录任务到达
    pub fn record_arrival(&self, model_name: &str, batch_size: usize) {
        const FIVE_MINUTES_DURATION: Duration = Duration::from_secs(300); // 5分钟

        let mut arrivals = self.arrivals.write().unwrap();
        let now = SystemTime::now();
        let model_arrivals = arrivals.entry(model_name.to_string()).or_default();

        // Keep timestamps ordered even if the wall clock moves backwards.
        // Sampling while holding the lock also orders concurrent arrivals.
        let timestamp = model_arrivals
            .back()
            .map_or(now, |(last, _)| now.max(*last));
        model_arrivals.push_back((timestamp, batch_size));

        // 保留最近5分钟的记录
        let cutoff = now - FIVE_MINUTES_DURATION;
        while model_arrivals
            .front()
            .is_some_and(|(time, _)| *time <= cutoff)
        {
            model_arrivals.pop_front();
        }
    }

    /// 预测模型请求速率
    pub async fn predict(&self) -> HashMap<String, f32> {
        const PREDICTION_WINDOW_SECS: u64 = 5; // 预测窗口为5秒
        let now = SystemTime::now();
        let cutoff = now - Duration::from_secs(PREDICTION_WINDOW_SECS);
        let arrivals = self.arrivals.read().unwrap();

        arrivals
            .iter()
            .map(|(model, times)| {
                let recent_total_batch: usize = times
                    .iter()
                    .rev()
                    .take_while(|(time, _)| *time > cutoff)
                    .map(|(_, batch_size)| *batch_size)
                    .sum();
                let rate = recent_total_batch as f32 / PREDICTION_WINDOW_SECS as f32; // items per second
                (model.clone(), rate)
            })
            .collect()
    }

    /// 获取特定模型的历史流量
    pub fn get_model_traffic_volume(&self, model_name: &str, window_secs: u64) -> usize {
        let now = SystemTime::now();
        let cutoff = now - Duration::from_secs(window_secs);
        let arrivals = self.arrivals.read().unwrap();

        arrivals
            .get(model_name)
            .map(|times| {
                times
                    .iter()
                    .rev()
                    .take_while(|(time, _)| *time > cutoff)
                    .map(|(_, batch_size)| *batch_size)
                    .sum()
            })
            .unwrap_or(0)
    }

    /// 清理过期的记录
    pub fn cleanup_old_records(&self, max_age_secs: u64) {
        let now = SystemTime::now();
        let cutoff = now - Duration::from_secs(max_age_secs);
        let mut arrivals = self.arrivals.write().unwrap();

        for model_arrivals in arrivals.values_mut() {
            while model_arrivals
                .front()
                .is_some_and(|(time, _)| *time <= cutoff)
            {
                model_arrivals.pop_front();
            }
        }

        // 移除空的模型记录
        arrivals.retain(|_, times| !times.is_empty());
    }

    /// 获取所有已记录的模型列表
    pub fn get_tracked_models(&self) -> Vec<String> {
        let arrivals = self.arrivals.read().unwrap();
        arrivals.keys().cloned().collect()
    }
}

/// Reject a replacement table that cannot predict the reference operating range.
pub(crate) fn validate_latency_coverage(
    reference: &crate::daemon::types::LatencyData,
    predictions: &crate::daemon::types::LatencyData,
) -> Result<(), String> {
    for (model, profiles) in reference {
        for (profile, curve) in profiles {
            let predicted = latency_profile::curve(predictions, model, profile)?;
            for &batch in curve.keys() {
                latency_profile::interpolate(predicted, batch)?;
            }
        }
    }
    Ok(())
}

/// 读取延迟预测CSV
pub async fn load_latency_csv(
    path: &std::path::Path,
) -> Result<crate::daemon::types::LatencyData, Box<dyn std::error::Error>> {
    Ok(latency_profile::parse_csv(
        &fs::read_to_string(path).await?,
    )?)
}
