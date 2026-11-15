//! Mock implementations for testing purposes.

use std::{
    borrow::Borrow,
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};

use model::{
    model::{InferenceModel, ModelError, ModelResult},
    tensor::DynTensor,
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use utils::ErrorMessage;

use crate::{Worker, WorkerError, WorkerResult};

/// Resolved costs for one model on one device/profile. Zero-cost unit mocks are
/// separate from the explicitly configured simulation defaults.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelLifecycle {
    /// Loading and warmup until the model can accept work.
    pub load_ms: u64,
    /// Unloading after all in-flight inference has completed.
    pub unload_ms: u64,
}

/// Uncalibrated execution-time variation; CV is standard deviation / mean.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionNoise {
    /// Standard deviation divided by mean; zero uses the CSV execution mean.
    pub cv: f64,
    /// Final stream seed supplied by the caller, without worker-side derivation.
    pub seed: u64,
}

impl ExecutionNoise {
    /// Reject invalid or excessively broad experimental variation.
    pub fn validate(&self) -> Result<(), String> {
        if !self.cv.is_finite() || !(0.0..=1.0).contains(&self.cv) {
            return Err("execution noise CV must be finite and in [0, 1]".into());
        }
        Ok(())
    }

    fn sample(&self, mean: f64, rng: &mut StdRng) -> f64 {
        if self.cv == 0.0 {
            return mean;
        }
        // Box-Muller uses the existing rand dependency. The shift preserves E[T].
        let z = (-2.0 * rng.random_range(f64::EPSILON..1.0).ln()).sqrt()
            * (std::f64::consts::TAU * rng.random::<f64>()).cos();
        let variance = (self.cv * self.cv).ln_1p();
        mean * (variance.sqrt() * z - variance / 2.0).exp()
    }
}

/// A mock model for testing purposes.
#[derive(Debug)]
pub struct MockModel {
    #[allow(dead_code)]
    name: String,
    #[allow(dead_code)]
    config: String,
    /// Batch size -> Latency in seconds
    latency_profile: latency_profile::Curve,
    lifecycle: ModelLifecycle,
    noise: ExecutionNoise,
    noise_seed: u64,
    load_generation: usize,
    noise_rng: Mutex<(StdRng, u64)>,
}

impl MockModel {
    /// Parses the latency profile from the given path.
    /// The profile should be in the format:
    /// ```text
    /// model_name,config_name,batch_size,latency_in_seconds
    /// ```
    /// For example:
    /// ```text
    /// resnet50,1g.10gb,1,0.05
    /// resnet50,1g.10gb,8,0.2
    /// bert,1g.10gb,1,0.1
    /// bert,1g.10gb,8,0.5
    /// ```
    /// This indicates that for model `resnet50`, the latency for batch size 1 is 0.05 seconds,
    /// and for batch size 8 is 0.2 seconds. For model `bert`, the latency for in config 1g.10gb
    /// batch size 1 is 0.1 seconds and for batch size 8 is 0.5 seconds.
    /// returns a map from model name to a map of batch size to latency.
    fn from_latency_profile(
        name: String,
        config: String,
        path: PathBuf,
    ) -> Result<Self, ErrorMessage> {
        let data = latency_profile::load_csv(&path).map_err(ErrorMessage::new)?;
        let profile = latency_profile::curve(&data, &name, &config)
            .map_err(ErrorMessage::new)?
            .clone();

        Ok(Self {
            name,
            config,
            latency_profile: profile,
            lifecycle: ModelLifecycle::default(),
            noise: ExecutionNoise::default(),
            noise_seed: 0,
            load_generation: 0,
            noise_rng: Mutex::new((StdRng::seed_from_u64(0), 0)),
        })
    }
}

/// Initialization configuration for [MockModel].
pub struct MockModelInitConfig {
    name: String,
    config: String,
    latency_profile_path: PathBuf,
    lifecycle: ModelLifecycle,
    noise: ExecutionNoise,
    noise_seeds: Option<Vec<u64>>,
}

impl MockModelInitConfig {
    /// Creates a new [MockModelInitConfig] with the given name and latency profile path.
    pub fn new(name: String, config: String, latency_profile_path: PathBuf) -> Self {
        Self {
            name,
            config,
            latency_profile_path,
            lifecycle: ModelLifecycle::default(),
            noise: ExecutionNoise::default(),
            noise_seeds: None,
        }
    }
    /// Apply already validated model lifecycle costs.
    pub fn with_lifecycle(mut self, lifecycle: ModelLifecycle) -> Self {
        self.lifecycle = lifecycle;
        self
    }
    /// Configure variation without changing the CSV mean or model lifecycle.
    pub fn with_execution_noise(mut self, noise: ExecutionNoise) -> Self {
        self.noise = noise;
        self
    }
    /// Final seeds for successive successful loads of this model on a worker.
    /// Duplicate loads reuse the current model and do not consume a seed.
    /// Without a schedule, each genuine reload restarts the supplied noise seed.
    pub fn with_execution_noise_seeds(mut self, seeds: Vec<u64>) -> Self {
        self.noise_seeds = Some(seeds);
        self
    }
}

impl InferenceModel for MockModel {
    type Model = Self;
    type Input = DynTensor;
    type Output = DynTensor;
    type Error = ErrorMessage;
    type LoadConfig = MockModelInitConfig;

    fn load(config: impl Into<Self::LoadConfig>) -> ModelResult<Self, Self::Error> {
        let config = config.into();
        if config.latency_profile_path.exists() {
            let MockModelInitConfig {
                name,
                config,
                latency_profile_path,
                lifecycle,
                noise,
                ..
            } = config;
            noise
                .validate()
                .map_err(|message| ModelError::LoadConfigError { message })?;
            MockModel::from_latency_profile(name, config, latency_profile_path)
                .map(|mut model| {
                    model.lifecycle = lifecycle;
                    model.noise_seed = noise.seed;
                    model.noise = noise;
                    model.noise_rng = Mutex::new((StdRng::seed_from_u64(model.noise_seed), 0));
                    model
                })
                .map_err(|e| ModelError::LoadConfigError {
                    message: format!("Failed to parse latency profile: {}", e),
                })
        } else {
            Err(ModelError::LoadConfigError {
                message: "Latency profile path does not exist".to_string(),
            })
        }
    }

    fn model(&self) -> &Self::Model {
        self
    }

    fn check_input(&self, _input: impl Borrow<Self::Input>) -> ModelResult<(), Self::Error> {
        Ok(())
    }

    async fn forward(
        &self,
        input: impl Into<Self::Input> + Send,
    ) -> ModelResult<Self::Output, Self::Error> {
        let input = input.into();
        let batch = input
            .dimensions()
            .first()
            .copied()
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| ModelError::InvalidInput {
                message: "input needs a valid batch dimension".into(),
            })?;
        let latency = latency_profile::interpolate(&self.latency_profile, batch)
            .map_err(|message| ModelError::InvalidInput { message })?;
        let (sampled_seconds, sequence) = {
            let mut state = self.noise_rng.lock().unwrap();
            let sampled = self.noise.sample(f64::from(latency), &mut state.0);
            state.1 += 1;
            (sampled, state.1)
        };
        let started = std::time::Instant::now();
        tokio::time::sleep(Duration::from_secs_f64(sampled_seconds)).await;
        tracing::info!(target: "simulation", event = "inference", model = %self.name, profile = %self.config, batch, profile_seconds = latency, sampled_seconds, noise_cv = self.noise.cv, noise_seed = self.noise_seed, load_generation = self.load_generation, sequence, elapsed_seconds = started.elapsed().as_secs_f64());

        Ok(input)
    }
}

/// A mock worker for testing purposes.
#[derive(Debug, Default)]
pub struct MockWorker {
    model: Option<Arc<MockModel>>,
    loaded_before: HashSet<String>,
    id: String,
    load_generations: HashMap<String, usize>,
}

impl MockWorker {
    /// Associate lifecycle events with the parent node's worker ID.
    pub fn with_id(id: String) -> Self {
        Self {
            id,
            ..Self::default()
        }
    }
}

impl Worker for MockWorker {
    type Model = MockModel;
    type ModelRef = Arc<MockModel>;

    async fn model(&self) -> Option<Self::ModelRef> {
        self.model.clone()
    }

    async fn load_model(
        &mut self,
        load_config: <Self::Model as InferenceModel>::LoadConfig,
    ) -> WorkerResult<Self::ModelRef, <Self::Model as InferenceModel>::Error> {
        if let Some(model) = &self.model {
            if model.name == load_config.name && model.config == load_config.config {
                return Ok(model.clone());
            }
        }
        // Validate the target before disturbing the current model.
        let mut load_config = load_config;
        let completed_loads = self
            .load_generations
            .get(&load_config.name)
            .copied()
            .unwrap_or_default();
        if let Some(seeds) = &load_config.noise_seeds {
            load_config.noise.seed = *seeds.get(completed_loads).ok_or_else(|| {
                ModelError::LoadConfigError {
                    message: format!(
                        "execution noise seed schedule exhausted for model {} at load generation {}",
                        load_config.name,
                        completed_loads + 1
                    ),
                }
            })?;
        }
        let mut model = MockModel::load(load_config)?;
        model.load_generation = completed_loads + 1;
        let model = Arc::new(model);
        if self.model.is_some() {
            self.unload_model().await?;
        }
        let first_load = !self.loaded_before.contains(&model.name);
        let started = std::time::Instant::now();
        tracing::info!(target: "simulation", event = "model_load_started", worker_id = %self.id, model = %model.name, profile = %model.config, first_load, noise_seed = model.noise_seed, load_generation = model.load_generation, delay_ms = model.lifecycle.load_ms);
        tokio::time::sleep(Duration::from_millis(model.lifecycle.load_ms)).await;
        self.loaded_before.insert(model.name.clone());
        self.load_generations
            .insert(model.name.clone(), model.load_generation);
        self.model = Some(model.clone());
        tracing::info!(target: "simulation", event = "model_load_finished", worker_id = %self.id, model = %model.name, profile = %model.config, first_load, noise_seed = model.noise_seed, load_generation = model.load_generation, elapsed_seconds = started.elapsed().as_secs_f64());
        Ok(model)
    }

    async fn unload_model(
        &mut self,
    ) -> WorkerResult<Self::ModelRef, <Self::Model as InferenceModel>::Error> {
        let model = self.model.as_ref().ok_or(WorkerError::ModelNotLoaded)?;
        let started = std::time::Instant::now();
        tracing::info!(target: "simulation", event = "model_unload_started", worker_id = %self.id, model = %model.name, profile = %model.config, delay_ms = model.lifecycle.unload_ms);
        tokio::time::sleep(Duration::from_millis(model.lifecycle.unload_ms)).await;
        tracing::info!(target: "simulation", event = "model_unload_finished", worker_id = %self.id, model = %model.name, profile = %model.config, elapsed_seconds = started.elapsed().as_secs_f64());
        // Cancellation before the wait completes retains ownership.
        Ok(self.model.take().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config(profile: &str) -> MockModelInitConfig {
        MockModelInitConfig::new(
            "resnet50".into(),
            profile.into(),
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../logs/model_latency.csv"),
        )
    }
    #[tokio::test(start_paused = true)]
    async fn waits_once_for_interpolated_batch_and_rejects_extrapolation() {
        let model = MockModel::load(config("1g.10gb")).unwrap();
        let expected = latency_profile::interpolate(&model.latency_profile, 3).unwrap();
        let input = DynTensor::random(vec![3, 0, 0, 0], model::tensor::TensorElementType::Float);
        let start = tokio::time::Instant::now();
        let output = model.forward(input).await.unwrap();
        assert_eq!(output.dimensions()[0], 3);
        assert!(start.elapsed().as_secs_f32() >= expected);
        assert!(start.elapsed().as_secs_f32() < expected + 0.003);
        assert!(
            model
                .forward(DynTensor::random(
                    vec![33, 0],
                    model::tensor::TensorElementType::Float
                ))
                .await
                .is_err()
        );
        assert!(
            model
                .forward(DynTensor::random(
                    vec![0, 0],
                    model::tensor::TensorElementType::Float
                ))
                .await
                .is_err()
        );
    }
    #[test]
    fn unknown_profile_fails_at_model_load() {
        assert!(MockModel::load(config("unknown")).is_err());
    }
    #[test]
    fn execution_noise_preserves_mean_cv_and_seeded_stream() {
        let noise = ExecutionNoise { cv: 0.1, seed: 7 };
        let mut rng = StdRng::seed_from_u64(noise.seed);
        let mut replay = rng.clone();
        assert_eq!(noise.sample(0.1, &mut rng), noise.sample(0.1, &mut replay));
        let (mut sum, mut squared) = (0.0, 0.0);
        for _ in 0..100_000 {
            let ratio = noise.sample(0.1, &mut rng) / 0.1;
            assert!(ratio.is_finite() && ratio > 0.0);
            sum += ratio;
            squared += ratio * ratio;
        }
        let mean = sum / 100_000.0;
        let cv = (squared / 100_000.0 - mean * mean).sqrt() / mean;
        assert!((mean - 1.0).abs() < 0.002);
        assert!((cv - noise.cv).abs() < 0.002);
        assert_eq!(ExecutionNoise::default().sample(0.1, &mut rng), 0.1);
        for cv in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
            assert!(ExecutionNoise { cv, seed: 7 }.validate().is_err());
        }
    }

    #[tokio::test(start_paused = true)]
    async fn noisy_forward_waits_for_the_sample_and_preserves_request_samples() {
        let model = MockModel::load(
            config("1g.10gb").with_execution_noise(ExecutionNoise { cv: 0.1, seed: 7 }),
        )
        .unwrap();
        let mut rng = model.noise_rng.lock().unwrap().0.clone();
        let mean = f64::from(latency_profile::interpolate(&model.latency_profile, 32).unwrap());
        let expected = model.noise.sample(mean, &mut rng);
        let start = tokio::time::Instant::now();
        let output = model
            .forward(DynTensor::random(
                vec![32, 0, 0, 0],
                model::tensor::TensorElementType::Float,
            ))
            .await
            .unwrap();
        assert_eq!(output.dimensions()[0], 32);
        let actual = start.elapsed().as_secs_f64();
        assert!(actual >= expected && actual < expected + 0.003);
    }

    #[tokio::test(start_paused = true)]
    async fn supplied_final_seed_is_independent_of_worker_id_model_and_profile() {
        let settings =
            || config("1g.10gb").with_execution_noise(ExecutionNoise { cv: 0.1, seed: 7 });
        let mut first_worker = MockWorker::with_id("different-worker-a".into());
        let mut second_worker = MockWorker::with_id("different-worker-b".into());
        let first = first_worker.load_model(settings()).await.unwrap();
        let second = second_worker
            .load_model(
                MockModelInitConfig::new(
                    "vgg16".into(),
                    "1g.20gb".into(),
                    settings().latency_profile_path,
                )
                .with_execution_noise(ExecutionNoise { cv: 0.1, seed: 7 }),
            )
            .await
            .unwrap();
        assert_eq!(first.noise_seed, 7);
        assert_eq!(second.noise_seed, 7);
        let mut first_rng = first.noise_rng.lock().unwrap().0.clone();
        let mut second_rng = second.noise_rng.lock().unwrap().0.clone();
        let mut external_rng = StdRng::seed_from_u64(7);
        for _ in 0..10 {
            let external_sample = first.noise.sample(1.0, &mut external_rng);
            assert_eq!(first.noise.sample(1.0, &mut first_rng), external_sample);
            assert_eq!(second.noise.sample(1.0, &mut second_rng), external_sample);
        }
    }

    #[tokio::test(start_paused = true)]
    async fn reload_without_a_schedule_restarts_the_supplied_seed() {
        let mut worker = MockWorker::default();
        let settings =
            || config("1g.10gb").with_execution_noise(ExecutionNoise { cv: 0.1, seed: 7 });
        let first = worker.load_model(settings()).await.unwrap();
        assert!(Arc::ptr_eq(
            &first,
            &worker.load_model(settings()).await.unwrap()
        ));
        worker.unload_model().await.unwrap();
        let reloaded = worker.load_model(settings()).await.unwrap();
        assert_eq!(first.noise_seed, 7);
        assert_eq!(reloaded.noise_seed, 7);
        assert_eq!(reloaded.load_generation, 2);
    }

    #[tokio::test(start_paused = true)]
    async fn explicit_seed_schedule_advances_only_for_successful_genuine_loads() {
        let mut worker = MockWorker::default();
        let settings = || {
            config("1g.10gb")
                .with_execution_noise(ExecutionNoise { cv: 0.1, seed: 7 })
                .with_execution_noise_seeds(vec![101, 102])
        };
        let first = worker.load_model(settings()).await.unwrap();
        assert_eq!(first.noise_seed, 101);
        assert_eq!(first.load_generation, 1);
        assert!(Arc::ptr_eq(
            &first,
            &worker.load_model(settings()).await.unwrap()
        ));
        let mut invalid = settings();
        invalid.config = "unknown".into();
        assert!(worker.load_model(invalid).await.is_err());
        assert!(Arc::ptr_eq(&first, &worker.model().await.unwrap()));
        worker.unload_model().await.unwrap();

        // Another model's loads do not consume this model's schedule.
        worker
            .load_model(
                MockModelInitConfig::new(
                    "vgg16".into(),
                    "1g.10gb".into(),
                    settings().latency_profile_path,
                )
                .with_execution_noise_seeds(vec![201]),
            )
            .await
            .unwrap();
        let second = worker.load_model(settings()).await.unwrap();
        assert_eq!(second.noise_seed, 102);
        assert_eq!(second.load_generation, 2);
        assert!(Arc::ptr_eq(
            &second,
            &worker.load_model(settings()).await.unwrap()
        ));
        worker.unload_model().await.unwrap();
        let error = format!("{:?}", worker.load_model(settings()).await.unwrap_err());
        assert!(error.contains("seed schedule exhausted"));
        assert!(error.contains("generation 3"));
        assert!(worker.model().await.is_none());
        assert_eq!(worker.load_generations["resnet50"], 2);
    }

    #[tokio::test(start_paused = true)]
    async fn an_explicit_empty_seed_schedule_fails_instead_of_using_a_fallback() {
        let mut worker = MockWorker::default();
        let error = format!(
            "{:?}",
            worker
                .load_model(config("1g.10gb").with_execution_noise_seeds(vec![]))
                .await
                .unwrap_err()
        );
        assert!(error.contains("seed schedule exhausted"));
        assert!(worker.load_generations.is_empty());
    }
    #[tokio::test(start_paused = true)]
    async fn lifecycle_waits_and_retries_do_not_charge_twice() {
        let mut worker = MockWorker::default();
        let costs = ModelLifecycle {
            load_ms: 1_000,
            unload_ms: 100,
        };
        let start = tokio::time::Instant::now();
        worker
            .load_model(config("1g.10gb").with_lifecycle(costs))
            .await
            .unwrap();
        assert!(start.elapsed() >= Duration::from_secs(1));
        let ready = tokio::time::Instant::now();
        worker
            .load_model(config("1g.10gb").with_lifecycle(costs))
            .await
            .unwrap();
        assert_eq!(ready.elapsed(), Duration::ZERO);
        assert!(
            tokio::time::timeout(Duration::from_millis(50), worker.unload_model())
                .await
                .is_err()
        );
        assert!(
            worker.model().await.is_some(),
            "cancelled unload must retain ownership"
        );
        let start = tokio::time::Instant::now();
        worker.unload_model().await.unwrap();
        assert!(start.elapsed() >= Duration::from_millis(100));
        assert!(worker.model().await.is_none());
        worker
            .load_model(config("1g.10gb").with_lifecycle(costs))
            .await
            .unwrap();
        assert_eq!(worker.loaded_before.len(), 1);
        let start = tokio::time::Instant::now();
        worker
            .load_model(
                MockModelInitConfig::new(
                    "vgg16".into(),
                    "1g.10gb".into(),
                    config("1g.10gb").latency_profile_path,
                )
                .with_lifecycle(costs),
            )
            .await
            .unwrap();
        assert!(start.elapsed() >= Duration::from_millis(1100));
        assert_eq!(worker.model().await.unwrap().name, "vgg16");
    }

    #[tokio::test(start_paused = true)]
    async fn cancelled_load_never_reports_ready() {
        let mut worker = MockWorker::default();
        assert!(
            tokio::time::timeout(
                Duration::from_millis(100),
                worker.load_model(
                    config("1g.10gb")
                        .with_lifecycle(ModelLifecycle {
                            load_ms: 1000,
                            unload_ms: 100
                        })
                        .with_execution_noise_seeds(vec![101])
                )
            )
            .await
            .is_err()
        );
        assert!(worker.model().await.is_none());
        assert!(worker.loaded_before.is_empty());
        assert!(worker.load_generations.is_empty());
        let retry = worker
            .load_model(config("1g.10gb").with_execution_noise_seeds(vec![101]))
            .await
            .unwrap();
        assert_eq!(retry.noise_seed, 101);
        assert_eq!(retry.load_generation, 1);
    }
}
