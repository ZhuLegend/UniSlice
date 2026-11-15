//! Explicit simulation-only random inputs. UUIDs never select a random stream.
use std::{
    collections::BTreeMap,
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

use serde::{Deserialize, Serialize};

/// Fully resolved inputs produced and snapshotted by the experiment runner.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RandomInputs {
    /// Current seed and keyed-draw contract.
    pub schema_version: u32,
    /// External replicate seed, for provenance only.
    pub root_seed: u64,
    /// Final seed for every routing/extraction/controller stream.
    pub seeds: BTreeMap<String, u64>,
    /// Logical physical worker slot -> model -> explicit load-generation seeds.
    pub worker_noise: BTreeMap<String, BTreeMap<String, Vec<u64>>>,
}

static INPUTS: OnceLock<Option<RandomInputs>> = OnceLock::new();
static WORKER_SLOTS: OnceLock<Mutex<WorkerSlots>> = OnceLock::new();

#[derive(Default)]
struct WorkerSlots {
    next: BTreeMap<String, usize>,
    identities: BTreeMap<uuid::Uuid, String>,
}

impl WorkerSlots {
    fn resolve(
        &mut self,
        inputs: &RandomInputs,
        prefix: &str,
        worker_id: uuid::Uuid,
    ) -> Result<BTreeMap<String, Vec<u64>>, String> {
        let key = if let Some(key) = self.identities.get(&worker_id) {
            if key.rsplit_once('/').map(|(parent, _)| parent) != Some(prefix) {
                return Err("worker already has a different explicit noise slot".into());
            }
            key.clone()
        } else {
            format!(
                "{prefix}/{}",
                self.next.get(prefix).copied().unwrap_or_default()
            )
        };
        let result = inputs
            .worker_noise
            .get(&key)
            .ok_or_else(|| format!("explicit worker noise slot exhausted: {key}"))?
            .clone();
        if !self.identities.contains_key(&worker_id) {
            *self.next.entry(prefix.into()).or_default() += 1;
            self.identities.insert(worker_id, key.clone());
        }
        tracing::info!(target: "simulation", event="worker_noise_slot", %worker_id, noise_slot=%key);
        Ok(result)
    }
}

fn inputs() -> Option<&'static RandomInputs> {
    INPUTS
        .get_or_init(|| {
            std::env::var_os("HSGS_EXPERIMENT_RANDOM_INPUTS").map(|path| {
                let value: RandomInputs = serde_json::from_slice(
                    &std::fs::read(path).expect("cannot read explicit experiment random inputs"),
                )
                .expect("invalid explicit experiment random inputs");
                assert_eq!(value.schema_version, 1, "unknown random-input schema");
                value
            })
        })
        .as_ref()
}

/// Final injected seed; missing entries fail rather than add hidden entropy.
pub fn seed(key: &str) -> Option<u64> {
    inputs().map(|value| {
        *value
            .seeds
            .get(key)
            .unwrap_or_else(|| panic!("missing explicit random stream: {key}"))
    })
}

/// Whether this process has explicitly configured experimental random inputs.
pub fn enabled() -> bool {
    inputs().is_some()
}

/// Resolve a parent/profile creation slot from the explicit external table.
pub fn worker_noise(
    namespace: &str,
    profile: &str,
    worker_id: uuid::Uuid,
) -> Result<BTreeMap<String, Vec<u64>>, String> {
    let Some(value) = inputs() else {
        return Ok(BTreeMap::new());
    };
    let prefix = format!("{namespace}/{profile}");
    WORKER_SLOTS
        .get_or_init(Default::default)
        .lock()
        .unwrap()
        .resolve(value, &prefix, worker_id)
}

/// External logical slot associated with a proxy; UUID is only an opaque lookup handle.
pub fn worker_noise_slot(worker_id: uuid::Uuid) -> Option<String> {
    WORKER_SLOTS
        .get()?
        .lock()
        .unwrap()
        .identities
        .get(&worker_id)
        .cloned()
}

/// SplitMix64 keyed draw, versioned by schema 1. Not a stream-seed derivation.
pub fn keyed_u64(seed: u64, ordinal: u64) -> u64 {
    let mut value = seed.wrapping_add(ordinal.wrapping_add(1).wrapping_mul(0x9e3779b97f4a7c15));
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

/// Uniform in [0,1), keyed by original request ordinal for stable route draws.
pub fn uniform(seed: u64, ordinal: u64) -> f32 {
    ((keyed_u64(seed, ordinal) >> 40) as f32) / 16_777_216.0
}

/// A local, auditable stream for probabilistic extraction or AIMD rounding.
#[derive(Debug)]
pub struct SeededStream {
    seed: u64,
    ordinal: AtomicU64,
}
impl SeededStream {
    /// Consume a final external seed without mixing in process identities.
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            ordinal: AtomicU64::new(0),
        }
    }
    /// Next integer draw. Each logical component owns its counter.
    pub fn next_u64(&self) -> u64 {
        keyed_u64(self.seed, self.ordinal.fetch_add(1, Ordering::Relaxed))
    }
    /// Next uniform draw.
    pub fn next_f32(&self) -> f32 {
        ((self.next_u64() >> 40) as f32) / 16_777_216.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn external_worker_slots_ignore_uuid_values_and_retries_keep_their_binding() {
        let prefix = "node/0/1g.10gb";
        let inputs = RandomInputs {
            schema_version: 1,
            root_seed: 101,
            seeds: BTreeMap::new(),
            worker_noise: BTreeMap::from([
                (
                    format!("{prefix}/0"),
                    BTreeMap::from([("resnet50".into(), vec![11, 12])]),
                ),
                (
                    format!("{prefix}/1"),
                    BTreeMap::from([("resnet50".into(), vec![21, 22])]),
                ),
            ]),
        };
        let mut first = WorkerSlots::default();
        let mut replay = WorkerSlots::default();
        let worker_a = uuid::Uuid::from_u128(1);
        let worker_b = uuid::Uuid::from_u128(2);
        let other_a = uuid::Uuid::from_u128(100);
        let other_b = uuid::Uuid::from_u128(200);
        assert_eq!(
            first.resolve(&inputs, prefix, worker_a).unwrap(),
            replay.resolve(&inputs, prefix, other_a).unwrap()
        );
        assert_eq!(
            first.resolve(&inputs, prefix, worker_a).unwrap()["resnet50"],
            vec![11, 12]
        );
        assert_eq!(first.next[prefix], 1);
        assert_eq!(
            first.resolve(&inputs, prefix, worker_b).unwrap(),
            replay.resolve(&inputs, prefix, other_b).unwrap()
        );
        assert_eq!(first.identities[&worker_a], format!("{prefix}/0"));
        assert_eq!(first.identities[&worker_b], format!("{prefix}/1"));
        assert!(
            first
                .resolve(&inputs, "another/0/1g.10gb", worker_a)
                .is_err()
        );
        assert!(
            first
                .resolve(&inputs, prefix, uuid::Uuid::from_u128(3))
                .is_err()
        );
        assert_eq!(first.next[prefix], 2);
        assert_eq!(first.identities.len(), 2);
    }

    #[test]
    fn explicit_streams_replay_and_request_draws_ignore_call_order() {
        let first = SeededStream::new(101);
        let second = SeededStream::new(101);
        let different = SeededStream::new(102);
        for ordinal in 0..100 {
            assert_eq!(first.next_u64(), second.next_u64());
            assert_ne!(keyed_u64(101, ordinal), different.next_u64());
            assert!((0.0..1.0).contains(&uniform(101, ordinal)));
        }
        assert_eq!(keyed_u64(0, 0), 0xe220a8397b1dcdaf);
        assert_eq!(uniform(101, 17), uniform(101, 17));
    }
}
