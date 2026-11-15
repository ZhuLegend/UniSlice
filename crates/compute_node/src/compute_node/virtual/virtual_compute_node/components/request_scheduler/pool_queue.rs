//! Local ownership and admission boundary for a pool's model queues.
use std::{
    collections::{HashSet, VecDeque},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use dashmap::DashMap;
use tokio::sync::{mpsc, watch};
use uuid::Uuid;

use super::model_queue::{RequestEntry, Requests};

/// Admission was rejected before any work was queued or executed.
pub const POOL_UNAVAILABLE: &str = "pool is not accepting unclaimed work";

/// Opaque movable fragments. Their original completion/merge channels travel
/// with them; transferring never creates a second execution of a request.
pub struct PendingRequests {
    pub(super) model: String,
    pub(super) entries: VecDeque<RequestEntry>,
}
impl PendingRequests {
    /// Model shared by these unclaimed fragments.
    pub fn model(&self) -> &str {
        &self.model
    }
    /// Number of fragments, which may differ from original request count.
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    /// Whether there is no unclaimed work.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    /// Existing fragment identities for handoff tracing.
    pub fn task_ids(&self) -> Vec<Uuid> {
        self.entries.iter().map(|entry| entry.work.id()).collect()
    }
}

#[derive(Default)]
pub(super) struct ModelAvailability {
    pub(super) workers: HashSet<Uuid>,
    epoch: u64,
}

#[derive(Clone)]
/// Admission, extraction and transfer boundary for one resource pool.
pub struct PoolRequestQueue {
    pub(super) queues: Arc<DashMap<String, Requests>>,
    states: Arc<DashMap<String, Arc<Mutex<ModelAvailability>>>>,
    events: Arc<DashMap<String, mpsc::UnboundedSender<()>>>,
    changes: watch::Sender<u64>,
    tracked: Arc<AtomicBool>,
}
impl PoolRequestQueue {
    pub(super) fn new(
        queues: Arc<DashMap<String, Requests>>,
        events: Arc<DashMap<String, mpsc::UnboundedSender<()>>>,
    ) -> Self {
        Self {
            queues,
            events,
            states: Default::default(),
            changes: watch::channel(0).0,
            tracked: Default::default(),
        }
    }
    /// Enforce live-worker admission; standalone queue tests may leave tracking disabled.
    pub(in crate::compute_node::r#virtual::virtual_compute_node::components) fn enable_worker_tracking(
        &self,
    ) {
        self.tracked.store(true, Ordering::Release);
    }
    pub(super) fn state(&self, model: &str) -> Arc<Mutex<ModelAvailability>> {
        self.states.entry(model.to_owned()).or_default().clone()
    }
    pub(super) fn serving_locked(&self, state: &ModelAvailability) -> bool {
        !self.tracked.load(Ordering::Acquire) || !state.workers.is_empty()
    }
    pub(super) fn worker_eligible(&self, state: &ModelAvailability, worker: Uuid) -> bool {
        !self.tracked.load(Ordering::Acquire) || state.workers.contains(&worker)
    }
    /// Current serving state and generation, used to reject stale handoffs.
    pub fn availability(&self, model: &str) -> (bool, u64) {
        let Some(state) = self.states.get(model).map(|state| state.clone()) else {
            return (!self.tracked.load(Ordering::Acquire), 0);
        };
        let state = state.lock().unwrap();
        (self.serving_locked(&state), state.epoch)
    }
    /// Whether this pool has encountered a worker or queue for this model.
    pub fn knows_model(&self, model: &str) -> bool {
        self.states.contains_key(model)
    }
    /// Models whose queues may contain pending work.
    pub fn models(&self) -> Vec<String> {
        self.queues
            .iter()
            .map(|entry| entry.key().clone())
            .collect()
    }
    /// Models with at least one worker eligible to claim a new batch.
    pub fn serving_models(&self) -> Vec<String> {
        let models = self
            .states
            .iter()
            .map(|entry| entry.key().clone())
            .collect::<Vec<_>>();
        models
            .into_iter()
            .filter(|model| self.availability(model).0)
            .collect()
    }
    /// Subscribe before reading state to avoid missed transitions.
    pub fn subscribe(&self) -> watch::Receiver<u64> {
        self.changes.subscribe()
    }
    pub(super) fn notify_change(&self) {
        self.changes
            .send_modify(|generation| *generation = generation.wrapping_add(1));
    }
    pub(super) fn wake(&self, model: &str) {
        if let Some(sender) = self.events.get(model) {
            let _ = sender.send(());
        }
    }

    /// Called synchronously by the registry while committing a worker transition.
    /// This lock also excludes enqueue, whole batch extraction and handoff.
    pub(in crate::compute_node::r#virtual::virtual_compute_node::components) fn worker_changed(
        &self,
        model: &str,
        worker: Uuid,
        running: bool,
    ) {
        let state = self.state(model);
        let mut state = state.lock().unwrap();
        let changed = if running {
            state.workers.insert(worker)
        } else {
            state.workers.remove(&worker)
        };
        if changed {
            state.epoch = state.epoch.wrapping_add(1);
            tracing::info!(target: "allocation", event = "pool_model_availability", model, worker_id = %worker, running_workers = state.workers.len(), epoch = state.epoch);
            self.notify_change();
        }
        drop(state);
        self.wake(model);
    }

    #[cfg(test)]
    pub(super) fn test_worker_changed(&self, model: &str, worker: Uuid, running: bool) {
        self.worker_changed(model, worker, running);
    }

    /// Take only unclaimed work after the last eligible worker has left.
    pub fn take_unavailable(&self, model: &str) -> Option<PendingRequests> {
        let state = self.state(model);
        let state = state.lock().unwrap();
        if self.serving_locked(&state) {
            return None;
        }
        let queue = self.queues.get(model)?;
        let mut entries = queue.drain();
        entries.retain(|entry| !entry.response_tx.is_closed());
        if entries.is_empty() {
            None
        } else {
            Some(PendingRequests {
                model: model.to_owned(),
                entries,
            })
        }
    }

    pub(super) fn accept(
        &self,
        mut pending: PendingRequests,
        epoch: u64,
    ) -> Result<(), PendingRequests> {
        let state = self.state(&pending.model);
        let state = state.lock().unwrap();
        if !self.serving_locked(&state) || state.epoch != epoch {
            return Err(pending);
        }
        let queue = self
            .queues
            .entry(pending.model.clone())
            .or_insert_with(|| Requests::new(VecDeque::new()));
        while let Some(entry) = pending.entries.pop_front() {
            if !entry.response_tx.is_closed() {
                queue.push_back(entry);
            }
        }
        drop(queue);
        drop(state);
        self.wake(&pending.model);
        Ok(())
    }

    /// Restore ownership if no target can accept. Does not emit another central
    /// event, so a global service gap cannot become a busy handoff loop.
    pub fn restore(&self, pending: PendingRequests) {
        let state = self.state(&pending.model);
        let _state = state.lock().unwrap();
        let queue = self
            .queues
            .entry(pending.model.clone())
            .or_insert_with(|| Requests::new(VecDeque::new()));
        for entry in pending.entries.into_iter().rev() {
            if !entry.response_tx.is_closed() {
                queue.push_front(entry);
            }
        }
        drop(queue);
        self.wake(&pending.model);
    }
}
