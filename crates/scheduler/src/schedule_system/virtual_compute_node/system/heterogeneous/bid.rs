//! Defines the `Bid` structure, representing a potential resource allocation.

use std::{
    any::Any,
    fmt::{Debug, Formatter},
    sync::Arc,
};

use uuid::Uuid;

use crate::schedule_system::{
    allocator::RankableShareTarget, virtual_compute_node::system::LatencyPredictor,
};

/// Represents a bid for a resource.
#[derive(Clone)]
pub struct Bid {
    /// The ID of the device.
    device_id: Uuid,
    /// The target id representing the specific resource configuration. resource from same device should have same target_id
    target_id: String,
    /// Additional data associated with the bid. This used by device which given this bid.
    pub(crate) data: Arc<dyn Any + Send + Sync>,
    latency_predictor: LatencyPredictor,
    resource_usage_rate: f32,
}

impl Bid {
    /// Creates a new `Bid`.
    ///
    /// # Arguments
    ///
    /// * `device_id` - The ID of the device.
    /// * `target_id` - The target hash representing the specific resource configuration.
    /// * `data` - Additional data associated with the bid.
    /// * `latency_predictor` - A function that predicts latency based on model ID and batch size.
    pub fn new(
        device_id: Uuid,
        target_id: String,
        data: Arc<dyn Any + Send + Sync>,
        latency_predictor: LatencyPredictor,
        resource_usage_rate: f32,
    ) -> Self {
        Self {
            device_id,
            target_id,
            data,
            latency_predictor,
            resource_usage_rate,
        }
    }

    /// Returns the device ID associated with the bid.
    pub fn device_id(&self) -> Uuid {
        self.device_id
    }

    /// Returns the target ID associated with the bid.
    pub fn target_id(&self) -> &str {
        &self.target_id
    }
}

impl Debug for Bid {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bid")
            .field("device_id", &self.device_id)
            .field("target_id", &self.target_id)
            .finish()
    }
}

impl RankableShareTarget for Bid {
    fn evaluate_latency(&self, model_id: &str, batch_size: u32) -> f32 {
        (*self.latency_predictor)(model_id, batch_size)
    }

    fn resource_usage_rate(&self) -> f32 {
        self.resource_usage_rate
    }
}
