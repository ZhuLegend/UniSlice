//! List of implementations of
//! [SchedulableSystem](crate::SchedulableSystem)
//! and [Scheduler](crate::Scheduler)
//! for compute nodes which implement in compute_node crate.

#[cfg(feature = "virtual_compute_node")]
use std::{collections::HashMap, pin::Pin, sync::Arc};

#[cfg(feature = "virtual_compute_node")]
pub mod virtual_compute_node;

#[cfg(feature = "virtual_compute_node")]
pub mod heterogeneous;

/// InferenceLatencyPredictor, input value is "model_name", "resource", "batch_size", return the latency in seconds
#[cfg(feature = "virtual_compute_node")]
pub type InferenceLatencyPredictor<ShareParam> =
    Arc<dyn Fn(&str, &ShareParam, u32) -> f32 + Send + Sync>;
/// LatencyPredictor, input value is "model_name", "batch_size", return the latency in seconds
#[cfg(feature = "virtual_compute_node")]
pub type LatencyPredictor = Arc<dyn Fn(&str, u32) -> f32 + Send + Sync>;
/// ModelRequestRatePredictor, input value is "model_name", return the request rate in requests per second
#[cfg(feature = "virtual_compute_node")]
pub type ModelRequestRatePredictor =
    Arc<dyn Fn() -> Pin<Box<dyn Future<Output = HashMap<String, f32>> + Send>> + Send + Sync>;
/// ResourceIdentifier, input value is "resource", return the resource identifier
#[cfg(feature = "virtual_compute_node")]
pub type ResourceIdentifier<Resource> = Box<dyn Fn(&Resource) -> String + Send>;
