//! Contains core traits for resource allocation abstractions.

use std::fmt::Debug;

/// A trait for a resource unit that can be evaluated for its performance.
pub trait RankableShareTarget: Debug {
    /// Evaluates the "cost" (e.g., latency) of this target for a given model and batch size.
    fn evaluate_latency(&self, model_id: &str, batch_size: u32) -> f32;

    /// The use rate of this resource unit, from 0.0 (idle) to 1.0 (fully utilized).
    ///
    /// For example, if a GPU can split its resources into two resource A, then the usage rate of A is 0.5.
    ///
    /// If a GPU can split its resources into four resource B, then the usage rate of B is 0.25.
    fn resource_usage_rate(&self) -> f32;
}

/// A trait abstracting a resource pool.
/// It allows the scheduler to be generic over the resource management backend.
pub trait ResourceAllocator: Debug {
    /// The type of resource unit this allocator manages.
    /// It must be rankable, cloneable, and orderable.
    type Target: RankableShareTarget + Clone;

    /// Returns all currently available resource units for selection.
    fn choosable_share_targets(&self) -> Vec<Self::Target>;

    /// Allocates a specific resource unit from the pool.
    /// Returns `Some(())` on success, `None` on failure.
    fn share(&mut self, target: Self::Target) -> Option<()>;

    /// Returns a previously allocated resource unit to the pool.
    ///
    /// Returns true on success, false if the target was invalid.
    fn refund(&mut self, target: Self::Target) -> bool;

    /// Returns the minimum possible allocation units.
    fn min_targets(&self) -> Vec<Self::Target>;

    /// Checks if a target can be shared from the current resource pool.
    fn can_share(&self, target: Self::Target) -> bool;
}
