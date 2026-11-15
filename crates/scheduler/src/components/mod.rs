//! Scheduler Component

use std::{fmt::Debug, ops::Deref};

use compute_node::ComputeNode;

// mod r#impl;

mod optimizer;

pub use optimizer::Optimizer;

/// Schedule System Context
pub trait SystemContext {
    /// Unchangeable system running environment, such as GPU、inference model type
    type Environment;

    /// Instantaneous system status, such as GPU memory usage and GPU utilization
    type Status;

    /// System configurable state, such as loaded model, GPU partitioning strategy
    type State;

    /// Unchangeable system running environment
    fn environment(&self) -> impl Deref<Target = Self::Environment>;

    /// Instantaneous system status
    fn status(&self) -> impl Deref<Target = Self::Status>;

    /// Update the instantaneous system status and return the new status
    fn update_status(&mut self) -> impl Future<Output = impl Deref<Target = Self::Status>> + Send;

    /// System configurable state
    fn state(&self) -> impl Deref<Target = Self::State>;

    /// Update system configurable state
    fn update_state(&mut self, new_state: Self::State);
}

/// Decide what state should system be in
pub trait BaseScheduler<Context: SystemContext> {
    /// schedule the system
    fn schedule(&self, context: impl Deref<Target = Context>) -> <Context as SystemContext>::State;
}

/// [ComputeNode] adapter
///
/// [ComputeNodeAdapter] is used to adapt [ComputeNode] to the system context
/// it can update system context according to a list of [ComputeNode]'s state
/// and generate list of [ComputeNode]'s state according target system context
pub trait ComputeNodeAdapter<Node: ComputeNode, Context: SystemContext> {
    /// Error type
    type Error: Debug;

    /// Adapt [ComputeNode] to the system context's state
    fn adapt_nodes_to_context(
        nodes: impl IntoIterator<Item = impl Deref<Target = Node::State>>,
    ) -> Context::State;

    /// Adapt system context to [ComputeNode]
    fn adapt_context_to_nodes(
        context: impl Deref<Target = Context>,
        target_state: impl Deref<Target = Context::State>,
    ) -> Result<Vec<Node::State>, Self::Error>;
}

/// [ComputeNode]'s command execute cost predictor
pub trait CommandCostPredictor<Node: ComputeNode, Context: SystemContext> {
    /// Predicted cost of commands
    type Cost: Ord + Eq;

    /// Predict the cost of commands
    fn predict(
        context: impl Deref<Target = Context>,
        commands: impl IntoIterator<Item = impl Deref<Target = <Node as ComputeNode>::Command>>,
    ) -> Self::Cost;
}

/// Reconcile a list of commands to change [ComputeNode]'s current state to the target state
pub trait Reconciler<Node: ComputeNode> {
    /// Reconcile a list of commands to change [ComputeNode]'s current state to the target state
    ///
    /// None means the target state can't be reached
    fn reconcile(
        current: impl Deref<Target = <Node as ComputeNode>::State>,
        target: impl Deref<Target = <Node as ComputeNode>::State>,
    ) -> Option<Vec<<Node as ComputeNode>::Command>>;
}

/// PossibleCommandInfo
pub struct PossibleCommandInfo<'a, ComputeNode, Cost>
where
    ComputeNode: compute_node::ComputeNode,
    Cost: Eq + Ord,
{
    pub(crate) target_state: &'a ComputeNode::State,
    pub(crate) commands: Vec<ComputeNode::Command>,
    pub(crate) cost: Cost,
}
