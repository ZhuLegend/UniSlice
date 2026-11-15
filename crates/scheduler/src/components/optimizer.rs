//! General optimizer implementation for the scheduler.

use std::{collections::HashMap, hash::Hash, iter::Sum, ops::Div};

use compute_node::ComputeNode;
use pathfinding::{
    num_traits::{Bounded, Signed, Zero},
    prelude::{Matrix, kuhn_munkres_min},
};

use crate::components::PossibleCommandInfo;

/// Use Kuhn-Munkres algorithm to optimize the assignment of compute nodes to target states.
///
/// The impossible path will be assigned a cost of `Cost::max_value() / 2`. Take care to choose a suitable `Cost` type that can handle this value without overflow.
pub struct KuhnMunkresOptimizer;

impl<ID: Hash + Eq, Node: ComputeNode, Cost: Ord + Eq> Optimizer<ID, Node, Cost>
    for KuhnMunkresOptimizer
where
    Node::State: Eq,
    Cost: Bounded + Sum<Cost> + Zero + Signed + Ord + Copy + From<u32> + Div<Cost>,
    Node::Command: Clone,
{
    fn optimize(
        target_state: &[Node::State],
        possible_commands: std::collections::HashMap<
            ID,
            Vec<crate::components::PossibleCommandInfo<Node, Cost>>,
        >,
    ) -> std::collections::HashMap<ID, Option<Vec<Node::Command>>> {
        assert_eq!(target_state.len(), possible_commands.len());
        let nodes = possible_commands.into_iter().collect::<Vec<_>>();
        let matrix = Matrix::from_fn(target_state.len(), target_state.len(), |(i, j)| {
            nodes[i]
                .1
                .iter()
                .find(|&info| info.target_state == &target_state[j])
                .map(|info| info.cost)
                .unwrap_or(Cost::max_value() / 2.into())
        });
        let (_, result) = kuhn_munkres_min(&matrix);

        nodes
            .into_iter()
            .zip(result)
            .map(|((node_idx, possible_commands), state_idx)| {
                if let Some(info) = possible_commands
                    .into_iter()
                    .find(|info| info.target_state == &target_state[state_idx])
                {
                    (node_idx, Some(info.commands))
                } else {
                    (node_idx, None)
                }
            })
            .collect()
    }
}

/// Check available commands to change [ComputeNode]'s current state to the target state, and choose the best one for each compute node
pub trait Optimizer<ID: Hash, Node: ComputeNode, Cost: Ord + Eq>
where
    Node::State: Eq,
{
    /// Check available commands to change [ComputeNode]'s current state to the target state, and choose the best one for each compute node
    ///
    /// Given a list of target states and a list of possible commands, decide which command to execute for each compute node to reach the target state
    #[allow(clippy::type_complexity)]
    fn optimize(
        target_state: &[Node::State],
        possible_commands: HashMap<ID, Vec<PossibleCommandInfo<Node, Cost>>>,
    ) -> HashMap<ID, Option<Vec<<Node as ComputeNode>::Command>>>;
}
