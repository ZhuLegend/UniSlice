use std::{collections::HashMap, marker::PhantomData, ops::Deref};

use compute_node::ComputeNode;
use futures::future::{JoinAll, TryJoinAll};
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::components::{
    BaseScheduler, CommandCostPredictor, ComputeNodeAdapter, PossibleCommandInfo, Reconciler,
    SystemContext, optimizer::Optimizer,
};

/// A Schedulable System which compose of [ComputeNodeAdapter] and [SystemContext]
pub struct SchedulableSystem<Node, Context, Adapter> {
    context: Mutex<Context>,
    adapter: PhantomData<fn() -> Adapter>,
    nodes: PhantomData<fn() -> Node>,
}

impl<Node, Context, Adapter> crate::SchedulableSystem for SchedulableSystem<Node, Context, Adapter>
where
    Node: ComputeNode + Send + Sync,
    Context: SystemContext + AsRef<HashMap<Uuid, Node>> + Clone + Send,
    Adapter: ComputeNodeAdapter<Node, Context>,
{
    type ComputeNodeID = Uuid;
    type ComputeNode = Node;
    type Context = Context;
    type Error = String;

    async fn query_context(&self) -> Self::Context {
        let mut context = self.context.lock().await;
        let _ = context.update_status().await;
        context.clone()
    }

    async fn execute_commands<
        I: IntoIterator<
                Item = (
                    Self::ComputeNodeID,
                    Vec<<<Self as crate::SchedulableSystem>::ComputeNode as ComputeNode>::Command>,
                ),
            > + Send,
    >(
        &self,
        command_pairs: I,
    ) -> Result<(), Self::Error> {
        let context = self.context.lock().await;
        let nodes = context.as_ref();
        let execute_results = command_pairs
            .into_iter()
            .map(async |(id, commands)| {
                let node = nodes.get(&id).ok_or("Node not found")?;
                commands
                    .into_iter()
                    .map(async |command| node.handle(command).await)
                    .collect::<TryJoinAll<_>>()
                    .await
            })
            .collect::<JoinAll<_>>()
            .await;

        if execute_results.iter().any(Result::is_err) {
            Err("Execution failed".to_string())
        } else {
            Ok(())
        }
    }
}

/// A Scheduler which composed by components
pub struct Scheduler<
    Context,
    NodeScheduler,
    ComputeNode,
    Adapter,
    Reconciler,
    CostPredictor,
    Optimizer,
> where
    NodeScheduler: BaseScheduler<Context>,
    Context: SystemContext,
{
    scheduler: NodeScheduler,

    #[allow(clippy::type_complexity)]
    system: PhantomData<fn() -> SchedulableSystem<ComputeNode, Context, Adapter>>,
    adapter: PhantomData<fn() -> Adapter>,
    reconciler: PhantomData<fn() -> Reconciler>,
    cost_predictor: PhantomData<fn() -> CostPredictor>,
    optimizer: PhantomData<fn() -> Optimizer>,
}

impl<Context, NodeScheduler, ComputeNode, Adapter, NodeReconciler, CostPredictor, DecideOptimizer>
    Scheduler<
        Context,
        NodeScheduler,
        ComputeNode,
        Adapter,
        NodeReconciler,
        CostPredictor,
        DecideOptimizer,
    >
where
    NodeScheduler: BaseScheduler<Context>,
    Context: AsRef<HashMap<Uuid, ComputeNode>> + SystemContext,
    ComputeNode: compute_node::ComputeNode,
    NodeReconciler: Reconciler<ComputeNode>,
    CostPredictor: CommandCostPredictor<ComputeNode, Context>,
{
    /// calculate possible commands to reach target state
    /// return (target state, commands)
    fn calculate_possible_command<'a, 'b>(
        curr_state: &'a ComputeNode::State,
        target_state: impl IntoIterator<Item = &'b ComputeNode::State>,
    ) -> impl Iterator<Item = (&'b ComputeNode::State, Vec<ComputeNode::Command>)>
    where
        <ComputeNode as compute_node::ComputeNode>::State: 'b,
    {
        target_state.into_iter().filter_map(move |state| {
            // Reconciler::reconcile returns None if the target state can't be reached, some simply ignore it
            let commands = NodeReconciler::reconcile(curr_state, state)?;
            Some((state, commands))
        })
    }

    /// get possible commands to reach target state
    ///
    /// return {compute node id: (current state, Vec<(target state, commands, command's cost)>)}
    async fn get_possible_commands<'a>(
        context: &'a Context,
        target_states: &'a [ComputeNode::State],
    ) -> Result<
        HashMap<
            Uuid,
            (
                ComputeNode::State,
                Vec<PossibleCommandInfo<'a, ComputeNode, CostPredictor::Cost>>,
            ),
        >,
        String,
    > {
        context
            .as_ref()
            .iter()
            .map(async |(id, node)| {
                let curr_state = node.state().await?;
                let command_infos = Self::calculate_possible_command(&curr_state, target_states)
                    .map(|(target_state, commands)| {
                        let cost = CostPredictor::predict(context, commands.iter());
                        PossibleCommandInfo {
                            target_state,
                            commands,
                            cost,
                        }
                    })
                    .collect::<Vec<_>>();

                Ok((*id, (curr_state, command_infos)))
            })
            .collect::<TryJoinAll<_>>()
            .await
            .map(|command_infos| command_infos.into_iter().collect::<HashMap<_, _>>())
    }
}

impl<Context, NodeScheduler, ComputeNode, Adapter, NodeReconciler, CostPredictor, DecideOptimizer>
    crate::Scheduler
    for Scheduler<
        Context,
        NodeScheduler,
        ComputeNode,
        Adapter,
        NodeReconciler,
        CostPredictor,
        DecideOptimizer,
    >
where
    NodeScheduler: BaseScheduler<Context>,
    Context: Clone + AsRef<HashMap<Uuid, ComputeNode>> + Send + Sync + SystemContext,
    ComputeNode: compute_node::ComputeNode + Send + Sync,
    ComputeNode::State: Eq,
    Adapter: ComputeNodeAdapter<ComputeNode, Context>,
    NodeReconciler: Reconciler<ComputeNode>,
    CostPredictor: CommandCostPredictor<ComputeNode, Context>,
    DecideOptimizer: Optimizer<Uuid, ComputeNode, CostPredictor::Cost>,
{
    type System = SchedulableSystem<ComputeNode, Context, Adapter>;

    type Error = String;

    async fn schedule(&self, context: impl Deref<Target=<Self::System as crate::SchedulableSystem>::Context>)
        -> Result<
            impl IntoIterator<
                Item=
                (
                    <Self::System as crate::SchedulableSystem>::ComputeNodeID,
                    Vec<<<Self::System as crate::SchedulableSystem>::ComputeNode as compute_node::ComputeNode>::Command>
                )
            >,
            Self::Error,
    >{
        let context = context.deref();
        let target_context = self.scheduler.schedule(context);
        let target_states = Adapter::adapt_context_to_nodes(context, &target_context)
            .map_err(|e| format!("Adapter error: {e:?}"))?;

        let mut possible_commands = Self::get_possible_commands(context, &target_states).await?;

        // sort possible commands by cost
        possible_commands.values_mut().for_each(|commands| {
            commands.1.sort_by(|a, b| a.cost.cmp(&b.cost));
        });

        let possible_commands = possible_commands
            .into_iter()
            .map(|(key, value)| (key, value.1))
            .collect();

        let decisions = DecideOptimizer::optimize(&target_states, possible_commands);

        Ok(decisions
            .into_iter()
            .filter_map(|(key, value)| value.map(|value| (key, value))))
    }
}
