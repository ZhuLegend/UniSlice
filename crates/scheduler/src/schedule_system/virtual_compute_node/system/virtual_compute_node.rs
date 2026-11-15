//! VirtualComputeNode system module.
//! This module contains the SystemContext for the VirtualComputeNode system and SchedulableSystem implementation.

use std::{collections::HashMap, fmt::Debug, iter, ops::RangeInclusive, pin::Pin, sync::Arc};

use bincode::{Decode, Encode};
use compute_node::{
    ComputeNode,
    compute_device::vgpu::{
        ShareableComputeDeviceResource,
        share_target::{ChoosableShareTargetContainer, ShareTarget},
    },
    compute_node::r#virtual::{
        base,
        virtual_compute_node::{Command, State, Status, VirtualComputeNode, WorkTask},
    },
    util::{Input, Output},
};
use serde::{Serialize, de::DeserializeOwned};
use tokio::try_join;
#[cfg(feature = "tracing")]
use tracing::info;

use crate::{
    SchedulableSystem, Scheduler, SystemContext,
    schedule_system::{
        allocator::{RankableShareTarget, ResourceAllocator},
        schedulers::greedy::GreedyScheduler,
        virtual_compute_node::{
            algorithms::buddy_cell::{
                BuddyCellAllocator, BuddyCellResource, CanExceedLimit,
                constrained::{BuddyCellResourceImplHelper, ConstrainedResource},
                regroup_buddy_cell,
            },
            system::{InferenceLatencyPredictor, ModelRequestRatePredictor},
        },
    },
};

/// Context for the [VirtualComputeNode] system.
pub struct Context<
    ShareParam: ShareTarget
        + Serialize
        + DeserializeOwned
        + Debug
        + Encode
        + Decode<()>
        + Send
        + Sync
        + 'static,
    ShareResource: ShareableComputeDeviceResource + for<'a> From<&'a ShareParam> + Clone,
    TargetContainer: ChoosableShareTargetContainer<Target = ShareParam>,
> {
    state: State<ShareParam>,
    status: Status,
    max_resource: ShareResource,
    container: TargetContainer,
    latency_predictor: InferenceLatencyPredictor<ShareParam>,
    predictor: ModelRequestRatePredictor,
    batch_size_range: RangeInclusive<u32>,
}

impl<
    ShareParam: ShareTarget
        + Serialize
        + DeserializeOwned
        + Debug
        + Encode
        + Decode<()>
        + Send
        + Sync
        + 'static,
    ShareResource: ShareableComputeDeviceResource + for<'a> From<&'a ShareParam> + Clone,
    TargetContainer: ChoosableShareTargetContainer<Target = ShareParam>,
> Debug for Context<ShareParam, ShareResource, TargetContainer>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Context")
            .field("state", &self.state)
            .field("status", &self.status)
            .finish()
    }
}

impl<
    ShareParam: ShareTarget
        + Serialize
        + DeserializeOwned
        + Debug
        + Encode
        + Decode<()>
        + Send
        + Sync
        + 'static,
    ShareResource: ShareableComputeDeviceResource + for<'a> From<&'a ShareParam> + Clone,
    TargetContainer: ChoosableShareTargetContainer<Target = ShareParam>,
> SystemContext for Context<ShareParam, ShareResource, TargetContainer>
{
    type State = State<ShareParam>;

    fn state(&self) -> &Self::State {
        &self.state
    }
}

impl<
    ShareParam: ShareTarget
        + Serialize
        + DeserializeOwned
        + Debug
        + Encode
        + Decode<()>
        + Send
        + Sync
        + 'static,
    ShareResource: ShareableComputeDeviceResource + for<'a> From<&'a ShareParam> + Clone,
    TargetContainer: ChoosableShareTargetContainer<Target = ShareParam>,
> AsRef<Context<ShareParam, ShareResource, TargetContainer>>
    for Context<ShareParam, ShareResource, TargetContainer>
{
    fn as_ref(&self) -> &Context<ShareParam, ShareResource, TargetContainer> {
        self
    }
}

/// Single device virtual compute node system
pub struct SingleDeviceVirtualComputeNodeSystem<
    ShareParam: ShareTarget
        + Serialize
        + DeserializeOwned
        + Debug
        + Encode
        + Decode<()>
        + Send
        + Sync
        + 'static,
    ShareResource: ShareableComputeDeviceResource + for<'a> From<&'a ShareParam> + Clone,
    DeviceState: base::DeviceState + Encode + Decode<()> + Send + Sync + 'static,
    TargetContainer: ChoosableShareTargetContainer<Target = ShareParam>,
> {
    node: VirtualComputeNode<ShareParam, DeviceState>,
    resource_template: ShareResource,
    container: TargetContainer,
    latency_predictor: InferenceLatencyPredictor<ShareParam>,
    model_request_rate_predictor: ModelRequestRatePredictor,
    batch_size_range: RangeInclusive<u32>,
}

impl<
    ShareParam: ShareTarget
        + Serialize
        + DeserializeOwned
        + Debug
        + Encode
        + Decode<()>
        + Send
        + Sync
        + 'static,
    ShareResource: ShareableComputeDeviceResource + for<'a> From<&'a ShareParam> + Clone,
    DeviceState: base::DeviceState + Encode + Decode<()> + Send + Sync + 'static,
    TargetContainer: ChoosableShareTargetContainer<Target = ShareParam>,
> SingleDeviceVirtualComputeNodeSystem<ShareParam, ShareResource, DeviceState, TargetContainer>
{
    /// Create a new single device virtual compute node system
    pub fn new(
        node: VirtualComputeNode<ShareParam, DeviceState>,
        resource_template: ShareResource,
        container: TargetContainer,
        latency_predictor: impl Fn(&str, &ShareParam, u32) -> f32 + Send + Sync + 'static,
        model_request_rate_predictor: impl Fn() -> Pin<
            Box<dyn Future<Output = HashMap<String, f32>> + Send>,
        > + Send
        + Sync
        + 'static,
        batch_size_range: RangeInclusive<u32>,
    ) -> Self {
        Self {
            node,
            resource_template,
            container,
            latency_predictor: Arc::new(latency_predictor),
            model_request_rate_predictor: Arc::new(model_request_rate_predictor),
            batch_size_range,
        }
    }

    /// Compute the work task on the virtual compute node system
    pub async fn compute(&self, work: WorkTask<Input>) -> Result<WorkTask<Output>, String>
    where
        ShareParam: Clone + Eq + std::hash::Hash + Ord,
    {
        ComputeNode::compute(&self.node, work).await
    }

    /// Perform maintenance on the virtual compute node system
    /// This method is used to perform any necessary maintenance tasks on the compute node,
    /// such as cleaning up resources or updating the state of the node.
    /// It is typically called periodically to ensure the node remains in a healthy state.
    pub fn perform_maintenance(&self)
    where
        ShareParam: Eq + std::hash::Hash + Ord,
    {
        self.node.perform_maintenance();
    }
}

impl<
    ShareParam: ShareTarget
        + Serialize
        + DeserializeOwned
        + Debug
        + Encode
        + Decode<()>
        + Send
        + Sync
        + 'static,
    ShareResource: ShareableComputeDeviceResource + for<'a> From<&'a ShareParam> + Clone,
    DeviceState: base::DeviceState + Encode + Decode<()> + Send + Sync + 'static,
    TargetContainer: ChoosableShareTargetContainer<Target = ShareParam>,
> SchedulableSystem
    for SingleDeviceVirtualComputeNodeSystem<
        ShareParam,
        ShareResource,
        DeviceState,
        TargetContainer,
    >
where
    ShareParam: Eq + std::hash::Hash + Ord,
    TargetContainer: Clone + Send + Sync,
{
    type Command = Command<ShareParam>;

    type Context = Context<ShareParam, ShareResource, TargetContainer>;

    type Error = String;

    async fn query_context(&self) -> Self::Context {
        let (state, status) = try_join!(
            <VirtualComputeNode<ShareParam, DeviceState> as ComputeNode>::state(&self.node),
            <VirtualComputeNode<ShareParam, DeviceState> as ComputeNode>::status(&self.node)
        )
        .unwrap();
        Context {
            state,
            status,
            container: self.container.clone(),
            max_resource: self.resource_template,
            latency_predictor: self.latency_predictor.clone(),
            predictor: self.model_request_rate_predictor.clone(),
            batch_size_range: self.batch_size_range.clone(),
        }
    }

    fn generate_command(
        _current: &Self::Context,
        target: <Self::Context as SystemContext>::State,
    ) -> impl IntoIterator<Item = Self::Command, IntoIter: Send> + Send {
        iter::once(Command::ChangeState {
            target_state: target,
        })
    }

    async fn execute_commands<I: IntoIterator<Item = Self::Command> + Send>(
        &self,
        commands: I,
    ) -> Result<(), Self::Error>
    where
        <I as IntoIterator>::IntoIter: Send,
    {
        for command in commands {
            <VirtualComputeNode<ShareParam, DeviceState> as ComputeNode>::handle(
                &self.node, command,
            )
            .await?;
        }

        Ok(())
    }
}

/// A wrapper around a share target that implements [BuddyCellShareTargetWrapper] using a latency predictor function.
#[derive(Clone)]
pub struct BuddyCellShareTargetWrapper<T: Clone + Ord> {
    target: T,
    latency_predictor: InferenceLatencyPredictor<T>,
    resource_usage_rate: f32,
}

impl<T: Debug + Clone + Ord> Debug for BuddyCellShareTargetWrapper<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BuddyCellShareTargetWrapper")
            .field("target", &self.target)
            .field("resource_usage_rate", &self.resource_usage_rate)
            .finish()
    }
}

impl<T: Debug + Clone + Ord> RankableShareTarget for BuddyCellShareTargetWrapper<T> {
    fn evaluate_latency(&self, model_id: &str, batch_size: u32) -> f32 {
        (self.latency_predictor)(model_id, &self.target, batch_size)
    }

    fn resource_usage_rate(&self) -> f32 {
        self.resource_usage_rate
    }
}

// Manual impls for Eq, Ord, etc. to delegate to the inner target
impl<T: Clone + Ord> PartialEq for BuddyCellShareTargetWrapper<T> {
    fn eq(&self, other: &Self) -> bool {
        self.target == other.target
    }
}
impl<T: Clone + Ord> Eq for BuddyCellShareTargetWrapper<T> {}
impl<T: Clone + Ord> PartialOrd for BuddyCellShareTargetWrapper<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<T: Clone + Ord> Ord for BuddyCellShareTargetWrapper<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.target.cmp(&other.target)
    }
}

struct AdaptableBuddyCellAllocator<R: BuddyCellResource<InnerResource: Debug>> {
    inner: BuddyCellAllocator<R>,
    latency_predictor: InferenceLatencyPredictor<R::ChoosableShareTarget>,
}

impl<R: BuddyCellResource<InnerResource: Debug>> Debug for AdaptableBuddyCellAllocator<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdaptableBuddyCellAllocator")
            .field("inner", &self.inner)
            .finish()
    }
}

impl<R: BuddyCellResource<InnerResource: Debug + Sized>> AdaptableBuddyCellAllocator<R> {
    fn new(
        inner: BuddyCellAllocator<R>,
        latency_predictor: InferenceLatencyPredictor<R::ChoosableShareTarget>,
    ) -> Self {
        Self {
            inner,
            latency_predictor,
        }
    }
}

impl<R: BuddyCellResource<InnerResource: Debug>> ResourceAllocator
    for AdaptableBuddyCellAllocator<R>
where
    R::ChoosableShareTarget: Ord + Clone,
{
    type Target = BuddyCellShareTargetWrapper<R::ChoosableShareTarget>;

    fn choosable_share_targets(&self) -> Vec<Self::Target> {
        self.inner
            .choosable_share_targets()
            .into_iter()
            .map(|target| {
                let resource = target.clone().into();
                BuddyCellShareTargetWrapper {
                    target,
                    latency_predictor: self.latency_predictor.clone(),
                    resource_usage_rate: resource.limit_used_rate(self.inner.limit()),
                }
            })
            .collect()
    }

    fn share(&mut self, target: Self::Target) -> Option<()> {
        self.inner.share(target.target)?;

        Some(())
    }

    fn refund(&mut self, target: Self::Target) -> bool {
        self.inner.refund(target.target)
    }

    fn min_targets(&self) -> Vec<Self::Target> {
        self.inner
            .min_targets()
            .iter()
            .cloned()
            .map(|target| {
                let resource = target.clone().into();
                BuddyCellShareTargetWrapper {
                    target,
                    latency_predictor: self.latency_predictor.clone(),
                    resource_usage_rate: resource.limit_used_rate(self.inner.limit()),
                }
            })
            .collect()
    }

    fn can_share(&self, target: Self::Target) -> bool {
        self.inner.can_share(target.target)
    }
}

impl<
    ShareParam: ShareTarget
        + Serialize
        + DeserializeOwned
        + Debug
        + Encode
        + Decode<()>
        + Send
        + Sync
        + Ord
        + 'static,
    ShareResource: ShareableComputeDeviceResource + for<'a> From<&'a ShareParam> + Clone,
    TargetContainer: ChoosableShareTargetContainer<Target = ShareParam>,
> Scheduler<Context<ShareParam, ShareResource, TargetContainer>> for GreedyScheduler
where
    ShareResource: BuddyCellResourceImplHelper<ShareParam = ShareParam>,
{
    type Error = String;

    async fn schedule(
        &self,
        context: impl AsRef<Context<ShareParam, ShareResource, TargetContainer>>,
    ) -> Result<
        <Context<ShareParam, ShareResource, TargetContainer> as SystemContext>::State,
        Self::Error,
    > {
        let context = context.as_ref();

        let gpu_num = context.state().slices.len();
        let resources = iter::repeat_n(context.max_resource, gpu_num)
            .map(|res| ConstrainedResource::<ShareResource>::new(res, res))
            .collect::<Vec<_>>();
        let container = &context.container;
        let buddy_cell_allocator = BuddyCellAllocator::new(
            resources,
            container.all_cases(),
            container.min(),
            context.max_resource,
        );

        #[cfg(feature = "tracing")]
        info!("Buddy cell allocator created with {gpu_num} GPUs");

        let adaptable_allocator = AdaptableBuddyCellAllocator::new(
            buddy_cell_allocator,
            context.latency_predictor.clone(),
        );

        let model_request_rate = (context.predictor)().await;

        #[cfg(feature = "tracing")]
        info!("Model request rate predicted: {model_request_rate:?}");
        let model_allocation = GreedyScheduler::try_schedule(
            adaptable_allocator,
            model_request_rate,
            context.batch_size_range.clone(),
        )?;
        #[cfg(feature = "tracing")]
        info!("Model allocation: {model_allocation:?}");
        let targets = model_allocation
            .values()
            .flatten()
            .map(|rankable_target| rankable_target.target.clone())
            .collect::<Vec<_>>();
        let grouped_targets =
            regroup_buddy_cell::<ConstrainedResource<ShareResource>>(targets, context.max_resource);

        let model_allocation = model_allocation
            .into_iter()
            .map(|(model_name, resource)| {
                (
                    model_name,
                    resource
                        .into_iter()
                        .map(|rankable_target| rankable_target.target)
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<HashMap<_, _>>();

        Ok(State {
            model_allocation,
            slices: grouped_targets,
        })
    }
}
