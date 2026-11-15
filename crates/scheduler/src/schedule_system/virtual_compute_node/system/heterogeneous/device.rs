//! Defines the `ShareableResourceAllocator` trait for heterogeneous resources.

use std::{
    any::{Any, TypeId},
    collections::HashMap,
    fmt::Debug,
    iter,
    pin::Pin,
    sync::Arc,
};

use bincode::{Decode, Encode};
use compute_node::{
    ComputeNode,
    compute_device::vgpu::{
        ShareableComputeDeviceResource,
        share_target::{ChoosableShareTargetContainer, ShareTarget},
    },
    compute_node::r#virtual::{
        base,
        virtual_compute_node::{
            Command::ChangeState,
            State, VirtualComputeNode, WorkTask,
            components::request_scheduler::{PendingRequests, PoolRequestQueue},
        },
    },
    util::{Input, Output},
};
use serde::{Serialize, de::DeserializeOwned};
use uuid::Uuid;

use crate::schedule_system::{
    allocator::RankableShareTarget,
    virtual_compute_node::{
        algorithms::buddy_cell::{
            BuddyCellAllocator, BuddyCellResource, CanExceedLimit,
            constrained::{BuddyCellResourceImplHelper, ConstrainedResource},
            regroup_buddy_cell,
        },
        system::{InferenceLatencyPredictor, ResourceIdentifier, heterogeneous::bid::Bid},
    },
};

/// A trait representing a device that can share its compute resource.
pub trait ShareableDevice {
    /// Live queue admission for managed pools; absent for other device adapters.
    fn request_queue(&self) -> Option<PoolRequestQueue> {
        None
    }
    /// Move unclaimed fragments into this pool, or return ownership unchanged.
    fn accept_pending(&self, pending: PendingRequests, _epoch: u64) -> Result<(), PendingRequests> {
        Err(pending)
    }
    /// The type of target used by [ShareableResourceAllocator].
    type Allocator: ShareableResourceAllocator;

    /// Returns the unique identifier of the device.
    fn id(&self) -> Uuid;

    /// Returns an allocator which can represent the shareable resources of the device.
    fn allocator(&self) -> Self::Allocator;

    /// Apply the given state to the device.
    ///
    /// state: A mapping from model names to their allocated targets.
    /// Returns a Future that resolves to a Result indicating success or failure.
    fn apply(
        &self,
        state: HashMap<String, Vec<<Self::Allocator as ShareableResourceAllocator>::Target>>,
    ) -> Pin<Box<dyn Future<Output = Result<(), String>> + Send + '_>>;

    /// Returns a list of supported models by the device.
    fn support_models(&self) -> Pin<Box<dyn Future<Output = Vec<String>> + Send + '_>>;

    /// Computes the output based on the input work.
    fn compute(
        &self,
        input: WorkTask<Input>,
    ) -> Pin<Box<dyn Future<Output = Result<WorkTask<Output>, String>> + Send + '_>>;

    /// Perform latency update for the device.
    /// This function can be used to refresh or recalculate latency metrics.
    fn perform_latency_update(&self);
}

/// A Boxed version of [ShareableDevice] for dynamic dispatch.
pub trait BoxedShareableDevice: Send + Sync {
    /// Live queue admission for managed pools.
    fn request_queue(&self) -> Option<PoolRequestQueue> {
        None
    }
    /// Move unclaimed fragments only while the target generation is serving.
    fn accept_pending(&self, pending: PendingRequests, _epoch: u64) -> Result<(), PendingRequests> {
        Err(pending)
    }
    /// The type of target used by [ShareableResourceAllocator].
    type Target: RankableShareTarget + Send;

    /// Returns the unique identifier of the device.
    fn id(&self) -> Uuid;

    /// Returns a boxed allocator for the device.
    fn boxed_allocator(&self) -> Box<dyn ShareableResourceAllocator<Target = Self::Target>>;

    /// Apply the given state to the device.
    fn apply(
        &self,
        state: HashMap<String, Vec<Self::Target>>,
    ) -> Pin<Box<dyn Future<Output = Result<(), String>> + Send + '_>>;

    /// Returns a list of supported models by the device.
    fn support_models(&self) -> Pin<Box<dyn Future<Output = Vec<String>> + Send + '_>>;

    /// Computes the output based on the input work.
    fn compute(
        &self,
        input: WorkTask<Input>,
    ) -> Pin<Box<dyn Future<Output = Result<WorkTask<Output>, String>> + Send + '_>>;

    /// Perform latency update for the device.
    fn perform_latency_update(&self);
}

impl<Device: ShareableDevice + Send + Sync> BoxedShareableDevice for Device
where
    Device::Allocator: 'static,
{
    type Target = <Device::Allocator as ShareableResourceAllocator>::Target;

    fn request_queue(&self) -> Option<PoolRequestQueue> {
        ShareableDevice::request_queue(self)
    }
    fn accept_pending(&self, pending: PendingRequests, epoch: u64) -> Result<(), PendingRequests> {
        ShareableDevice::accept_pending(self, pending, epoch)
    }

    fn id(&self) -> Uuid {
        ShareableDevice::id(self)
    }

    fn boxed_allocator(
        &self,
    ) -> Box<
        dyn ShareableResourceAllocator<
            Target = <Device::Allocator as ShareableResourceAllocator>::Target,
        >,
    > {
        Box::new(self.allocator())
    }

    fn apply(
        &self,
        state: HashMap<String, Vec<Self::Target>>,
    ) -> Pin<Box<dyn Future<Output = Result<(), String>> + Send + '_>> {
        ShareableDevice::apply(self, state)
    }

    fn support_models(&self) -> Pin<Box<dyn Future<Output = Vec<String>> + Send + '_>> {
        ShareableDevice::support_models(self)
    }

    fn compute(
        &self,
        input: WorkTask<Input>,
    ) -> Pin<Box<dyn Future<Output = Result<WorkTask<Output>, String>> + Send + '_>> {
        ShareableDevice::compute(self, input)
    }

    fn perform_latency_update(&self) {
        ShareableDevice::perform_latency_update(self)
    }
}

/// A trait representing a device that can be shared among multiple models.
pub trait ShareableResourceAllocator: Send + Debug {
    /// The type of target that can be shared.
    type Target: RankableShareTarget + Send;
    /// Returns a list of available targets that can be shared.
    fn available_targets(&self) -> Vec<Self::Target>;
    /// Shares a target.
    fn share(&mut self, target: &Self::Target) -> bool;
    /// Refunds a target.
    fn refund(&mut self, target: &Self::Target) -> bool;
    /// Returns the minimum possible allocation units.
    fn min_targets(&self) -> Vec<Self::Target>;
}

/// An adapter that allows a `BuddyCellAllocator` to be used as a `ShareableResourceAllocator`.
pub struct BuddyCellAllocatorAdapter<Resource: BuddyCellResource<InnerResource: Debug>> {
    inner: BuddyCellAllocator<Resource>,
    device_id: Uuid,
    identifier: ResourceIdentifier<Resource::ChoosableShareTarget>,
    latency_predictor: InferenceLatencyPredictor<Resource::ChoosableShareTarget>,
}

impl<Resource: BuddyCellResource<InnerResource: Debug>> Debug
    for BuddyCellAllocatorAdapter<Resource>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BuddyCellAllocatorAdapter")
            .field("device_id", &self.device_id)
            .field("inner", &self.inner)
            .finish()
    }
}

impl<
    Resource: BuddyCellResource<InnerResource: Debug, ChoosableShareTarget: Send + Sync + 'static + Sized>,
> BuddyCellAllocatorAdapter<Resource>
{
    fn get_resource<'b>(&self, target: &'b Bid) -> Option<&'b Resource::ChoosableShareTarget> {
        if target.device_id() == self.device_id {
            target.data.downcast_ref::<Resource::ChoosableShareTarget>()
        } else {
            None
        }
    }

    fn target_to_bid(&self, target: Resource::ChoosableShareTarget) -> Bid {
        let latency_predictor = self.latency_predictor.clone();
        let target_cloned = target.clone();
        let predictor = move |model_id: &str, batch_size: u32| {
            latency_predictor(model_id, &target_cloned, batch_size)
        };
        let resource = target.clone().into();

        Bid::new(
            self.device_id,
            (self.identifier)(&target),
            Arc::new(target) as Arc<dyn Any + Send + Sync>,
            Arc::new(predictor),
            resource.limit_used_rate(self.inner.limit()),
        )
    }
}

impl<Resource: BuddyCellResource<InnerResource: Debug>> ShareableResourceAllocator
    for BuddyCellAllocatorAdapter<Resource>
where
    Resource: Send + Sync,
    Resource::ChoosableShareTarget: 'static + Send + Sync,
{
    type Target = Bid;

    fn available_targets(&self) -> Vec<Self::Target> {
        self.inner
            .choosable_share_targets()
            .into_iter()
            .map(|target| self.target_to_bid(target))
            .collect()
    }

    fn share(&mut self, target: &Self::Target) -> bool {
        if let Some(resource) = self.get_resource(target) {
            self.inner.share(resource.clone()).is_some()
        } else {
            false
        }
    }

    fn refund(&mut self, target: &Self::Target) -> bool {
        if let Some(resource) = self.get_resource(target) {
            self.inner.refund(resource.clone())
        } else {
            false
        }
    }

    fn min_targets(&self) -> Vec<Self::Target> {
        self.inner
            .min_targets()
            .iter()
            .map(|target| self.target_to_bid(target.clone()))
            .collect()
    }
}

/// An adapter that allows a `VirtualComputeNode` to be used as a `ShareableDevice`.
pub struct VirtualComputeNodeAdapter<
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
    id: Uuid,
    node: VirtualComputeNode<ShareParam, DeviceState>,
    resource_template: ShareResource,
    container: TargetContainer,
    latency_predictor: InferenceLatencyPredictor<ShareParam>,
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
> VirtualComputeNodeAdapter<ShareParam, ShareResource, DeviceState, TargetContainer>
{
    /// Creates a new `VirtualComputeNodeAdapter`.
    pub fn new(
        node: VirtualComputeNode<ShareParam, DeviceState>,
        resource_template: ShareResource,
        container: TargetContainer,
        latency_predictor: InferenceLatencyPredictor<ShareParam>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            node,
            resource_template,
            container,
            latency_predictor,
        }
    }

    fn get_buddy_cell_allocator(&self) -> BuddyCellAllocator<ConstrainedResource<ShareResource>>
    where
        ShareParam: Ord,
        ShareResource: BuddyCellResourceImplHelper<ShareParam = ShareParam>,
    {
        let gpu_num = self.node.gpu_num();
        let resources = iter::repeat_n(self.resource_template, gpu_num)
            .map(|res| ConstrainedResource::<ShareResource>::new(res, res))
            .collect::<Vec<_>>();
        let container = &self.container;
        BuddyCellAllocator::new(
            resources,
            container.all_cases(),
            container.min(),
            self.resource_template,
        )
    }

    /// Use a stable identity supplied by the configuration builder.
    pub fn with_id(mut self, id: Uuid) -> Self {
        self.id = id;
        self
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
> ShareableDevice
    for VirtualComputeNodeAdapter<ShareParam, ShareResource, DeviceState, TargetContainer>
where
    ShareResource: BuddyCellResourceImplHelper<ShareParam = ShareParam>,
    ShareParam: Clone + Eq + std::hash::Hash + Ord,
    TargetContainer: Send + Sync,
{
    type Allocator = BuddyCellAllocatorAdapter<ConstrainedResource<ShareResource>>;

    fn request_queue(&self) -> Option<PoolRequestQueue> {
        Some(self.node.request_queue())
    }
    fn accept_pending(&self, pending: PendingRequests, epoch: u64) -> Result<(), PendingRequests> {
        self.node.accept_pending(pending, epoch)
    }

    fn id(&self) -> Uuid {
        self.id
    }

    fn allocator(&self) -> Self::Allocator {
        let device_id = self.id;
        let identifier = Box::new(|target: &ShareParam| {
            format!(
                "{:?}-{:?}-{:?}-{:?}",
                TypeId::of::<ShareParam>(),
                TypeId::of::<ShareResource>(),
                TypeId::of::<DeviceState>(),
                target
            )
        });
        let latency_predictor = self.latency_predictor.clone();
        let buddy_cell_allocator = self.get_buddy_cell_allocator();

        BuddyCellAllocatorAdapter {
            inner: buddy_cell_allocator,
            device_id,
            identifier,
            latency_predictor,
        }
    }

    fn apply(
        &self,
        model_allocation: HashMap<
            String,
            Vec<<Self::Allocator as ShareableResourceAllocator>::Target>,
        >,
    ) -> Pin<Box<dyn Future<Output = Result<(), String>> + Send + '_>> {
        Box::pin(async move {
            let model_allocation = model_allocation
                .into_iter()
                .map(|(model_name, bids)| {
                    (
                        model_name,
                        bids.into_iter()
                            .map(|bid| {
                                bid.data
                                    .downcast_ref::<ShareParam>()
                                    .unwrap_or_else(|| panic!("Applying invalid bid to device. Device ID: {:?}, Bid: {:?}", self.id, bid))
                                    .clone()
                            })
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<HashMap<_, _>>();
            #[cfg(feature = "tracing")]
            tracing::trace!("Applying model allocation: {:?}", model_allocation);

            let targets = model_allocation
                .values()
                .flatten()
                .cloned()
                .collect::<Vec<_>>();
            let grouped_targets = regroup_buddy_cell::<ConstrainedResource<ShareResource>>(
                targets,
                self.resource_template,
            );

            #[cfg(feature = "tracing")]
            tracing::trace!("Regrouped targets: {:?}", grouped_targets);

            self.node
                .handle(ChangeState {
                    target_state: State {
                        model_allocation,
                        slices: grouped_targets,
                    },
                })
                .await?;

            #[cfg(feature = "tracing")]
            tracing::trace!("Model allocated");

            Ok(())
        })
    }

    fn support_models(&self) -> Pin<Box<dyn Future<Output = Vec<String>> + Send + '_>> {
        Box::pin(async move {
            if let Ok(state) = self.node.state().await {
                state
                    .model_allocation
                    .into_iter()
                    .filter_map(|(model, slices)| {
                        if !slices.is_empty() {
                            Some(model)
                        } else {
                            None
                        }
                    })
                    .collect()
            } else {
                Vec::new()
            }
        })
    }

    fn compute(
        &self,
        input: WorkTask<Input>,
    ) -> Pin<Box<dyn Future<Output = Result<WorkTask<Output>, String>> + Send + '_>> {
        Box::pin(async move { self.node.compute(input).await })
    }

    fn perform_latency_update(&self) {
        self.node.perform_maintenance()
    }
}
