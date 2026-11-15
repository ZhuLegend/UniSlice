//! Defines the `BidAllocator` responsible for managing bids from devices.

use std::collections::BTreeMap;

use uuid::Uuid;

use crate::schedule_system::{
    allocator::ResourceAllocator,
    virtual_compute_node::system::heterogeneous::{bid::Bid, device::ShareableResourceAllocator},
};

/// The allocator for the heterogeneous system.
#[derive(Default, Debug)]
pub struct BidAllocator {
    devices: BTreeMap<Uuid, Box<dyn ShareableResourceAllocator<Target = Bid>>>,
}

impl BidAllocator {
    /// Creates a new `BidAllocator`.
    pub fn new() -> Self {
        Default::default()
    }
}

impl FromIterator<Box<dyn ShareableResourceAllocator<Target = Bid>>> for BidAllocator {
    fn from_iter<T: IntoIterator<Item = Box<dyn ShareableResourceAllocator<Target = Bid>>>>(
        iter: T,
    ) -> Self {
        let devices = iter
            .into_iter()
            .filter_map(|d| {
                let min_targets = d.min_targets();
                if min_targets.is_empty() {
                    None
                } else {
                    let id = min_targets.first().unwrap().device_id();
                    assert!(
                        min_targets.iter().all(|bid| bid.device_id() == id),
                        "All bids in min_targets must have the same device_id (expected: {:?}); \
                        found mismatched device_ids in allocator.",
                        id
                    );
                    Some((id, d))
                }
            })
            .collect();

        BidAllocator { devices }
    }
}

impl ResourceAllocator for BidAllocator {
    type Target = Bid;

    fn choosable_share_targets(&self) -> Vec<Self::Target> {
        self.devices
            .values()
            .flat_map(|device| device.available_targets())
            .collect()
    }

    fn share(&mut self, target: Self::Target) -> Option<()> {
        let device = self.devices.get_mut(&target.device_id())?;
        if device.share(&target) {
            return Some(());
        }
        None
    }

    fn refund(&mut self, target: Self::Target) -> bool {
        if let Some(device) = self.devices.get_mut(&target.device_id()) {
            device.refund(&target)
        } else {
            false
        }
    }

    fn min_targets(&self) -> Vec<Self::Target> {
        self.devices
            .values()
            .flat_map(|device| device.min_targets())
            .collect()
    }

    fn can_share(&self, target: Self::Target) -> bool {
        if let Some(device) = self.devices.get(&target.device_id()) {
            device
                .available_targets()
                .iter()
                .any(|t| t.target_id() == target.target_id())
        } else {
            false
        }
    }
}
