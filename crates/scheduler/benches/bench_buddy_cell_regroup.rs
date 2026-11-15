//! 性能基准测试：大规模资源重组

use std::collections::HashMap;

use compute_node::compute_device::vgpu::shareable::test::{TestResource, TestShareTarget};
use divan::AllocProfiler;
use rand::prelude::*;
use scheduler::schedule_system::virtual_compute_node::algorithms::buddy_cell::{
    BuddyCellAllocator, BuddyCellResource, constrained::ConstrainedResource, regroup_buddy_cell,
};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

#[divan::bench(
    args = [1, 2, 4, 8, 16, 32, 64, 96, 128, 256, 512, 1024, 2048, 4096, 8192],
)]
fn large_scale_regroup_bench(original_cluster_size: usize) {
    let mut rng = rand::rng();
    let resources = vec![
        ConstrainedResource::new(TestResource::MAX_SELF, TestResource::MAX_SELF);
        original_cluster_size
    ];
    let mut remaining_resources = BuddyCellAllocator::new(
        resources.clone(),
        TestShareTarget::ALL_CASES.to_vec(),
        TestShareTarget::MIN.to_vec(),
        TestResource::MAX_SELF,
    );
    let mut split_resources: Vec<TestShareTarget> = Vec::new();
    let available_targets = vec![
        TestShareTarget::G4Gb40,
        TestShareTarget::G3Gb40,
        TestShareTarget::G2Gb20,
        TestShareTarget::G1Gb10,
    ];
    let mut can_continue = true;
    while can_continue {
        let target = available_targets.choose(&mut rng).unwrap();
        if remaining_resources.can_share(*target) {
            if remaining_resources.share(*target).is_some() {
                split_resources.push(*target);
            }
        } else {
            can_continue = false;
            for target in &available_targets {
                if remaining_resources.can_share(*target) {
                    can_continue = true;
                    break;
                }
            }
        }
        if split_resources.len() > original_cluster_size * 7 {
            break;
        }
    }
    let regrouped = regroup_buddy_cell::<ConstrainedResource<TestResource>>(
        split_resources.clone(),
        TestResource::MAX_SELF,
    );
    let regrouped_size = regrouped.len();

    if regrouped_size > original_cluster_size {
        dbg!(regrouped, split_resources);
        panic!(
            "Regrouped size {regrouped_size} exceeds original cluster size {original_cluster_size}"
        );
    }
}

fn main() {
    // Run registered benchmarks.
    divan::main();
}

// 更多参数组合可按需添加
