use std::{collections::HashMap, path::PathBuf, sync::Arc, time::Duration};

use compute_node::{
    compute_device::{
        mock::nv::MockNvidiaMigShareableGpu,
        vgpu::resource_spec::nv::mig::{A100Template, NvidiaMigResource, VDeviceScale},
    },
    compute_node::{
        mock::MockVirtualGpuWorkerProxy,
        r#virtual::{
            base::BaseVirtualComputeNode,
            virtual_compute_node::{
                Command, VirtualComputeNode,
                components::request_scheduler::batch_size_extractor::BatchExtractorStrategy,
            },
        },
    },
};
use divan::Bencher;
use scheduler::{
    SchedulableSystem, Scheduler,
    daemon::{
        config_builder::HeterogeneousSystemBuilder,
        predictor::load_latency_csv,
        spec::{DaemonSpec, nvidia_spec::MockNvidiaMigDaemon},
        types::Config,
    },
    schedule_system::{
        schedulers::greedy::GreedyScheduler,
        virtual_compute_node::system::virtual_compute_node::{
            Context, SingleDeviceVirtualComputeNodeSystem,
        },
    },
};
use uuid::Uuid;
use worker::mock::MockWorker;

#[divan::bench(args = [1, 2, 4, 8, 16, 32, 64, 96, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],)]
fn bench_heterogeneous_scheduling(bencher: Bencher, device_num: usize) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_time()
        .build()
        .unwrap();
    let latency_path: PathBuf = "/mnt/3T_disk2/zzj/HSGS/model_latency.csv".into();
    let base_compute_node = BaseVirtualComputeNode::<
        MockWorker,
        MockNvidiaMigShareableGpu,
        MockVirtualGpuWorkerProxy,
    >::new(
        0..=device_num,
        latency_path.clone(),
        latency_path.clone(),
        latency_path.clone(),
    );

    let model_request_rate = HashMap::from([
        ("alexnet".to_string(), 0.0f32),
        ("efficientnet".to_string(), 0.0f32),
        ("mnasnet".to_string(), 0.0f32),
        ("mobilenet".to_string(), 0.0f32),
        ("resnet18".to_string(), 0.0f32),
        ("resnet50".to_string(), 0.0f32),
        ("vgg16".to_string(), 0.0f32),
        ("vgg19".to_string(), 0.0f32),
    ]);
    let (compute_node, data) = runtime.block_on(async {
        let mut compute_node = VirtualComputeNode::new(
            BatchExtractorStrategy::timeout_split_and_satisfy(Duration::from_millis(8)),
            |_, _, _| 0.0f32,
        );

        compute_node
            .add_virtual_compute_node(Uuid::new_v4(), base_compute_node.into_dyn())
            .await
            .unwrap();
        (compute_node, load_latency_csv(&latency_path).await.unwrap())
    });
    let latency_predictor = move |model: &str, resource: &VDeviceScale, batch_size: u32| -> f32 {
        let resource_key = resource.name();
        data.get(model)
            .and_then(|model_data| model_data.get(resource_key))
            .and_then(|batch_data| {
                if let Some(latency) = batch_data.get(&batch_size).copied() {
                    return Some(latency);
                }

                let lower = batch_data
                    .iter()
                    .filter(|&(&k, _)| k < batch_size)
                    .max_by_key(|&(&k, _)| k);
                let upper = batch_data
                    .iter()
                    .filter(|&(&k, _)| k > batch_size)
                    .min_by_key(|&(&k, _)| k);

                if let (Some((&x1, &y1)), Some((&x2, &y2))) = (lower, upper) {
                    let x = batch_size;
                    let interpolated_y = y1 + (x - x1) as f32 * (y2 - y1) / (x2 - x1) as f32;
                    Some(interpolated_y)
                } else {
                    // Fallback to closest
                    batch_data
                        .iter()
                        .min_by_key(|(k, _)| (**k as i32 - batch_size as i32).abs())
                        .map(|(_, v)| *v)
                }
            })
            .unwrap_or(100.0)
    };
    let system = SingleDeviceVirtualComputeNodeSystem::new(
        compute_node,
        NvidiaMigResource { ce: 7, memory: 80 },
        VDeviceScale::A100(A100Template::Huge),
        latency_predictor,
        move || {
            let rate = model_request_rate.clone();
            Box::pin(async move { rate })
        },
        1u32..=32u32,
    );
    let context = runtime.block_on(async { system.query_context().await });
    let scheduler = GreedyScheduler;

    bencher.bench(|| runtime.block_on(async { scheduler.schedule(&context).await }))
}

fn main() {
    // Run registered benchmarks.
    divan::main();
}
