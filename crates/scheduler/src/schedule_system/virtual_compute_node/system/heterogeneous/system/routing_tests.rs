use std::{
    pin::Pin,
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use compute_node::compute_node::r#virtual::{
    BaseWork,
    virtual_compute_node::components::{
        request_scheduler::{
            PendingRequests, PoolRequestQueue, RequestScheduler,
            batch_size_extractor::BatchExtractorStrategy,
        },
        worker_manager::WorkerRequest,
        worker_registry::{WorkerInfo, WorkerRegistry, WorkerStatus},
    },
};

use super::*;
use crate::schedule_system::virtual_compute_node::system::heterogeneous::device::ShareableResourceAllocator;

struct QueueDevice {
    id: Uuid,
    scheduler: Arc<RequestScheduler>,
    registry:
        WorkerRegistry<compute_node::compute_device::vgpu::resource_spec::nv::mig::VDeviceScale>,
}
impl BoxedShareableDevice for QueueDevice {
    type Target = Bid;
    fn id(&self) -> Uuid {
        self.id
    }
    fn request_queue(&self) -> Option<PoolRequestQueue> {
        Some(self.scheduler.pool_queue())
    }
    fn accept_pending(&self, pending: PendingRequests, epoch: u64) -> Result<(), PendingRequests> {
        self.scheduler.accept_pending(pending, epoch)
    }
    fn boxed_allocator(&self) -> Box<dyn ShareableResourceAllocator<Target = Bid>> {
        unimplemented!()
    }
    fn apply(
        &self,
        _: HashMap<String, Vec<Bid>>,
    ) -> Pin<Box<dyn Future<Output = Result<(), String>> + Send + '_>> {
        unimplemented!()
    }
    fn support_models(&self) -> Pin<Box<dyn Future<Output = Vec<String>> + Send + '_>> {
        Box::pin(async { self.scheduler.pool_queue().serving_models() })
    }
    fn compute(
        &self,
        work: WorkTask<Input>,
    ) -> Pin<Box<dyn Future<Output = Result<WorkTask<Output>, String>> + Send + '_>> {
        Box::pin(async {
            Ok(WorkTask {
                model: work.model.clone(),
                work: self
                    .scheduler
                    .add_compute_task(work.model, work.work)
                    .await?,
            })
        })
    }
    fn perform_latency_update(&self) {}
}
fn queue_fixture() -> (
    HeterogeneousSystem,
    Arc<QueueDevice>,
    Arc<QueueDevice>,
    Uuid,
    Uuid,
) {
    let device = || {
        let scheduler = Arc::new(RequestScheduler::with_strategy(
            BatchExtractorStrategy::split_and_satisfy(),
        ));
        let registry = WorkerRegistry::with_request_queue(scheduler.pool_queue());
        Arc::new(QueueDevice {
            id: Uuid::new_v4(),
            scheduler,
            registry,
        })
    };
    let (a, b) = (device(), device());
    let (wa, wb) = (Uuid::new_v4(), Uuid::new_v4());
    for (device, worker_id, status) in [
        (&a, wa, WorkerStatus::Running),
        (&b, wb, WorkerStatus::Paused),
    ] {
        assert!(device.registry.register_worker(WorkerInfo {
            worker_id,
            model_name: "m".into(),
            node_id: device.id,
            device_id: device.id,
            device_params:
                compute_node::compute_device::vgpu::resource_spec::nv::mig::VDeviceScale::A100(
                    compute_node::compute_device::vgpu::resource_spec::nv::mig::A100Template::Tiny
                ),
            created_at: Instant::now(),
            status,
        }));
    }
    let devices: HashMap<_, Arc<dyn BoxedShareableDevice<Target = Bid>>> = HashMap::from([
        (
            a.id,
            a.clone() as Arc<dyn BoxedShareableDevice<Target = Bid>>,
        ),
        (
            b.id,
            b.clone() as Arc<dyn BoxedShareableDevice<Target = Bid>>,
        ),
    ]);
    (
        HeterogeneousSystem::new(
            devices,
            Arc::new(|| Box::pin(async { HashMap::new() })),
            1..=32,
        ),
        a,
        b,
        wa,
        wb,
    )
}
fn work() -> WorkTask<Input> {
    WorkTask {
        model: "m".into(),
        work: BaseWork::new(Uuid::new_v4(), ndarray::Array2::<f32>::zeros((1, 1)).into()),
    }
}
async fn wait_queued(device: &QueueDevice) {
    tokio::time::timeout(Duration::from_secs(2), async {
        while device.scheduler.get_model_queue_batch_size(&"m".into()) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}
#[tokio::test]
async fn coordinator_moves_old_requests_after_a_global_gap_and_wakes_new_arrivals() {
    let (system, a, b, wa, wb) = queue_fixture();
    let system = Arc::new(system);
    let request = work();
    let id = request.work.id();
    let first = {
        let system = system.clone();
        tokio::spawn(async move { system.compute(request).await })
    };
    wait_queued(&a).await;
    a.registry.update_worker_status(wa, WorkerStatus::Paused);
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(!first.is_finished());
    assert!(system.get_supporting_devices("m").is_empty());
    let second = {
        let system = system.clone();
        tokio::spawn(async move { system.compute(work()).await })
    };
    b.registry.update_worker_status(wb, WorkerStatus::Running);
    tokio::time::timeout(Duration::from_secs(2), async {
        while b.scheduler.get_model_queue_batch_size(&"m".into()) < 2 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let (resp, rx) = tokio::sync::oneshot::channel();
    b.scheduler
        .request_batch(WorkerRequest {
            worker_id: wb,
            model_name: "m".into(),
            desired_batch: 2,
            resp,
        })
        .await
        .unwrap();
    for (work, tx) in rx.await.unwrap() {
        let bytes = bincode::encode_to_vec(&work, bincode::config::standard()).unwrap();
        let (output, _): (BaseWork<Output>, _) =
            bincode::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
        tx.send(output).unwrap();
    }
    assert_eq!(first.await.unwrap().unwrap().work.id(), id);
    second.await.unwrap().unwrap();
    assert_eq!(a.scheduler.get_model_queue_batch_size(&"m".into()), 0);
}
#[tokio::test]
async fn unavailable_model_has_a_finite_deadline() {
    let (system, a, _, wa, _) = queue_fixture();
    a.registry.update_worker_status(wa, WorkerStatus::Paused);
    let system = system.with_request_timeout(Duration::from_millis(30));
    let error = system.compute(work()).await.unwrap_err();
    assert!(error.contains("deadline exceeded"));
    assert_eq!(a.scheduler.get_model_queue_batch_size(&"m".into()), 0);
}

#[derive(Default)]
struct Device {
    id: Uuid,
    models: Mutex<Vec<String>>,
    fail: AtomicBool,
}

impl BoxedShareableDevice for Device {
    type Target = Bid;
    fn id(&self) -> Uuid {
        self.id
    }
    fn boxed_allocator(&self) -> Box<dyn ShareableResourceAllocator<Target = Bid>> {
        unimplemented!()
    }
    fn apply(
        &self,
        state: HashMap<String, Vec<Bid>>,
    ) -> Pin<Box<dyn Future<Output = Result<(), String>> + Send + '_>> {
        Box::pin(async move {
            // Deliberately allow partial mutation before failure.
            *self.models.lock().unwrap() = state
                .into_iter()
                .filter_map(|(model, bids)| (!bids.is_empty()).then_some(model))
                .collect();
            if self.fail.load(Ordering::SeqCst) {
                Err("injected apply failure".into())
            } else {
                Ok(())
            }
        })
    }
    fn support_models(&self) -> Pin<Box<dyn Future<Output = Vec<String>> + Send + '_>> {
        Box::pin(async { self.models.lock().unwrap().clone() })
    }
    fn compute(
        &self,
        _: WorkTask<Input>,
    ) -> Pin<Box<dyn Future<Output = Result<WorkTask<Output>, String>> + Send + '_>> {
        unimplemented!()
    }
    fn perform_latency_update(&self) {}
}

fn fixture() -> (HeterogeneousSystem, Arc<Device>, Arc<Device>) {
    let a = Arc::new(Device {
        id: Uuid::new_v4(),
        ..Default::default()
    });
    let b = Arc::new(Device {
        id: Uuid::new_v4(),
        ..Default::default()
    });
    let devices: HashMap<_, Arc<dyn BoxedShareableDevice<Target = Bid>>> = HashMap::from([
        (
            a.id,
            a.clone() as Arc<dyn BoxedShareableDevice<Target = Bid>>,
        ),
        (
            b.id,
            b.clone() as Arc<dyn BoxedShareableDevice<Target = Bid>>,
        ),
    ]);
    (
        HeterogeneousSystem::new(
            devices,
            Arc::new(|| Box::pin(async { HashMap::new() })),
            1..=32,
        ),
        a,
        b,
    )
}

fn allocation(
    device: &Device,
    model: &str,
    throughput: f32,
    copies: usize,
) -> (Uuid, HashMap<String, Vec<Bid>>) {
    let bid = Bid::new(
        device.id,
        "same-profile".into(),
        Arc::new(()),
        Arc::new(move |_, batch| batch as f32 / throughput),
        0.25,
    );
    (
        device.id,
        HashMap::from([(model.into(), vec![bid; copies])]),
    )
}

#[tokio::test]
async fn apply_keeps_profile_weights_across_cache_and_latency_updates() {
    let (system, a, b) = fixture();
    system
        .execute_commands([HashMap::from([
            allocation(&a, "m", 100.0, 2),
            allocation(&b, "m", 100.0, 1),
        ])])
        .await
        .unwrap();
    system.refresh_all_device_cache().await;
    system.perform_latency_update().await;
    system.execute_commands([HashMap::new()]).await.unwrap();
    let weights = system.get_normalized_weights("m");
    assert!((weights[&a.id] - 2.0 / 3.0).abs() < 1e-6);
    assert!((weights[&b.id] - 1.0 / 3.0).abs() < 1e-6);
    assert_eq!(system.get_supporting_devices("m").len(), 2);
}

#[cfg(feature = "scheduler_daemon")]
#[tokio::test]
async fn execution_feedback_updates_allocation_scores_and_periodic_routing_weights() {
    use compute_node::compute_device::vgpu::shareable::test::TestShareTarget;
    let (system, a, b) = fixture();
    let base = Arc::new(|_: &str, _: &TestShareTarget, batch: u32| 0.12 * batch as f32);
    let (predict, observe) = crate::daemon::predictor::with_online_correction(a.id, base);
    let profile = TestShareTarget::G7Gb80;
    let bid = Bid::new(
        a.id,
        "slice".into(),
        Arc::new(()),
        Arc::new(move |model, batch| predict(model, &profile, batch)),
        0.25,
    );
    system
        .execute_commands([HashMap::from([
            (a.id, HashMap::from([("m".into(), vec![bid.clone()])])),
            allocation(&b, "m", 1.0 / 0.12, 1),
        ])])
        .await
        .unwrap();
    assert!((system.get_normalized_weights("m")[&a.id] - 0.5).abs() < 1e-6);
    for _ in 0..10 {
        observe("m", &profile, 8, 0.8);
    }
    let observed = system.get_applied_bids("m");
    assert!((observed[&a.id][0].evaluate_latency("m", 8) - 0.8).abs() < 0.005);
    assert!((system.get_normalized_weights("m")[&a.id] - 0.5).abs() < 1e-6);
    // The same bid used by allocation sees the learned duration without reapplying a plan.
    assert!((bid.evaluate_latency("m", 8) - 0.8).abs() < 0.005);
    system
        .device_ema_latency
        .entry(a.id)
        .or_default()
        .insert("m".into(), 99.0);
    system.perform_latency_update().await;
    let capacity = 8.0 / bid.evaluate_latency("m", 8);
    let expected = capacity / (capacity + 1.0 / 0.12);
    assert!((system.get_normalized_weights("m")[&a.id] - expected).abs() < 1e-6);
    assert!(expected > 0.54);
    system
        .execute_commands([HashMap::from([(a.id, HashMap::new())])])
        .await
        .unwrap();
    system.perform_latency_update().await;
    assert_eq!(
        system.get_normalized_weights("m"),
        HashMap::from([(b.id, 1.0)])
    );
}

#[tokio::test]
async fn equal_profile_names_use_each_devices_current_predictor() {
    let (system, a, b) = fixture();
    for speed in [200.0, 400.0] {
        system
            .execute_commands([HashMap::from([
                allocation(&a, "m", speed, 1),
                allocation(&b, "m", 100.0, 1),
            ])])
            .await
            .unwrap();
        assert!((system.get_normalized_weights("m")[&a.id] - speed / (speed + 100.0)).abs() < 1e-6);
    }
}

#[tokio::test]
async fn reallocation_removes_old_models_and_preserves_untouched_devices() {
    let (system, a, b) = fixture();
    system
        .execute_commands([HashMap::from([
            allocation(&a, "m", 200.0, 1),
            allocation(&b, "m", 100.0, 1),
        ])])
        .await
        .unwrap();
    system
        .execute_commands([HashMap::from([allocation(&a, "n", 200.0, 1)])])
        .await
        .unwrap();
    assert_eq!(
        system.get_normalized_weights("m"),
        HashMap::from([(b.id, 1.0)])
    );
    assert_eq!(
        system.get_normalized_weights("n"),
        HashMap::from([(a.id, 1.0)])
    );
    system
        .execute_commands([HashMap::from([(a.id, HashMap::new())])])
        .await
        .unwrap();
    assert!(system.get_normalized_weights("n").is_empty());
    assert!(system.get_supporting_devices("n").is_empty());
}

#[tokio::test]
async fn partial_failure_does_not_publish_proposed_or_stale_weights() {
    let (system, a, b) = fixture();
    system
        .execute_commands([HashMap::from([allocation(&a, "m", 200.0, 1)])])
        .await
        .unwrap();
    a.fail.store(true, Ordering::SeqCst);
    assert!(
        system
            .execute_commands([HashMap::from([
                allocation(&a, "m", 400.0, 1),
                allocation(&b, "m", 100.0, 1),
            ])])
            .await
            .is_err()
    );
    system.perform_latency_update().await;
    assert_eq!(
        system.get_normalized_weights("m"),
        HashMap::from([(b.id, 1.0)])
    );
    assert_eq!(system.get_supporting_devices("m").len(), 2);
    assert_eq!(system.select_device_by_weight("m", &[a.id]), Some(a.id));
}

#[tokio::test]
async fn refresh_prunes_models_no_longer_supported_and_unknown_device_errors() {
    let (system, a, _) = fixture();
    system
        .execute_commands([HashMap::from([allocation(&a, "m", 200.0, 1)])])
        .await
        .unwrap();
    a.models.lock().unwrap().clear();
    system.refresh_all_device_cache().await;
    system.perform_latency_update().await;
    assert!(system.get_normalized_weights("m").is_empty());
    assert!(
        system
            .execute_commands([HashMap::from([(Uuid::new_v4(), HashMap::new())])])
            .await
            .is_err()
    );
}
