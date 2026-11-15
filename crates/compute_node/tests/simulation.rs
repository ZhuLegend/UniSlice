#![cfg(all(feature = "mock", feature = "vgpu", feature = "bin_help"))]
use std::{collections::HashMap, path::PathBuf, sync::Arc, time::Duration};

use compute_node::{
    ComputeNode,
    compute_device::{
        mock::{
            MockComputeDevice, MockDeviceSettings, hw::MockHuaweiMigShareableGpu,
            nv::MockNvidiaMigShareableGpu,
        },
        vgpu::{
            ShareableComputeDevice,
            resource_spec::{
                hw::{HuaweiMigResource, VChipScale},
                nv::mig::{A100Template, NvidiaMigResource, VDeviceScale},
            },
            share_target::ChoosableShareTargetContainer,
        },
    },
    compute_node::{
        mock::MockVirtualGpuWorkerProxy,
        r#virtual::base::{BaseVirtualComputeNode, Command},
    },
};
use uuid::Uuid;
use worker::mock::MockWorker;

type Node =
    BaseVirtualComputeNode<MockWorker, MockNvidiaMigShareableGpu, MockVirtualGpuWorkerProxy>;
fn slice() -> VDeviceScale {
    VDeviceScale::A100(A100Template::Tiny)
}
fn settings() -> MockDeviceSettings {
    MockDeviceSettings {
        random_namespace: None,
        create_delay: Duration::from_secs(30),
        destroy_delay: Duration::from_secs(30),
        inference_timeout: Duration::from_secs(300),
        control_timeout: Duration::from_secs(300),
        model_lifecycle: Default::default(),
        execution_noise: Default::default(),
        share_targets: vec!["1g.10gb".into()],
        aliases: HashMap::new(),
    }
}
fn node(binary: &str) -> (Arc<Node>, Uuid) {
    node_with_settings(binary, settings())
}
fn node_with_settings(binary: &str, settings: MockDeviceSettings) -> (Arc<Node>, Uuid) {
    let device_id = Uuid::new_v4();
    let device = MockNvidiaMigShareableGpu::with_capacity(
        MockComputeDevice::with_simulation("test".into(), settings),
        NvidiaMigResource { ce: 1, memory: 10 },
    );
    let csv = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../logs/model_latency.csv");
    (
        Arc::new(Node::from_devices(
            HashMap::from([(device_id, (0, device))]),
            csv.clone(),
            csv,
            PathBuf::from(binary),
        )),
        device_id,
    )
}
fn create(id: Uuid, device_id: Uuid) -> Command<VDeviceScale> {
    Command::CreateWorker {
        id,
        device_id,
        params: slice(),
    }
}
async fn yield_tasks() {
    for _ in 0..20 {
        tokio::task::yield_now().await;
    }
}

#[cfg(unix)]
#[tokio::test]
async fn child_exiting_before_registration_refunds_its_reservation() {
    let (node, device) = node_with_settings(
        "/usr/bin/false",
        MockDeviceSettings {
            create_delay: Duration::ZERO,
            destroy_delay: Duration::ZERO,
            ..settings()
        },
    );
    for _ in 0..2 {
        let error = node
            .handle(create(Uuid::new_v4(), device))
            .await
            .unwrap_err();
        assert!(
            error.contains("startup pipe") || error.contains("before registration"),
            "{error}"
        );
        assert!(error.contains("Ok(())"), "cleanup must succeed: {error}");
    }
    node.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_child_startup_binds_distinct_owned_listeners() {
    let nodes = (0..12)
        .map(|_| {
            node_with_settings(
                env!("CARGO_BIN_EXE_mock_vgpu"),
                MockDeviceSettings {
                    create_delay: Duration::ZERO,
                    destroy_delay: Duration::ZERO,
                    ..settings()
                },
            )
        })
        .collect::<Vec<_>>();
    let ids = (0..nodes.len()).map(|_| Uuid::new_v4()).collect::<Vec<_>>();
    let results = futures::future::join_all(
        nodes
            .iter()
            .zip(&ids)
            .map(|((node, device), id)| node.handle(create(*id, *device))),
    )
    .await;
    assert!(
        results.iter().all(Result::is_ok),
        "every child must report its owned listener before readiness RPC"
    );
    let results = futures::future::join_all(
        nodes
            .iter()
            .zip(ids)
            .map(|((node, _), id)| node.handle(Command::DestroyWorker { id })),
    )
    .await;
    assert!(results.iter().all(Result::is_ok));
}

#[test]
fn resource_guards_exhaust_capacity_and_refund_only_after_last_clone() {
    let device = MockNvidiaMigShareableGpu::from(MockComputeDevice::new("nv".into()));
    let slices: Vec<_> = (0..7)
        .map(|_| device.share(slice()).unwrap().unwrap())
        .collect();
    assert_eq!(device.remain_resource().ce, 0);
    assert!(device.share(slice()).is_none());
    let clone = slices[0].clone();
    drop(slices);
    assert_eq!(device.remain_resource().ce, 6);
    drop(clone);
    assert_eq!(device.remain_resource(), device.total_resources());
    let hw = MockHuaweiMigShareableGpu::from(MockComputeDevice::new("hw".into()));
    let target = VChipScale::from_names(["vir05_1c_16g"])[0];
    let slices: Vec<_> = (0..4).map(|_| hw.share(target).unwrap().unwrap()).collect();
    assert!(hw.share(target).is_none());
    assert_eq!(
        hw.remain_resource(),
        HuaweiMigResource {
            ai_core: 0,
            ai_cpu: 3,
            memory: 0
        }
    );
    drop(slices);
    assert_eq!(hw.remain_resource(), hw.total_resources());
}

#[test]
fn whitelist_failure_refunds_and_separate_devices_do_not_pool_capacity() {
    let a = MockNvidiaMigShareableGpu::with_capacity(
        MockComputeDevice::with_simulation("a".into(), settings()),
        NvidiaMigResource { ce: 3, memory: 40 },
    );
    let b = MockNvidiaMigShareableGpu::with_capacity(
        MockComputeDevice::new("b".into()),
        NvidiaMigResource { ce: 3, memory: 40 },
    );
    assert!(
        a.share(VDeviceScale::A100(A100Template::Medium))
            .unwrap()
            .is_err()
    );
    assert_eq!(a.remain_resource(), a.total_resources());
    assert!(a.share(VDeviceScale::A100(A100Template::Large)).is_none());
    assert!(b.share(VDeviceScale::A100(A100Template::Large)).is_none());
}

#[tokio::test]
async fn lifecycle_reserves_while_creating_and_destroying_and_deduplicates() {
    // Parent and child must use the same real clock. Pausing only the parent's
    // Tokio clock can jump past an RPC deadline before the OS schedules the child.
    let (node, device_id) = node_with_settings(
        env!("CARGO_BIN_EXE_mock_vgpu"),
        MockDeviceSettings {
            create_delay: Duration::from_millis(200),
            destroy_delay: Duration::from_millis(200),
            ..settings()
        },
    );
    let id = Uuid::new_v4();
    let first = tokio::spawn({
        let node = node.clone();
        async move { node.handle(create(id, device_id)).await }
    });
    let retry = tokio::spawn({
        let node = node.clone();
        async move { node.handle(create(id, device_id)).await }
    });
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(!first.is_finished());
    assert!(node.state().await.unwrap().workers.is_empty());
    assert!(
        node.handle(create(Uuid::new_v4(), device_id))
            .await
            .is_err()
    );
    first.await.unwrap().unwrap();
    retry.await.unwrap().unwrap();
    assert_eq!(node.state().await.unwrap().workers.len(), 1);
    let destroy_started = std::time::Instant::now();
    let destroy = tokio::spawn({
        let node = node.clone();
        async move { node.handle(Command::DestroyWorker { id }).await }
    });
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(!destroy.is_finished());
    assert!(
        node.handle(create(Uuid::new_v4(), device_id))
            .await
            .is_err()
    );
    destroy.abort();
    let _ = destroy.await;
    node.handle(Command::DestroyWorker { id }).await.unwrap();
    assert!(destroy_started.elapsed() >= Duration::from_millis(200));
    assert!(node.state().await.unwrap().workers.is_empty());
    let replacement = Uuid::new_v4();
    node.handle(create(replacement, device_id)).await.unwrap();
    node.shutdown().await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn cancelled_or_failed_creation_releases_reservation() {
    let (node, device_id) = node("/no/such/mock-worker");
    let first = tokio::spawn({
        let node = node.clone();
        async move { node.handle(create(Uuid::new_v4(), device_id)).await }
    });
    yield_tasks().await;
    first.abort();
    let _ = first.await;
    let start = tokio::time::Instant::now();
    let result = node.handle(create(Uuid::new_v4(), device_id)).await;
    assert!(result.unwrap_err().contains("spawn"));
    assert!(start.elapsed() >= Duration::from_secs(30));
    assert!(
        node.handle(create(Uuid::new_v4(), device_id))
            .await
            .unwrap_err()
            .contains("spawn")
    );
}

#[tokio::test]
async fn inference_timeout_reaches_child_worker_proxy() {
    use compute_node::{
        compute_node::r#virtual::{BaseWork, base::Work},
        util::Input,
    };
    use worker::model::tensor::{DynTensor, TensorElementType};
    let (node, device_id) = node_with_settings(
        env!("CARGO_BIN_EXE_mock_vgpu"),
        MockDeviceSettings {
            create_delay: Duration::ZERO,
            destroy_delay: Duration::ZERO,
            inference_timeout: Duration::from_millis(1),
            ..settings()
        },
    );
    let id = Uuid::new_v4();
    node.handle(create(id, device_id)).await.unwrap();
    node.handle(Command::LoadModel {
        id,
        name: "resnet50".into(),
    })
    .await
    .unwrap();
    let result = node
        .compute(Work {
            target_worker: id,
            work: BaseWork::<Input>::new(
                Uuid::new_v4(),
                DynTensor::random(vec![1, 0, 0, 0], TensorElementType::Float),
            ),
        })
        .await;
    assert!(
        result
            .unwrap_err()
            .contains("Timeout waiting for RPC response")
    );
    node.handle(Command::DestroyWorker { id }).await.unwrap();
}

#[tokio::test]
#[ignore = "real wall clock: initial creation 15 s, replacement 30 s, final cleanup 15 s"]
async fn wall_clock_replacement_is_thirty_seconds() {
    let (node, device_id) = node_with_settings(
        env!("CARGO_BIN_EXE_mock_vgpu"),
        MockDeviceSettings {
            create_delay: Duration::from_secs(15),
            destroy_delay: Duration::from_secs(15),
            ..settings()
        },
    );
    let first = Uuid::new_v4();
    node.handle(create(first, device_id)).await.unwrap();
    let start = std::time::Instant::now();
    node.handle(Command::DestroyWorker { id: first })
        .await
        .unwrap();
    let second = Uuid::new_v4();
    node.handle(create(second, device_id)).await.unwrap();
    let elapsed = start.elapsed();
    println!("replacement elapsed: {elapsed:?}");
    assert!(elapsed >= Duration::from_secs(30));
    assert!(elapsed < Duration::from_secs(45));
    node.handle(Command::DestroyWorker { id: second })
        .await
        .unwrap();
}

#[tokio::test]
async fn model_readiness_is_delayed_and_shutdown_waits_for_unload_and_destroy() {
    let (node, device_id) = node_with_settings(
        env!("CARGO_BIN_EXE_mock_vgpu"),
        MockDeviceSettings {
            create_delay: Duration::ZERO,
            destroy_delay: Duration::from_millis(100),
            model_lifecycle: [(
                "1g.10gb".into(),
                [(
                    "resnet50".into(),
                    worker::mock::ModelLifecycle {
                        load_ms: 120,
                        unload_ms: 80,
                    },
                )]
                .into(),
            )]
            .into(),
            ..settings()
        },
    );
    let id = Uuid::new_v4();
    node.handle(create(id, device_id)).await.unwrap();
    let start = std::time::Instant::now();
    node.handle(Command::LoadModel {
        id,
        name: "resnet50".into(),
    })
    .await
    .unwrap();
    assert!(start.elapsed() >= Duration::from_millis(120));
    let start = std::time::Instant::now();
    node.handle(Command::LoadModel {
        id,
        name: "resnet50".into(),
    })
    .await
    .unwrap();
    assert!(start.elapsed() < Duration::from_millis(100));
    let start = std::time::Instant::now();
    node.shutdown().await.unwrap();
    assert!(start.elapsed() >= Duration::from_millis(180));
    assert!(node.state().await.unwrap().workers.is_empty());
    assert!(
        node.handle(create(Uuid::new_v4(), device_id))
            .await
            .unwrap_err()
            .contains("shutting down")
    );
}

#[tokio::test]
async fn timed_out_model_load_retry_observes_the_completed_load() {
    let (node, device_id) = node_with_settings(
        env!("CARGO_BIN_EXE_mock_vgpu"),
        MockDeviceSettings {
            create_delay: Duration::ZERO,
            destroy_delay: Duration::ZERO,
            control_timeout: Duration::from_millis(100),
            model_lifecycle: [(
                "1g.10gb".into(),
                [(
                    "resnet50".into(),
                    worker::mock::ModelLifecycle {
                        load_ms: 250,
                        unload_ms: 20,
                    },
                )]
                .into(),
            )]
            .into(),
            ..settings()
        },
    );
    let id = Uuid::new_v4();
    node.handle(create(id, device_id)).await.unwrap();
    assert!(
        node.handle(Command::LoadModel {
            id,
            name: "resnet50".into()
        })
        .await
        .unwrap_err()
        .contains("Timeout")
    );
    // The child keeps the accepted command alive after the RPC caller times out.
    tokio::time::sleep(Duration::from_millis(200)).await;
    let state = node
        .handle(Command::LoadModel {
            id,
            name: "resnet50".into(),
        })
        .await
        .unwrap();
    assert_eq!(state.workers[&id].model_name.as_deref(), Some("resnet50"));
    // Another full load would exceed the same 100 ms budget again.
    node.shutdown().await.unwrap();
}
