//! scheduler模块的综合测试
//!
//! 这个模块包含对scheduler/mod.rs中所有函数的详细测试，
//! 覆盖正常情况、边缘情况和错误处理。

use std::collections::HashMap;

use uuid::Uuid;

use super::{
    DeviceAllocationMap, build_assignment_pairs, calculate_reallocation_cost,
    compare_and_generate_commands, compute_optimal_device_assignment, extract_device_share_params,
    generate_migration_commands,
};
use crate::compute_node::r#virtual::{
    base::Command as BaseCommand,
    virtual_compute_node::{State, VirtualComputeNode},
};

#[tokio::test]
async fn failed_unload_does_not_execute_the_dependent_load() {
    use std::sync::{Arc, Mutex};

    use crate::compute_node::r#virtual::virtual_compute_node::{
        base_virtual_compute_node_ext::{
            CommandFuture, ComputeFuture, DynBaseVirtualComputeNode, StateFuture, StatusFuture,
        },
        components::{
            request_scheduler::batch_size_extractor::BatchExtractorStrategy,
            worker_registry::{WorkerInfo, WorkerStatus},
        },
        tests::MockResponse,
    };
    struct FailingNode(Arc<Mutex<Vec<BaseCommand<String>>>>);
    impl DynBaseVirtualComputeNode for FailingNode {
        type ShareParam = String;
        type DeviceState = MockResponse;
        fn command(&self, command: BaseCommand<String>) -> CommandFuture<'_, String> {
            Box::pin(async move {
                self.0.lock().unwrap().push(command);
                Err("injected command failure".into())
            })
        }
        fn state(&self) -> StateFuture<'_, String> {
            Box::pin(async { Err("unused".into()) })
        }
        fn status(&self) -> StatusFuture<'_, MockResponse> {
            Box::pin(async { Err("unused".into()) })
        }
        fn compute(
            &self,
            _: crate::compute_node::r#virtual::base::Work<crate::util::Input>,
        ) -> ComputeFuture<'_> {
            Box::pin(async { Err("unused".into()) })
        }
    }
    let node = VirtualComputeNode::<String, MockResponse>::new(
        BatchExtractorStrategy::split_and_satisfy(),
        |_, _, _| 0.01,
    );
    let node_id = Uuid::new_v4();
    let worker_id = Uuid::new_v4();
    let calls = Arc::new(Mutex::new(Vec::new()));
    node.worker_manager
        .nodes
        .insert(node_id, Arc::new(FailingNode(calls.clone())));
    node.worker_manager
        .get_registry()
        .register_worker(WorkerInfo {
            worker_id,
            model_name: "old".into(),
            node_id,
            device_id: Uuid::new_v4(),
            device_params: "slice".into(),
            created_at: std::time::Instant::now(),
            status: WorkerStatus::Paused,
        });
    let result = node
        .execute_migration_plan_ordered(vec![(
            node_id,
            vec![
                BaseCommand::UnloadModel { id: worker_id },
                BaseCommand::LoadModel {
                    id: worker_id,
                    name: "new".into(),
                },
            ],
        )])
        .await;
    assert!(result.is_err());
    assert_eq!(
        node.worker_manager
            .get_registry()
            .get_worker_info(worker_id)
            .unwrap()
            .status,
        WorkerStatus::Paused
    );
    let calls = calls.lock().unwrap();
    assert!(
        calls
            .iter()
            .any(|c| matches!(c, BaseCommand::UnloadModel { .. }))
    );
    assert!(
        calls
            .iter()
            .all(|c| !matches!(c, BaseCommand::LoadModel { .. }))
    );
}

/// 测试extract_device_share_params函数
#[tokio::test]
async fn command_state_is_used_when_a_followup_query_would_fail() {
    use std::sync::Arc;

    use crate::compute_node::r#virtual::{
        base,
        virtual_compute_node::{
            base_virtual_compute_node_ext::{
                CommandFuture, ComputeFuture, DynBaseVirtualComputeNode, StateFuture, StatusFuture,
            },
            components::{
                request_scheduler::batch_size_extractor::BatchExtractorStrategy,
                worker_registry::WorkerStatus,
            },
            tests::MockResponse,
        },
    };
    struct ConfirmedUnload;
    impl DynBaseVirtualComputeNode for ConfirmedUnload {
        type ShareParam = String;
        type DeviceState = MockResponse;
        fn command(&self, command: BaseCommand<String>) -> CommandFuture<'_, String> {
            Box::pin(async move {
                let BaseCommand::UnloadModel { id } = command else {
                    return Err("unexpected command".into());
                };
                Ok(base::State {
                    workers: HashMap::from([(
                        id,
                        base::WorkerState {
                            model_name: None,
                            params: "slice".into(),
                            device_id: Uuid::nil(),
                            experiment_noise_slot: None,
                        },
                    )]),
                })
            })
        }
        fn state(&self) -> StateFuture<'_, String> {
            Box::pin(async { Err("injected query failure".into()) })
        }
        fn status(&self) -> StatusFuture<'_, MockResponse> {
            Box::pin(async { Err("unused".into()) })
        }
        fn compute(&self, _: base::Work<crate::util::Input>) -> ComputeFuture<'_> {
            Box::pin(async { Err("unused".into()) })
        }
    }
    let node = VirtualComputeNode::<String, MockResponse>::new(
        BatchExtractorStrategy::split_and_satisfy(),
        |_, _, _| 0.01,
    );
    let node_id = Uuid::new_v4();
    let id = Uuid::new_v4();
    node.worker_manager
        .nodes
        .insert(node_id, Arc::new(ConfirmedUnload));
    node.add_worker(id, "old".into(), node_id, Uuid::nil(), "slice".into())
        .unwrap();
    node.worker_manager
        .get_registry()
        .update_worker_status(id, WorkerStatus::Paused);
    node.execute_command_with_sync(node_id, BaseCommand::UnloadModel { id })
        .await
        .unwrap();
    assert!(
        node.worker_manager
            .get_registry()
            .get_worker_info(id)
            .is_none()
    );
}

#[cfg(test)]
mod test_extract_device_share_params {
    use super::*;

    #[test]
    fn test_extract_empty_allocation() {
        let empty_allocation: DeviceAllocationMap<String> = HashMap::new();
        let result = extract_device_share_params(&empty_allocation);
        assert!(result.is_empty());
    }

    #[test]
    fn test_extract_single_device_single_worker() {
        let device_id = Uuid::new_v4();
        let node_id = Uuid::new_v4();
        let worker_id = Uuid::new_v4();

        let mut allocation = HashMap::new();
        allocation.insert(
            device_id,
            (
                node_id,
                vec![(worker_id, "param1".to_string(), Some("model1".to_string()))],
            ),
        );

        let result = extract_device_share_params(&allocation);
        assert_eq!(result.len(), 1);
        assert_eq!(result[&device_id], vec!["param1".to_string()]);
    }

    #[test]
    fn test_extract_single_device_multiple_workers() {
        let device_id = Uuid::new_v4();
        let node_id = Uuid::new_v4();
        let worker1_id = Uuid::new_v4();
        let worker2_id = Uuid::new_v4();

        let mut allocation = HashMap::new();
        allocation.insert(
            device_id,
            (
                node_id,
                vec![
                    (worker1_id, "param1".to_string(), Some("model1".to_string())),
                    (worker2_id, "param2".to_string(), Some("model2".to_string())),
                ],
            ),
        );

        let result = extract_device_share_params(&allocation);
        assert_eq!(result.len(), 1);
        assert_eq!(result[&device_id].len(), 2);
        assert!(result[&device_id].contains(&"param1".to_string()));
        assert!(result[&device_id].contains(&"param2".to_string()));
    }

    #[test]
    fn test_extract_multiple_devices() {
        let device1_id = Uuid::new_v4();
        let device2_id = Uuid::new_v4();
        let node_id = Uuid::new_v4();
        let worker1_id = Uuid::new_v4();
        let worker2_id = Uuid::new_v4();

        let mut allocation = HashMap::new();
        allocation.insert(
            device1_id,
            (
                node_id,
                vec![(worker1_id, "param1".to_string(), Some("model1".to_string()))],
            ),
        );
        allocation.insert(
            device2_id,
            (
                node_id,
                vec![(worker2_id, "param2".to_string(), Some("model2".to_string()))],
            ),
        );

        let result = extract_device_share_params(&allocation);
        assert_eq!(result.len(), 2);
        assert_eq!(result[&device1_id], vec!["param1".to_string()]);
        assert_eq!(result[&device2_id], vec!["param2".to_string()]);
    }

    #[test]
    fn test_extract_device_with_no_workers() {
        let device_id = Uuid::new_v4();
        let node_id = Uuid::new_v4();

        let mut allocation = HashMap::new();
        allocation.insert(device_id, (node_id, vec![]));

        let result = extract_device_share_params::<String>(&allocation);
        assert_eq!(result.len(), 1);
        assert!(result[&device_id].is_empty());
    }
}

/// 测试calculate_reallocation_cost函数
#[cfg(test)]
mod test_calculate_reallocation_cost {
    use super::*;

    #[test]
    fn test_identical_params() {
        let current = vec!["param1".to_string(), "param2".to_string()];
        let target = vec!["param1".to_string(), "param2".to_string()];
        let cost = calculate_reallocation_cost(&current, &target);
        assert_eq!(cost, 0);
    }

    #[test]
    fn test_completely_different_params() {
        let current = vec!["param1".to_string(), "param2".to_string()];
        let target = vec!["param3".to_string(), "param4".to_string()];
        let cost = calculate_reallocation_cost(&current, &target);
        assert_eq!(cost, 4); // 需要移除2个，添加2个
    }

    #[test]
    fn test_partial_overlap() {
        let current = vec!["param1".to_string(), "param2".to_string()];
        let target = vec!["param1".to_string(), "param3".to_string()];
        let cost = calculate_reallocation_cost(&current, &target);
        assert_eq!(cost, 2); // 需要移除param2，添加param3
    }

    #[test]
    fn test_empty_current() {
        let current: Vec<String> = vec![];
        let target = vec!["param1".to_string(), "param2".to_string()];
        let cost = calculate_reallocation_cost(&current, &target);
        assert_eq!(cost, 2); // 需要添加2个
    }

    #[test]
    fn test_empty_target() {
        let current = vec!["param1".to_string(), "param2".to_string()];
        let target: Vec<String> = vec![];
        let cost = calculate_reallocation_cost(&current, &target);
        assert_eq!(cost, 2); // 需要移除2个
    }

    #[test]
    fn test_both_empty() {
        let current: Vec<String> = vec![];
        let target: Vec<String> = vec![];
        let cost = calculate_reallocation_cost(&current, &target);
        assert_eq!(cost, 0);
    }

    #[test]
    fn test_duplicate_params() {
        let current = vec![
            "param1".to_string(),
            "param1".to_string(),
            "param2".to_string(),
        ];
        let target = vec!["param1".to_string(), "param3".to_string()];
        let cost = calculate_reallocation_cost(&current, &target);
        // current有2个param1和1个param2，target需要1个param1和1个param3
        // 保留1个param1，移除1个param1和1个param2，添加1个param3
        assert_eq!(cost, 3);
    }
}

/// 测试compute_optimal_device_assignment函数
#[cfg(test)]
mod test_compute_optimal_device_assignment {
    use super::*;

    #[test]
    fn test_assignment_equal_devices() {
        let device1 = Uuid::new_v4();
        let device2 = Uuid::new_v4();
        let target1 = Uuid::new_v4();
        let target2 = Uuid::new_v4();

        let mut current_params = HashMap::new();
        current_params.insert(device1, vec!["param1".to_string()]);
        current_params.insert(device2, vec!["param2".to_string()]);

        let mut target_params = HashMap::new();
        target_params.insert(target1, vec!["param1".to_string()]);
        target_params.insert(target2, vec!["param2".to_string()]);

        let result = compute_optimal_device_assignment(&current_params, &target_params).unwrap();
        assert_eq!(result.len(), 2);

        // 验证分配是最优的（成本为0）
        for (_, _, cost) in &result {
            assert_eq!(*cost, 0);
        }
    }

    #[test]
    fn test_assignment_more_current_than_target() {
        let device1 = Uuid::new_v4();
        let device2 = Uuid::new_v4();
        let device3 = Uuid::new_v4();
        let target1 = Uuid::new_v4();
        let target2 = Uuid::new_v4();

        let mut current_params = HashMap::new();
        current_params.insert(device1, vec!["param1".to_string()]);
        current_params.insert(device2, vec!["param2".to_string()]);
        current_params.insert(device3, vec!["param3".to_string()]);

        let mut target_params = HashMap::new();
        target_params.insert(target1, vec!["param1".to_string()]);
        target_params.insert(target2, vec!["param2".to_string()]);

        assert!(compute_optimal_device_assignment(&current_params, &target_params).is_err());
    }

    #[test]
    fn test_assignment_more_target_than_current() {
        let device1 = Uuid::new_v4();
        let target1 = Uuid::new_v4();
        let target2 = Uuid::new_v4();

        let mut current_params = HashMap::new();
        current_params.insert(device1, vec!["param1".to_string()]);

        let mut target_params = HashMap::new();
        target_params.insert(target1, vec!["param1".to_string()]);
        target_params.insert(target2, vec!["param2".to_string()]);

        let result = compute_optimal_device_assignment(&current_params, &target_params);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("equal device counts"));
    }

    #[test]
    fn test_assignment_empty_current() {
        let current_params: HashMap<Uuid, Vec<String>> = HashMap::new();
        let target1 = Uuid::new_v4();

        let mut target_params = HashMap::new();
        target_params.insert(target1, vec!["param1".to_string()]);

        let result = compute_optimal_device_assignment(&current_params, &target_params);
        assert!(result.is_err());
    }

    #[test]
    fn test_assignment_empty_target() {
        let device1 = Uuid::new_v4();
        let mut current_params = HashMap::new();
        current_params.insert(device1, vec!["param1".to_string()]);

        let target_params: HashMap<Uuid, Vec<String>> = HashMap::new();

        assert!(compute_optimal_device_assignment(&current_params, &target_params).is_err());
    }

    #[test]
    fn test_assignment_optimal_matching() {
        let device1 = Uuid::new_v4();
        let device2 = Uuid::new_v4();
        let target1 = Uuid::new_v4();
        let target2 = Uuid::new_v4();

        let mut current_params = HashMap::new();
        current_params.insert(device1, vec!["param1".to_string(), "param2".to_string()]);
        current_params.insert(device2, vec!["param3".to_string()]);

        let mut target_params = HashMap::new();
        target_params.insert(target1, vec!["param3".to_string()]);
        target_params.insert(target2, vec!["param1".to_string(), "param2".to_string()]);

        let result = compute_optimal_device_assignment(&current_params, &target_params).unwrap();
        assert_eq!(result.len(), 2);
        assert!(result.contains(&(device1, target2, 0)));
        assert!(result.contains(&(device2, target1, 0)));

        // 验证总成本是最小的
        let total_cost: u32 = result.iter().map(|(_, _, cost)| cost).sum();
        assert_eq!(total_cost, 0); // 完美匹配应该是0成本
    }
}

/// 测试compare_and_generate_commands函数
#[cfg(test)]
mod test_compare_and_generate_commands {
    use super::*;

    #[test]
    fn test_no_changes_needed() {
        let device_id = Uuid::new_v4();
        let worker_id = Uuid::new_v4();

        let current_allocs = vec![(worker_id, "param1".to_string(), Some("model1".to_string()))];
        let target_allocs = vec![("param1".to_string(), "model1".to_string())];

        let commands = compare_and_generate_commands(&current_allocs, &target_allocs, device_id);
        assert!(commands.is_empty());
    }

    #[test]
    fn test_model_reload_same_param() {
        let device_id = Uuid::new_v4();
        let worker_id = Uuid::new_v4();

        let current_allocs = vec![(worker_id, "param1".to_string(), Some("model1".to_string()))];
        let target_allocs = vec![("param1".to_string(), "model2".to_string())];

        let commands = compare_and_generate_commands(&current_allocs, &target_allocs, device_id);
        assert_eq!(commands.len(), 2);

        // 验证第一个命令是UnloadModel
        match &commands[0] {
            BaseCommand::UnloadModel { id } => {
                assert_eq!(
                    *id, worker_id,
                    "UnloadModel command should target the correct worker"
                );
            }
            _ => panic!(
                "Expected UnloadModel command as first command, got: {:?}",
                commands[0]
            ),
        }

        // 验证第二个命令是LoadModel
        match &commands[1] {
            BaseCommand::LoadModel { id, name } => {
                assert_eq!(
                    *id, worker_id,
                    "LoadModel command should target the same worker"
                );
                assert_eq!(
                    name, "model2",
                    "LoadModel command should load the correct model"
                );
            }
            _ => panic!(
                "Expected LoadModel command as second command, got: {:?}",
                commands[1]
            ),
        }
    }

    #[test]
    fn test_create_new_worker() {
        let device_id = Uuid::new_v4();

        let current_allocs = vec![];
        let target_allocs = vec![("param1".to_string(), "model1".to_string())];

        let commands = compare_and_generate_commands(&current_allocs, &target_allocs, device_id);
        assert_eq!(commands.len(), 2);

        // 提取worker ID以验证命令一致性
        let created_worker_id = match &commands[0] {
            BaseCommand::CreateWorker {
                id,
                device_id: cmd_device_id,
                params,
            } => {
                assert_eq!(
                    *cmd_device_id, device_id,
                    "CreateWorker should target the correct device"
                );
                assert_eq!(
                    params, "param1",
                    "CreateWorker should use the correct parameters"
                );
                *id
            }
            _ => panic!(
                "Expected CreateWorker command as first command, got: {:?}",
                commands[0]
            ),
        };

        // 验证LoadModel命令使用相同的worker ID
        match &commands[1] {
            BaseCommand::LoadModel { id, name } => {
                assert_eq!(
                    *id, created_worker_id,
                    "LoadModel should target the newly created worker"
                );
                assert_eq!(name, "model1", "LoadModel should load the correct model");
            }
            _ => panic!(
                "Expected LoadModel command as second command, got: {:?}",
                commands[1]
            ),
        }
    }

    #[test]
    fn test_destroy_unused_worker() {
        let device_id = Uuid::new_v4();
        let worker_id = Uuid::new_v4();

        let current_allocs = vec![(worker_id, "param1".to_string(), Some("model1".to_string()))];
        let target_allocs = vec![];

        let commands = compare_and_generate_commands(&current_allocs, &target_allocs, device_id);
        assert_eq!(commands.len(), 1);

        match &commands[0] {
            BaseCommand::DestroyWorker { id } => {
                assert_eq!(
                    *id, worker_id,
                    "DestroyWorker should target the correct worker"
                );
            }
            _ => panic!("Expected DestroyWorker command, got: {:?}", commands[0]),
        }
    }

    #[test]
    fn test_complex_scenario() {
        let device_id = Uuid::new_v4();
        let worker1_id = Uuid::new_v4();
        let worker2_id = Uuid::new_v4();
        let worker3_id = Uuid::new_v4();

        let current_allocs = vec![
            (worker1_id, "param1".to_string(), Some("model1".to_string())),
            (worker2_id, "param2".to_string(), Some("model2".to_string())),
            (worker3_id, "param3".to_string(), Some("model3".to_string())),
        ];

        let target_allocs = vec![
            ("param1".to_string(), "model1".to_string()), // 保持不变
            ("param2".to_string(), "model4".to_string()), // 更换模型
            ("param4".to_string(), "model5".to_string()), // 新增
                                                          // param3/model3 被移除
        ];

        let commands = compare_and_generate_commands(&current_allocs, &target_allocs, device_id);

        // 分离命令以便详细验证
        let mut unload_commands = Vec::new();
        let mut load_commands = Vec::new();
        let mut create_commands = Vec::new();
        let mut destroy_commands = Vec::new();

        for cmd in &commands {
            match cmd {
                BaseCommand::UnloadModel { id } => unload_commands.push(*id),
                BaseCommand::LoadModel { id, name } => load_commands.push((*id, name.clone())),
                BaseCommand::CreateWorker {
                    id,
                    device_id: cmd_device_id,
                    params,
                } => {
                    create_commands.push((*id, *cmd_device_id, params.clone()));
                }
                BaseCommand::DestroyWorker { id } => destroy_commands.push(*id),
            }
        }

        // 验证数量
        assert_eq!(
            unload_commands.len(),
            1,
            "Should have exactly 1 unload command"
        );
        assert_eq!(
            load_commands.len(),
            2,
            "Should have exactly 2 load commands"
        );
        assert_eq!(
            create_commands.len(),
            1,
            "Should have exactly 1 create command"
        );
        assert_eq!(
            destroy_commands.len(),
            1,
            "Should have exactly 1 destroy command"
        );

        // 验证unload命令：应该是worker2（重新加载model4）
        assert_eq!(
            unload_commands[0], worker2_id,
            "Unload should target worker2"
        );

        // 验证destroy命令：应该是worker3（被移除）
        assert_eq!(
            destroy_commands[0], worker3_id,
            "Destroy should target worker3"
        );

        // 验证create命令：应该为param4创建新worker
        let (created_worker_id, created_device_id, created_params) = &create_commands[0];
        assert_eq!(
            *created_device_id, device_id,
            "Create command should target correct device"
        );
        assert_eq!(created_params, "param4", "Create command should use param4");

        // 验证load命令：应该包含worker2加载model4和新worker加载model5
        let mut found_worker2_load = false;
        let mut found_new_worker_load = false;

        for (load_worker_id, load_model_name) in &load_commands {
            if *load_worker_id == worker2_id {
                assert_eq!(load_model_name, "model4", "Worker2 should load model4");
                found_worker2_load = true;
            } else if *load_worker_id == *created_worker_id {
                assert_eq!(load_model_name, "model5", "New worker should load model5");
                found_new_worker_load = true;
            } else {
                panic!("Unexpected load command for worker: {load_worker_id}");
            }
        }

        assert!(found_worker2_load, "Should have load command for worker2");
        assert!(
            found_new_worker_load,
            "Should have load command for new worker"
        );
    }

    #[test]
    fn test_multiple_workers_same_param() {
        let device_id = Uuid::new_v4();
        let worker1_id = Uuid::new_v4();
        let worker2_id = Uuid::new_v4();

        let current_allocs = vec![
            (worker1_id, "param1".to_string(), Some("model1".to_string())),
            (worker2_id, "param1".to_string(), Some("model1".to_string())),
        ];

        let target_allocs = vec![("param1".to_string(), "model1".to_string())];

        let commands = compare_and_generate_commands(&current_allocs, &target_allocs, device_id);
        assert_eq!(
            commands.len(),
            1,
            "Should have exactly 1 command to remove redundant worker"
        );

        // 应该销毁一个多余的worker
        match &commands[0] {
            BaseCommand::DestroyWorker { id } => {
                assert!(
                    id == &worker1_id || id == &worker2_id,
                    "DestroyWorker should target one of the redundant workers, got: {id}"
                );
            }
            _ => panic!("Expected DestroyWorker command, got: {:?}", commands[0]),
        }
    }
}

/// 测试generate_migration_commands函数
#[cfg(test)]
mod test_generate_migration_commands {
    use super::*;

    #[test]
    fn test_empty_assignment() {
        let assignment_pairs: HashMap<Uuid, Vec<(String, String)>> = HashMap::new();
        let current_allocation: DeviceAllocationMap<String> = HashMap::new();

        let commands = generate_migration_commands(&assignment_pairs, &current_allocation);
        assert!(commands.is_empty());
    }

    #[test]
    fn test_single_device_no_changes() {
        let device_id = Uuid::new_v4();
        let node_id = Uuid::new_v4();
        let worker_id = Uuid::new_v4();

        let mut assignment_pairs = HashMap::new();
        assignment_pairs.insert(
            device_id,
            vec![("param1".to_string(), "model1".to_string())],
        );

        let mut current_allocation = HashMap::new();
        current_allocation.insert(
            device_id,
            (
                node_id,
                vec![(worker_id, "param1".to_string(), Some("model1".to_string()))],
            ),
        );

        let commands = generate_migration_commands(&assignment_pairs, &current_allocation);
        assert!(commands.is_empty()); // 无需变更
    }

    #[test]
    fn test_single_device_with_changes() {
        let device_id = Uuid::new_v4();
        let node_id = Uuid::new_v4();
        let worker_id = Uuid::new_v4();

        let mut assignment_pairs = HashMap::new();
        assignment_pairs.insert(
            device_id,
            vec![("param1".to_string(), "model2".to_string())],
        );

        let mut current_allocation = HashMap::new();
        current_allocation.insert(
            device_id,
            (
                node_id,
                vec![(worker_id, "param1".to_string(), Some("model1".to_string()))],
            ),
        );

        let commands = generate_migration_commands(&assignment_pairs, &current_allocation);
        assert_eq!(
            commands.len(),
            1,
            "Should have commands for exactly 1 device"
        );

        let (migration_node_id, device_commands) = &commands[0];
        assert_eq!(
            *migration_node_id, node_id,
            "Migration should target the correct node"
        );
        assert_eq!(
            device_commands.len(),
            2,
            "Should have exactly 2 commands (unload + load)"
        );

        // 验证第一个命令是UnloadModel
        match &device_commands[0] {
            BaseCommand::UnloadModel { id } => {
                assert_eq!(
                    *id, worker_id,
                    "UnloadModel should target the correct worker"
                );
            }
            _ => panic!(
                "Expected UnloadModel as first command, got: {:?}",
                device_commands[0]
            ),
        }

        // 验证第二个命令是LoadModel
        match &device_commands[1] {
            BaseCommand::LoadModel { id, name } => {
                assert_eq!(*id, worker_id, "LoadModel should target the same worker");
                assert_eq!(
                    name, "model2",
                    "LoadModel should load the correct new model"
                );
            }
            _ => panic!(
                "Expected LoadModel as second command, got: {:?}",
                device_commands[1]
            ),
        }
    }

    #[test]
    fn test_multiple_devices() {
        let device1_id = Uuid::new_v4();
        let device2_id = Uuid::new_v4();
        let node_id = Uuid::new_v4();
        let worker1_id = Uuid::new_v4();

        let mut assignment_pairs = HashMap::new();
        assignment_pairs.insert(
            device1_id,
            vec![("param1".to_string(), "model2".to_string())],
        );
        assignment_pairs.insert(
            device2_id,
            vec![("param3".to_string(), "model3".to_string())],
        );

        let mut current_allocation = HashMap::new();
        current_allocation.insert(
            device1_id,
            (
                node_id,
                vec![(worker1_id, "param1".to_string(), Some("model1".to_string()))],
            ),
        );
        current_allocation.insert(device2_id, (node_id, vec![]));

        let commands = generate_migration_commands(&assignment_pairs, &current_allocation);

        // 由于两个设备都属于同一个节点，命令可能会被合并到一个节点命令中
        // 或者分别为每个有变更的设备生成命令
        assert!(
            !commands.is_empty() && commands.len() <= 2,
            "Should have commands for 1-2 node entries, got {}",
            commands.len()
        );

        // 分类命令并验证
        let mut unload_commands = Vec::new();
        let mut load_commands = Vec::new();
        let mut create_commands = Vec::new();

        for (migration_node_id, device_commands) in &commands {
            assert_eq!(
                *migration_node_id, node_id,
                "All migration commands should target the same node"
            );

            for cmd in device_commands {
                match cmd {
                    BaseCommand::UnloadModel { id } => unload_commands.push(*id),
                    BaseCommand::LoadModel { id, name } => load_commands.push((*id, name.clone())),
                    BaseCommand::CreateWorker {
                        id,
                        device_id: cmd_device_id,
                        params,
                    } => create_commands.push((*id, *cmd_device_id, params.clone())),
                    _ => {}
                }
            }
        }

        // 验证命令数量和内容
        assert_eq!(unload_commands.len(), 1, "Should have 1 unload command");
        assert_eq!(load_commands.len(), 2, "Should have 2 load commands");
        assert_eq!(create_commands.len(), 1, "Should have 1 create command");

        // 验证unload命令
        assert_eq!(
            unload_commands[0], worker1_id,
            "Unload should target worker1"
        );

        // 验证create命令
        let (created_worker_id, created_device_id, created_params) = &create_commands[0];
        assert_eq!(
            *created_device_id, device2_id,
            "Create should target device2"
        );
        assert_eq!(created_params, "param3", "Create should use param3");

        // 验证load命令
        let mut worker1_reloaded = false;
        let mut new_worker_loaded = false;

        for (load_worker_id, model_name) in &load_commands {
            if *load_worker_id == worker1_id {
                assert_eq!(model_name, "model2", "Worker1 should load model2");
                worker1_reloaded = true;
            } else if *load_worker_id == *created_worker_id {
                assert_eq!(model_name, "model3", "New worker should load model3");
                new_worker_loaded = true;
            } else {
                panic!("Unexpected worker in load command: {load_worker_id}");
            }
        }

        assert!(worker1_reloaded, "Worker1 should be reloaded with model2");
        assert!(new_worker_loaded, "New worker should be loaded with model3");
    }
}

/// 测试build_assignment_pairs函数
#[cfg(test)]
mod test_build_assignment_pairs {
    use super::*;

    #[test]
    fn test_simple_direct_assignment() {
        let device_id = Uuid::new_v4();
        let target_device_id = Uuid::new_v4();
        let node_id = Uuid::new_v4();
        let worker_id = Uuid::new_v4();

        let mut current_allocation = HashMap::new();
        current_allocation.insert(
            device_id,
            (
                node_id,
                vec![(worker_id, "param1".to_string(), Some("model1".to_string()))],
            ),
        );

        let mut target_model_allocation = HashMap::new();
        target_model_allocation.insert("model1".to_string(), vec!["param1".to_string()]);

        let optimal_assignment = vec![(device_id, target_device_id, 0)];

        let mut target_device_params = HashMap::new();
        target_device_params.insert(target_device_id, vec!["param1".to_string()]);

        let result = build_assignment_pairs(
            &current_allocation,
            &target_model_allocation,
            &optimal_assignment,
            &target_device_params,
        );

        assert_eq!(
            result.len(),
            1,
            "Should have assignments for exactly 1 device"
        );
        assert_eq!(
            result[&device_id].len(),
            1,
            "Device should have exactly 1 parameter assignment"
        );

        let (assigned_param, assigned_model) = &result[&device_id][0];
        assert_eq!(
            assigned_param, "param1",
            "Should assign the correct parameter"
        );
        assert_eq!(assigned_model, "model1", "Should assign the correct model");
    }

    #[test]
    fn test_model_reallocation() {
        let device_id = Uuid::new_v4();
        let target_device_id = Uuid::new_v4();
        let node_id = Uuid::new_v4();
        let worker_id = Uuid::new_v4();

        let mut current_allocation = HashMap::new();
        current_allocation.insert(
            device_id,
            (
                node_id,
                vec![(worker_id, "param1".to_string(), Some("model1".to_string()))],
            ),
        );

        let mut target_model_allocation = HashMap::new();
        target_model_allocation.insert("model2".to_string(), vec!["param1".to_string()]);

        let optimal_assignment = vec![(device_id, target_device_id, 1)];

        let mut target_device_params = HashMap::new();
        target_device_params.insert(target_device_id, vec!["param1".to_string()]);

        let result = build_assignment_pairs(
            &current_allocation,
            &target_model_allocation,
            &optimal_assignment,
            &target_device_params,
        );

        assert_eq!(
            result.len(),
            1,
            "Should have assignments for exactly 1 device"
        );
        assert_eq!(
            result[&device_id].len(),
            1,
            "Device should have exactly 1 parameter assignment"
        );

        let (assigned_param, assigned_model) = &result[&device_id][0];
        assert_eq!(
            assigned_param, "param1",
            "Should assign the correct parameter"
        );
        assert_eq!(
            assigned_model, "model2",
            "Should reallocate to the new model"
        );
    }

    #[test]
    fn test_multiple_params_prioritization() {
        let device_id = Uuid::new_v4();
        let target_device_id = Uuid::new_v4();
        let node_id = Uuid::new_v4();
        let worker1_id = Uuid::new_v4();
        let worker2_id = Uuid::new_v4();

        let mut current_allocation = HashMap::new();
        current_allocation.insert(
            device_id,
            (
                node_id,
                vec![
                    (worker1_id, "param1".to_string(), Some("model1".to_string())),
                    (worker2_id, "param2".to_string(), Some("model2".to_string())),
                ],
            ),
        );

        let mut target_model_allocation = HashMap::new();
        target_model_allocation.insert("model1".to_string(), vec!["param1".to_string()]);
        target_model_allocation.insert("model3".to_string(), vec!["param2".to_string()]);

        let optimal_assignment = vec![(device_id, target_device_id, 1)];

        let mut target_device_params = HashMap::new();
        target_device_params.insert(
            target_device_id,
            vec!["param1".to_string(), "param2".to_string()],
        );

        let result = build_assignment_pairs(
            &current_allocation,
            &target_model_allocation,
            &optimal_assignment,
            &target_device_params,
        );

        assert_eq!(
            result.len(),
            1,
            "Should have assignments for exactly 1 device"
        );
        assert_eq!(
            result[&device_id].len(),
            2,
            "Device should have exactly 2 parameter assignments"
        );

        // 验证具体的分配内容
        let assignments = &result[&device_id];
        let param1_assignment = assignments
            .iter()
            .find(|(param, _)| param == "param1")
            .expect("Should have assignment for param1");
        let param2_assignment = assignments
            .iter()
            .find(|(param, _)| param == "param2")
            .expect("Should have assignment for param2");

        assert_eq!(
            param1_assignment.1, "model1",
            "param1 should keep its existing model1"
        );
        assert_eq!(
            param2_assignment.1, "model3",
            "param2 should be reassigned to model3"
        );
    }

    #[test]
    fn test_insufficient_models() {
        let device_id = Uuid::new_v4();
        let target_device_id = Uuid::new_v4();
        let node_id = Uuid::new_v4();

        let mut current_allocation = HashMap::new();
        current_allocation.insert(device_id, (node_id, vec![]));

        let mut target_model_allocation = HashMap::new();
        target_model_allocation.insert("model1".to_string(), vec!["param1".to_string()]);

        let optimal_assignment = vec![(device_id, target_device_id, 0)];

        let mut target_device_params = HashMap::new();
        target_device_params.insert(
            target_device_id,
            vec!["param1".to_string(), "param2".to_string()],
        ); // 需要2个param，但model1只提供1个

        // 这应该导致panic，因为没有足够的模型分配
        std::panic::catch_unwind(|| {
            build_assignment_pairs(
                &current_allocation,
                &target_model_allocation,
                &optimal_assignment,
                &target_device_params,
            )
        })
        .expect_err("Should panic due to insufficient model allocation");
    }

    #[test]
    fn test_complex_multi_device_scenario() {
        let device1_id = Uuid::new_v4();
        let device2_id = Uuid::new_v4();
        let target1_id = Uuid::new_v4();
        let target2_id = Uuid::new_v4();
        let node_id = Uuid::new_v4();
        let worker1_id = Uuid::new_v4();
        let worker2_id = Uuid::new_v4();

        let mut current_allocation = HashMap::new();
        current_allocation.insert(
            device1_id,
            (
                node_id,
                vec![(worker1_id, "param1".to_string(), Some("model1".to_string()))],
            ),
        );
        current_allocation.insert(
            device2_id,
            (
                node_id,
                vec![(worker2_id, "param2".to_string(), Some("model2".to_string()))],
            ),
        );

        let mut target_model_allocation = HashMap::new();
        target_model_allocation.insert("model1".to_string(), vec!["param2".to_string()]);
        target_model_allocation.insert("model2".to_string(), vec!["param1".to_string()]);

        let optimal_assignment = vec![(device1_id, target1_id, 1), (device2_id, target2_id, 1)];

        let mut target_device_params = HashMap::new();
        target_device_params.insert(target1_id, vec!["param1".to_string()]);
        target_device_params.insert(target2_id, vec!["param2".to_string()]);

        let result = build_assignment_pairs(
            &current_allocation,
            &target_model_allocation,
            &optimal_assignment,
            &target_device_params,
        );

        assert_eq!(
            result.len(),
            2,
            "Should have assignments for exactly 2 devices"
        );
        assert_eq!(
            result[&device1_id].len(),
            1,
            "Device1 should have exactly 1 assignment"
        );
        assert_eq!(
            result[&device2_id].len(),
            1,
            "Device2 should have exactly 1 assignment"
        );

        // 验证device1的分配：param1 -> model2
        let (device1_param, device1_model) = &result[&device1_id][0];
        assert_eq!(device1_param, "param1", "Device1 should be assigned param1");
        assert_eq!(
            device1_model, "model2",
            "Device1 should be assigned model2 (swapped from device2)"
        );

        // 验证device2的分配：param2 -> model1
        let (device2_param, device2_model) = &result[&device2_id][0];
        assert_eq!(device2_param, "param2", "Device2 should be assigned param2");
        assert_eq!(
            device2_model, "model1",
            "Device2 should be assigned model1 (swapped from device1)"
        );
    }
}

/// 测试VirtualComputeNode的handle_change_state函数（集成测试）
#[cfg(test)]
mod test_handle_change_state_integration {
    use super::*;
    use crate::compute_node::r#virtual::virtual_compute_node::tests::MockResponse;

    // 由于handle_change_state是一个复杂的异步方法，需要完整的VirtualComputeNode实例
    // 这里我们测试其主要的错误条件和边界情况

    #[test]
    fn test_build_target_device_params() {
        use crate::compute_node::r#virtual::virtual_compute_node::VirtualComputeNode;

        // let _devices = create_test_devices(2);
        let state = State {
            model_allocation: HashMap::new(),
            slices: vec![
                vec!["param1".to_string(), "param2".to_string()],
                vec!["param3".to_string()],
            ],
        };

        let target_params =
            VirtualComputeNode::<String, MockResponse>::build_target_device_params(&state, 2);
        assert_eq!(target_params.len(), 2);

        // 验证切片内容
        let mut slice_lengths: Vec<_> = target_params.values().map(|v| v.len()).collect();
        slice_lengths.sort();
        assert_eq!(slice_lengths, vec![1, 2]);
    }

    #[test]
    fn test_target_state_validation_error_conditions() {
        // 测试目标状态验证的边界条件

        // 1. 空的model_allocation但有slices
        let state_with_empty_models = State {
            model_allocation: HashMap::new(),
            slices: vec![vec!["param1".to_string()]],
        };

        let target_params = VirtualComputeNode::<String, MockResponse>::build_target_device_params(
            &state_with_empty_models,
            1,
        );
        assert_eq!(target_params.len(), 1);

        // 2. 空的slices但有model_allocation
        let mut model_allocation = HashMap::new();
        model_allocation.insert("model1".to_string(), vec!["param1".to_string()]);

        let state_with_empty_slices = State {
            model_allocation,
            slices: vec![],
        };

        let target_params = VirtualComputeNode::<String, MockResponse>::build_target_device_params(
            &state_with_empty_slices,
            0,
        );
        assert!(target_params.is_empty());
    }
}

/// 测试工具函数和边缘情况
#[cfg(test)]
mod test_edge_cases {
    use super::*;

    #[test]
    fn test_uuid_consistency() {
        // 测试UUID在整个流程中的一致性
        let device_id = Uuid::new_v4();
        let target_device_id = Uuid::new_v4();

        // 确保UUID在不同函数调用中保持一致
        let mut params1 = HashMap::new();
        params1.insert(device_id, vec!["param1".to_string()]);

        let mut params2 = HashMap::new();
        params2.insert(target_device_id, vec!["param1".to_string()]);

        let assignment = compute_optimal_device_assignment(&params1, &params2).unwrap();
        assert_eq!(assignment[0].0, device_id);
        assert_eq!(assignment[0].1, target_device_id);
    }

    #[test]
    fn test_string_parameter_handling() {
        // 测试字符串参数的各种情况
        let params = vec![
            "".to_string(),                       // 空字符串
            "normal_param".to_string(),           // 普通参数
            "param with spaces".to_string(),      // 包含空格
            "param-with-dashes".to_string(),      // 包含连字符
            "param_with_underscores".to_string(), // 包含下划线
            "UPPERCASE".to_string(),              // 大写
            "123456".to_string(),                 // 数字
            "αβγ".to_string(),                    // 非ASCII字符
        ];

        let cost = calculate_reallocation_cost(&params, &params);
        assert_eq!(cost, 0); // 相同参数应该成本为0
    }

    #[test]
    fn test_large_scale_scenario() {
        // 测试大规模场景
        let num_devices = 100;
        let num_params_per_device = 10;

        let mut current_params = HashMap::new();
        let mut target_params = HashMap::new();

        for i in 0..num_devices {
            let device_id = Uuid::new_v4();
            let target_id = Uuid::new_v4();

            let params: Vec<String> = (0..num_params_per_device)
                .map(|j| format!("param_{i}_{j}"))
                .collect();

            current_params.insert(device_id, params.clone());
            target_params.insert(target_id, params);
        }

        // 这应该能够处理大规模场景而不出错
        let result = compute_optimal_device_assignment(&current_params, &target_params);
        assert!(result.is_ok());
        let assignment = result.unwrap();
        assert_eq!(assignment.len(), num_devices);

        // 验证总成本应该是0（完美匹配）
        let total_cost: u32 = assignment.iter().map(|(_, _, cost)| cost).sum();
        assert_eq!(total_cost, 0);
    }

    #[test]
    fn test_memory_efficiency() {
        // 测试内存使用效率 - 确保函数不会泄漏或过度分配
        let device_id = Uuid::new_v4();
        let large_param_count = 1000;

        let _large_params: Vec<String> = (0..large_param_count)
            .map(|i| format!("param_{i}"))
            .collect();

        let mut allocation = HashMap::new();
        allocation.insert(device_id, (device_id, vec![])); // 空的worker列表

        // 多次调用应该不会导致内存问题
        for _ in 0..100 {
            let result = extract_device_share_params::<String>(&allocation);
            assert_eq!(result.len(), 1);
            assert!(result[&device_id].is_empty());
        }
    }

    #[test]
    fn test_concurrent_uuid_usage() {
        // 测试并发UUID使用的场景
        let device_ids: Vec<Uuid> = (0..10).map(|_| Uuid::new_v4()).collect();

        // 确保所有UUID都是唯一的
        for i in 0..device_ids.len() {
            for j in i + 1..device_ids.len() {
                assert_ne!(device_ids[i], device_ids[j]);
            }
        }

        // 在多个操作中使用这些UUID
        let mut params = HashMap::new();
        for (i, &device_id) in device_ids.iter().enumerate() {
            params.insert(device_id, vec![format!("param_{}", i)]);
        }

        let extracted = extract_device_share_params(
            &device_ids
                .iter()
                .enumerate()
                .map(|(i, &device_id)| {
                    (
                        device_id,
                        (
                            device_id,
                            vec![(
                                Uuid::new_v4(),
                                format!("param_{i}"),
                                Some(format!("model_{i}")),
                            )],
                        ),
                    )
                })
                .collect::<HashMap<_, _>>(),
        );

        assert_eq!(extracted.len(), device_ids.len());
    }
}

/// 性能和压力测试
#[cfg(test)]
mod test_performance {
    use super::*;

    #[test]
    fn test_optimal_assignment_performance() {
        // 测试匈牙利算法的性能
        let device_count = 50; // 适中的规模以避免测试超时

        let mut current_params = HashMap::new();
        let mut target_params = HashMap::new();

        // 创建最坏情况：所有设备都需要完全重新分配
        for i in 0..device_count {
            let current_device = Uuid::new_v4();
            let target_device = Uuid::new_v4();

            current_params.insert(current_device, vec![format!("current_param_{}", i)]);
            target_params.insert(target_device, vec![format!("target_param_{}", i)]);
        }

        let start = std::time::Instant::now();
        let result = compute_optimal_device_assignment(&current_params, &target_params);
        let duration = start.elapsed();

        assert!(result.is_ok());
        assert!(duration.as_millis() < 1000); // 应该在1秒内完成
    }

    #[test]
    fn test_command_generation_performance() {
        // 测试命令生成的性能
        let worker_count = 1000;
        let device_id = Uuid::new_v4();

        let current_allocs: Vec<_> = (0..worker_count)
            .map(|i| {
                (
                    Uuid::new_v4(),
                    format!("param_{i}"),
                    Some(format!("model_{i}")),
                )
            })
            .collect();

        let target_allocs: Vec<_> = (0..worker_count)
            .map(|i| (format!("param_{i}"), format!("new_model_{i}")))
            .collect();

        let start = std::time::Instant::now();
        let commands = compare_and_generate_commands(&current_allocs, &target_allocs, device_id);
        let duration = start.elapsed();

        // 验证命令数量和类型
        assert_eq!(
            commands.len(),
            worker_count * 2,
            "Should have 2 commands per worker"
        );

        let unload_count = commands
            .iter()
            .filter(|cmd| matches!(cmd, BaseCommand::UnloadModel { .. }))
            .count();
        let load_count = commands
            .iter()
            .filter(|cmd| matches!(cmd, BaseCommand::LoadModel { .. }))
            .count();

        assert_eq!(
            unload_count, worker_count,
            "Should have one unload command per worker"
        );
        assert_eq!(
            load_count, worker_count,
            "Should have one load command per worker"
        );

        // 验证命令配对正确性（采样检查前10个worker）
        for (i, (worker_id, _, _)) in current_allocs.iter().enumerate().take(10.min(worker_count)) {
            let expected_model = format!("new_model_{i}");

            // 找到对应的unload和load命令
            let unload_found = commands
                .iter()
                .any(|cmd| matches!(cmd, BaseCommand::UnloadModel { id } if id == worker_id));

            let load_found = commands.iter().any(|cmd| {
                matches!(cmd, BaseCommand::LoadModel { id, name } if id == worker_id && name == &expected_model)
            });

            assert!(unload_found, "Should have unload command for worker {i}");
            assert!(
                load_found,
                "Should have load command for worker {i} with model {expected_model}"
            );
        }

        assert!(
            duration.as_millis() < 100,
            "Performance test: should complete in <100ms, took {}ms",
            duration.as_millis()
        );
    }
}

/// 错误处理和恢复测试
#[cfg(test)]
mod test_error_handling {
    use super::*;

    #[test]
    fn test_assignment_with_invalid_data() {
        // 测试无效数据的处理
        let mut current_params = HashMap::new();
        let mut target_params = HashMap::new();

        // 插入重复的设备ID（不应该发生，但测试健壮性）
        let device_id = Uuid::new_v4();
        current_params.insert(device_id, vec!["param1".to_string()]);
        current_params.insert(device_id, vec!["param2".to_string()]); // 覆盖前一个

        target_params.insert(device_id, vec!["param2".to_string()]);

        let result = compute_optimal_device_assignment(&current_params, &target_params);
        assert!(result.is_ok());
    }

    #[test]
    fn test_command_generation_with_invalid_workers() {
        // 测试无效worker数据的处理
        let device_id = Uuid::new_v4();

        let current_allocs = vec![
            (Uuid::new_v4(), "param1".to_string(), None), // 没有模型的worker
            (
                Uuid::new_v4(),
                "param2".to_string(),
                Some("model1".to_string()),
            ),
        ];

        let target_allocs = vec![("param1".to_string(), "model2".to_string())];

        let commands = compare_and_generate_commands(&current_allocs, &target_allocs, device_id);

        // 应该能够处理没有模型的worker
        assert!(!commands.is_empty());
    }

    #[test]
    fn test_empty_device_allocation_extraction() {
        // 测试空设备分配的提取
        let empty_allocation: HashMap<Uuid, (Uuid, Vec<(Uuid, String, Option<String>)>)> =
            HashMap::new();

        let result = extract_device_share_params(&empty_allocation);
        assert!(result.is_empty());
    }

    #[test]
    fn test_allocation_consistency() {
        // 测试分配的一致性检查
        let device_id = Uuid::new_v4();
        let node_id = Uuid::new_v4();

        let mut current_allocation = HashMap::new();
        current_allocation.insert(device_id, (node_id, vec![]));

        let mut target_model_allocation = HashMap::new();
        target_model_allocation.insert("model1".to_string(), vec!["param1".to_string()]);

        let optimal_assignment = vec![(device_id, device_id, 0)];

        let mut target_device_params = HashMap::new();
        target_device_params.insert(device_id, vec!["param1".to_string()]);

        let result = build_assignment_pairs(
            &current_allocation,
            &target_model_allocation,
            &optimal_assignment,
            &target_device_params,
        );

        assert_eq!(
            result.len(),
            1,
            "Should have assignments for exactly 1 device"
        );
        assert_eq!(
            result[&device_id].len(),
            1,
            "Device should have exactly 1 assignment"
        );

        // 验证分配的具体内容
        let (assigned_param, assigned_model) = &result[&device_id][0];
        assert_eq!(assigned_param, "param1", "Should assign param1");
        assert_eq!(assigned_model, "model1", "Should assign model1");
    }
}

/// 数据完整性测试
#[cfg(test)]
mod test_data_integrity {
    use super::*;

    #[test]
    fn test_parameter_preservation() {
        // 测试参数在整个流程中的保持
        let original_params = vec![
            "param1".to_string(),
            "param2".to_string(),
            "param3".to_string(),
        ];

        let device_id = Uuid::new_v4();
        let mut allocation = HashMap::new();
        allocation.insert(
            device_id,
            (
                device_id,
                original_params
                    .iter()
                    .enumerate()
                    .map(|(i, param)| (Uuid::new_v4(), param.clone(), Some(format!("model_{i}"))))
                    .collect(),
            ),
        );

        let extracted = extract_device_share_params(&allocation);
        let extracted_params = &extracted[&device_id];

        // 验证所有原始参数都被保留
        for param in &original_params {
            assert!(extracted_params.contains(param));
        }

        assert_eq!(extracted_params.len(), original_params.len());
    }

    #[test]
    fn test_model_allocation_consistency() {
        // 测试模型分配的一致性
        let device_id = Uuid::new_v4();
        let node_id = Uuid::new_v4();

        let mut current_allocation = HashMap::new();
        current_allocation.insert(
            device_id,
            (
                node_id,
                vec![
                    (
                        Uuid::new_v4(),
                        "param1".to_string(),
                        Some("model1".to_string()),
                    ),
                    (
                        Uuid::new_v4(),
                        "param2".to_string(),
                        Some("model1".to_string()),
                    ),
                ],
            ),
        );

        let mut target_model_allocation = HashMap::new();
        target_model_allocation.insert(
            "model1".to_string(),
            vec!["param1".to_string(), "param2".to_string()],
        );

        let optimal_assignment = vec![(device_id, device_id, 0)];

        let mut target_device_params = HashMap::new();
        target_device_params.insert(device_id, vec!["param1".to_string(), "param2".to_string()]);

        let result = build_assignment_pairs(
            &current_allocation,
            &target_model_allocation,
            &optimal_assignment,
            &target_device_params,
        );

        // 验证分配结果的一致性
        assert_eq!(
            result.len(),
            1,
            "Should have assignments for exactly 1 device"
        );
        let assignments = &result[&device_id];
        assert_eq!(assignments.len(), 2, "Should have exactly 2 assignments");

        // 验证具体的分配内容
        let param1_assignment = assignments
            .iter()
            .find(|(param, _)| param == "param1")
            .expect("Should have assignment for param1");
        let param2_assignment = assignments
            .iter()
            .find(|(param, _)| param == "param2")
            .expect("Should have assignment for param2");

        assert_eq!(
            param1_assignment.1, "model1",
            "param1 should be assigned model1"
        );
        assert_eq!(
            param2_assignment.1, "model1",
            "param2 should be assigned model1"
        );
    }

    #[test]
    fn test_worker_id_uniqueness() {
        // 测试Worker ID的唯一性
        let device_id = Uuid::new_v4();

        let current_allocs = vec![];
        let target_allocs = vec![
            ("param1".to_string(), "model1".to_string()),
            ("param2".to_string(), "model2".to_string()),
        ];

        let commands = compare_and_generate_commands(&current_allocs, &target_allocs, device_id);

        // 验证创建的Worker命令并提取ID
        let mut worker_ids = Vec::new();
        let mut create_commands = Vec::new();
        let mut load_commands = Vec::new();

        for command in &commands {
            match command {
                BaseCommand::CreateWorker {
                    id,
                    device_id: cmd_device_id,
                    params,
                } => {
                    worker_ids.push(*id);
                    create_commands.push((*id, *cmd_device_id, params.clone()));
                }
                BaseCommand::LoadModel { id, name } => {
                    load_commands.push((*id, name.clone()));
                }
                _ => {}
            }
        }

        assert_eq!(create_commands.len(), 2, "Should create exactly 2 workers");
        assert_eq!(
            load_commands.len(),
            2,
            "Should have exactly 2 load commands"
        );

        // 验证所有Worker ID都是唯一的
        assert_ne!(worker_ids[0], worker_ids[1], "Worker IDs should be unique");

        // 验证create命令的具体内容
        for (_worker_id, cmd_device_id, params) in &create_commands {
            assert_eq!(
                *cmd_device_id, device_id,
                "CreateWorker should target correct device"
            );
            assert!(
                params == "param1" || params == "param2",
                "CreateWorker should use correct params, got: {params}"
            );
        }

        // 验证load命令对应正确的worker
        for (load_worker_id, model_name) in &load_commands {
            assert!(
                worker_ids.contains(load_worker_id),
                "LoadModel should target a created worker"
            );
            assert!(
                model_name == "model1" || model_name == "model2",
                "LoadModel should load correct model, got: {model_name}"
            );
        }
    }
}

#[test]
fn regrouped_plan_matches_all_devices_and_cleans_the_idle_one() {
    use crate::compute_node::r#virtual::virtual_compute_node::tests::MockResponse;
    let node = Uuid::new_v4();
    let current = (0..8)
        .map(|_| {
            (
                Uuid::new_v4(),
                (
                    node,
                    vec![(
                        Uuid::new_v4(),
                        "tiny".to_string(),
                        Some("model".to_string()),
                    )],
                ),
            )
        })
        .collect::<HashMap<_, _>>();
    let state = State {
        model_allocation: HashMap::from([("model".into(), vec!["tiny".to_string(); 7])]),
        slices: vec![vec!["tiny".to_string()]; 7],
    };
    let targets = VirtualComputeNode::<String, MockResponse>::build_target_device_params(
        &state,
        current.len(),
    );
    assert_eq!(targets.len(), 8);
    assert_eq!(
        targets.values().filter(|params| params.is_empty()).count(),
        1
    );
    let assignment =
        compute_optimal_device_assignment(&extract_device_share_params(&current), &targets)
            .unwrap();
    assert_eq!(assignment.len(), 8);
    let pairs = build_assignment_pairs(&current, &state.model_allocation, &assignment, &targets);
    let commands = super::generate_migration_commands(&pairs, &current);
    assert_eq!(commands.len(), 1);
    assert_eq!(commands[0].1.len(), 1);
    assert!(matches!(
        commands[0].1[0],
        BaseCommand::DestroyWorker { .. }
    ));
}
