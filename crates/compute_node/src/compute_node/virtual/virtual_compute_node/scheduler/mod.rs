use std::{collections::HashMap, fmt::Debug, hash::Hash, sync::Arc};

use bincode::{Decode, Encode};
use futures::future::{join_all, try_join_all};
use pathfinding::{kuhn_munkres::kuhn_munkres_min, matrix::Matrix};
use serde::{Serialize, de::DeserializeOwned};
use tokio::time::Duration;
use tracing::{debug, error, trace};
use utils::IntoIteratorExt;
use uuid::Uuid;

use crate::{
    compute_device::vgpu::share_target::ShareTarget,
    compute_node::r#virtual::{
        base,
        base::{Command as BaseCommand, Command},
        virtual_compute_node::{
            MigrationCommand, State, VirtualComputeNode,
            base_virtual_compute_node_ext::DynBaseVirtualComputeNode,
            components::{NodeId, WorkerId, worker_registry::WorkerStatus},
        },
    },
};

#[cfg(test)]
mod tests;
mod utils;

// 新增类型定义
type DeviceAllocationMap<ShareParam> =
    HashMap<Uuid, (NodeId, Vec<(WorkerId, ShareParam, Option<String>)>)>;
type DeviceShareParamMap<ShareParam> = HashMap<Uuid, Vec<ShareParam>>;

impl<
    ShareParam: ShareTarget
        + Serialize
        + DeserializeOwned
        + Debug
        + Send
        + Sync
        + Encode
        + Decode<()>
        + Ord
        + Clone
        + Hash
        + 'static,
    DeviceState: base::DeviceState + Encode + Decode<()> + Send + Sync + 'static,
> VirtualComputeNode<ShareParam, DeviceState>
{
    /// 处理状态变更命令
    pub(super) async fn handle_change_state(
        &self,
        target_state: State<ShareParam>,
    ) -> Result<State<ShareParam>, String> {
        // 步骤1: 获取当前设备状态
        let current_device_allocation = self.get_current_device_allocation().await?;

        if target_state.slices.is_empty() || target_state.model_allocation.is_empty() {
            return Err("Empty allocation is not supported".to_string());
        }

        // 验证设备数量
        if target_state.slices.len() > current_device_allocation.len() {
            error!(?target_state, "Error change state");
            return Err(format!(
                "Target state requires {} devices but only {} are available",
                target_state.slices.len(),
                current_device_allocation.len()
            ));
        }

        // 步骤2: 计算最优设备分配
        let current_device_params = extract_device_share_params(&current_device_allocation);
        let target_device_params =
            Self::build_target_device_params(&target_state, current_device_allocation.len());
        let optimal_assignment =
            compute_optimal_device_assignment(&current_device_params, &target_device_params)?;

        // 步骤3: 构建模型分配策略
        let assignment_pairs = build_assignment_pairs(
            &current_device_allocation,
            &target_state.model_allocation,
            &optimal_assignment,
            &target_device_params,
        );

        debug!("Assignment pairs: {:?}", assignment_pairs);

        // 步骤4: 生成迁移命令
        let migration_commands =
            generate_migration_commands(&assignment_pairs, &current_device_allocation);

        debug!("Migration commands: {:?}", migration_commands);

        // 步骤5: 执行迁移计划（现在每个命令会自动处理worker状态）
        let execution_result = self
            .execute_migration_plan_ordered(migration_commands)
            .await;

        // 步骤6: 处理执行结果
        if let Err(failed_commands) = execution_result {
            return Err(format!(
                "Failed to execute migration plan: {:?}",
                failed_commands
            ));
        }

        // 返回最新状态 - 不再需要同步worker状态，因为每个命令都已经同步了状态
        Ok(self.get_state())
    }

    /// 步骤1: 获取当前设备分配状态
    /// 返回 HashMap<DeviceId, (NodeId, Vec<(ShareParam, ModelId)>)>
    async fn get_current_device_allocation(
        &self,
    ) -> Result<DeviceAllocationMap<ShareParam>, String> {
        let mut result = HashMap::new();

        // 并行获取所有节点的状态和设备信息
        let node_futures: Vec<_> = self
            .worker_manager
            .nodes
            .iter()
            .map(|entry| {
                let node_id = *entry.key();
                let node = entry.value().clone();

                async move {
                    let node_state = node
                        .state()
                        .await
                        .map_err(|e| format!("Failed to get state from node {node_id}: {e}"))?;
                    let devices = node.status().await?.devices;
                    Ok::<_, String>((node_id, node_state, devices))
                }
            })
            .collect();

        let node_results = try_join_all(node_futures).await?;

        // 处理并行获取的结果
        for (node_id, node_state, devices) in node_results {
            // 初始化设备映射
            for device_id in devices.into_keys() {
                result
                    .entry(device_id)
                    .or_insert_with(|| (node_id, Vec::new()));
            }

            // 按设备分组工作者，收集 ShareParam 和 ModelId
            for (worker_id, worker_state) in node_state.workers {
                result
                    .entry(worker_state.device_id)
                    .or_insert_with(|| (node_id, Vec::new()))
                    .1
                    .push((worker_id, worker_state.params, worker_state.model_name));
            }
        }

        Ok(result)
    }

    /// 步骤2b: 构建目标设备参数映射
    fn build_target_device_params(
        target_state: &State<ShareParam>,
        device_count: usize,
    ) -> DeviceShareParamMap<ShareParam> {
        target_state
            .slices
            .iter()
            .cloned()
            .chain(
                std::iter::repeat_with(Vec::new)
                    .take(device_count.saturating_sub(target_state.slices.len())),
            )
            .map(|slice| (Uuid::new_v4(), slice))
            .collect()
    }

    /// 执行单个命令并立即同步状态，带细粒度Worker状态控制
    async fn execute_command_with_sync(
        &self,
        node_id: Uuid,
        command: BaseCommand<ShareParam>,
    ) -> Result<(), String> {
        // 1. 获取节点
        let node = self
            .worker_manager
            .nodes
            .get(&node_id)
            .ok_or_else(|| format!("Node {} not found", node_id))?
            .value()
            .clone();

        // Observe drain separately from model/slice operations without changing
        // the migration order or the set of affected workers.
        let drain_started = std::time::Instant::now();
        let draining_id = match &command {
            BaseCommand::UnloadModel { id } | BaseCommand::DestroyWorker { id } => Some(*id),
            _ => None,
        };
        if let Some(id) = draining_id {
            tracing::info!(target: "allocation", event = "worker_drain_started", %id, %node_id);
        }
        // 2. 在执行命令前暂停相关worker
        match self.pause_worker_before_command(&command).await {
            Ok(_) => {}
            Err(_) => {
                if let Some(id) = draining_id {
                    tracing::info!(target: "allocation", event = "worker_drain_finished", %id, %node_id, success = false, elapsed_seconds = drain_started.elapsed().as_secs_f64());
                }
                return Err("Wait worker to pause timeout".to_string());
            }
        };

        if let Some(id) = draining_id {
            tracing::info!(target: "allocation", event = "worker_drain_finished", %id, %node_id, success = true, elapsed_seconds = drain_started.elapsed().as_secs_f64());
        }

        // 3. 执行命令（统一重试逻辑）
        let command_result = self
            .execute_command_with_retry(&node, command.clone(), node_id)
            .await;

        // A successful command already returns the authoritative state.
        // An uncertain outcome must not make a paused worker available again.
        match command_result {
            Ok(state) => {
                self.sync_worker_after_command(node_id, &command, &state)
                    .await
            }
            Err(error) => {
                if let Ok(state) = node.state().await {
                    self.sync_worker_after_command(node_id, &command, &state)
                        .await
                        .map_err(|sync_error| format!("{error}; {sync_error}"))
                } else {
                    Err(error)
                }
            }
        }
    }

    /// 统一的命令重试执行逻辑
    async fn execute_command_with_retry(
        &self,
        node: &Arc<
            dyn DynBaseVirtualComputeNode<ShareParam = ShareParam, DeviceState = DeviceState>,
        >,
        command: BaseCommand<ShareParam>,
        node_id: Uuid,
    ) -> Result<base::State<ShareParam>, String> {
        let mut last_error = None;

        for attempt in 0..=3 {
            match node.command(command.clone()).await {
                Ok(state) => {
                    if attempt > 0 {
                        trace!(
                            ?command,
                            "Command succeeded on attempt {} on node {}",
                            attempt + 1,
                            node_id
                        );
                    }
                    return Ok(state);
                }
                Err(e) => {
                    last_error = Some(e);
                    if attempt < 3 {
                        trace!(
                            ?command,
                            "Command failed on attempt {} on node {}, retrying...",
                            attempt + 1,
                            node_id
                        );
                        tokio::time::sleep(Duration::from_millis(100 * (attempt + 1) as u64)).await;
                    }
                }
            }
        }

        Err(format!(
            "Command failed after 4 attempts on node {}: {:?}",
            node_id, last_error
        ))
    }

    /// 在执行命令前暂停相关worker，返回被暂停的worker ID
    async fn pause_worker_before_command(
        &self,
        command: &BaseCommand<ShareParam>,
    ) -> Result<Option<Uuid>, ()> {
        match command {
            BaseCommand::UnloadModel { id } | BaseCommand::DestroyWorker { id } => {
                // 检查worker是否存在且正在运行
                if let Some(info) = self.worker_manager.get_registry().get_worker_info(*id)
                    && info.status == WorkerStatus::Running
                {
                    self.worker_manager
                        .get_registry()
                        .update_worker_status(*id, WorkerStatus::Pausing);
                    let worker_id = *id;

                    trace!(
                        "Set worker {} to Pausing state before executing command",
                        id
                    );

                    // 等待worker状态变为Paused
                    let timeout = Some(Duration::from_secs(60));
                    return match self
                        .worker_manager
                        .get_registry()
                        .get_status_listener(worker_id)
                        .wait_for_worker_status(WorkerStatus::Paused, timeout)
                        .await
                    {
                        Ok(()) => {
                            trace!("Worker {} successfully paused", id);
                            Ok(Some(*id))
                        }
                        Err(e) => {
                            tracing::warn!("Failed to wait for worker {} to pause: {}", id, e);
                            // 恢复worker状态到Running
                            self.worker_manager
                                .get_registry()
                                .update_worker_status(*id, WorkerStatus::Running);
                            Err(())
                        }
                    };
                }
            }
            BaseCommand::CreateWorker { .. } | BaseCommand::LoadModel { .. } => {
                // 这些命令不需要暂停现有worker
            }
        }
        Ok(None)
    }

    /// Reconcile only a state confirmed by the node, never the old registry.
    async fn sync_worker_after_command(
        &self,
        node_id: Uuid,
        command: &BaseCommand<ShareParam>,
        state: &base::State<ShareParam>,
    ) -> Result<(), String> {
        match command {
            BaseCommand::CreateWorker { id, .. } => {
                if !state.workers.contains_key(id) {
                    return Err(format!("Worker {id} not found after creation"));
                }
            }
            BaseCommand::LoadModel { id, name } => {
                let worker = state
                    .workers
                    .get(id)
                    .ok_or_else(|| format!("Worker {id} not found after model load"))?;
                if worker.model_name.as_ref() != Some(name) {
                    return Err(format!("Worker {id} model mismatch after load"));
                }
                self.add_worker_with_noise_slot(
                    *id,
                    name.clone(),
                    node_id,
                    worker.device_id,
                    worker.params.clone(),
                    worker.experiment_noise_slot.as_deref(),
                )
                .map_err(|e| format!("Failed to register worker: {e}"))?;
            }
            BaseCommand::UnloadModel { id } => {
                let worker = state
                    .workers
                    .get(id)
                    .ok_or_else(|| format!("Worker {id} not found after model unload"))?;
                if worker.model_name.is_some() {
                    return Err(format!("Worker {id} still has a model loaded"));
                }
                if self
                    .worker_manager
                    .get_registry()
                    .get_worker_info(*id)
                    .is_some()
                {
                    self.remove_worker(*id)
                        .await
                        .map_err(|e| format!("Failed to unregister worker: {e}"))?;
                }
            }
            BaseCommand::DestroyWorker { id } => {
                if state.workers.contains_key(id) {
                    return Err(format!("Worker {id} still exists after destruction"));
                }
                if self
                    .worker_manager
                    .get_registry()
                    .get_worker_info(*id)
                    .is_some()
                {
                    self.remove_worker(*id)
                        .await
                        .map_err(|e| format!("Failed to unregister worker: {e}"))?;
                }
            }
        }
        Ok(())
    }

    /// 从命令中提取device_id
    fn extract_device_id(
        &self,
        command: &BaseCommand<ShareParam>,
        pending_workers: &mut HashMap<Uuid, Uuid>,
    ) -> Uuid {
        match command {
            BaseCommand::CreateWorker { device_id, id, .. } => {
                // 记录新创建的worker到pending_workers中
                pending_workers.insert(*id, *device_id);
                *device_id
            }
            BaseCommand::LoadModel { id, .. }
            | BaseCommand::UnloadModel { id }
            | BaseCommand::DestroyWorker { id } => {
                // 首先尝试从worker注册信息中获取device_id
                if let Some(info) = self.worker_manager.get_registry().get_worker_info(*id) {
                    return info.device_id;
                }

                // 如果在注册信息中找不到，则从pending_workers中查找
                if let Some(&device_id) = pending_workers.get(id) {
                    return device_id;
                }

                // 如果两个地方都找不到，说明逻辑有问题，panic
                panic!(
                    "Worker {} not found in both registry and pending_workers",
                    id
                );
            }
        }
    }

    /// 改进的迁移计划执行 - 按device顺序执行，每个命令完成后立即同步
    async fn execute_migration_plan_ordered(
        &self,
        migration_plan: Vec<MigrationCommand<ShareParam>>,
    ) -> Result<(), Vec<MigrationCommand<ShareParam>>> {
        // 按device_id分组命令，确保同一device上的命令顺序执行
        let mut device_commands: HashMap<(Uuid, Uuid), Vec<BaseCommand<ShareParam>>> =
            HashMap::new();

        // 维护pending_workers映射，记录新创建的worker和其device_id的关系
        let mut pending_workers: HashMap<Uuid, Uuid> = HashMap::new();

        for (node_id, commands) in migration_plan {
            for command in commands {
                let device_id = self.extract_device_id(&command, &mut pending_workers);
                device_commands
                    .entry((node_id, device_id))
                    .or_default()
                    .push(command);
            }
        }

        // 并发执行不同device上的命令序列
        let tasks: Vec<_> = device_commands
            .into_iter()
            .map(|((node_id, device_id), commands)| {
                async move {
                    let mut failed_commands = Vec::new();

                    // 同一device上的命令必须顺序执行
                    let mut commands = commands.into_iter();
                    while let Some(command) = commands.next() {
                        if let Err(e) = self
                            .execute_command_with_sync(node_id, command.clone())
                            .await
                        {
                            trace!(
                                "Command failed on device {} of node {}: {}",
                                device_id, node_id, e
                            );
                            failed_commands.push(command);
                            // Later commands on this GPU depend on the successful
                            // prefix (e.g. pause/unload before loading another model).
                            // Keep them unapplied for a fresh plan; other GPUs finish
                            // independently through join_all.
                            failed_commands.extend(commands);
                            break;
                        }
                    }

                    if failed_commands.is_empty() {
                        Ok(())
                    } else {
                        Err((node_id, failed_commands))
                    }
                }
            })
            .collect();

        // 等待所有device的命令序列完成
        let results = join_all(tasks).await;
        let errors: Vec<_> = results
            .into_iter()
            .filter_map(|result| result.err())
            .collect();

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// 步骤2a: 从当前设备状态中提取 HashMap<DeviceId, Vec<ShareParam>>
fn extract_device_share_params<ShareParam: Clone>(
    current_allocation: &DeviceAllocationMap<ShareParam>,
) -> DeviceShareParamMap<ShareParam> {
    current_allocation
        .iter()
        .map(|(device_id, (_, allocations))| {
            let share_params = allocations
                .iter()
                .map(|(_, share_param, _)| share_param.clone())
                .collect();
            (*device_id, share_params)
        })
        .collect()
}

/// 步骤2c: 使用匈牙利算法计算最优设备分配
fn compute_optimal_device_assignment<ShareParam: Clone + Eq + Hash + Send + Sync + Debug>(
    current_params: &DeviceShareParamMap<ShareParam>,
    target_params: &DeviceShareParamMap<ShareParam>,
) -> Result<Vec<(Uuid, Uuid, u32)>, String> {
    let current_devices: Vec<Uuid> = current_params.keys().cloned().collect();
    let target_devices: Vec<Uuid> = target_params.keys().cloned().collect();

    if target_devices.len() != current_devices.len() {
        error!(
            ?current_params,
            ?target_params,
            "Unexpect allocation request",
        );
        return Err(format!(
            "Device matching requires equal device counts: target {}, current {}",
            target_devices.len(),
            current_devices.len()
        ));
    }

    if current_devices.is_empty() {
        return Ok(Vec::new());
    }

    // 创建成本矩阵：成本 = 需要重新分配的 ShareParam 数量
    let mut cost_matrix = Matrix::new(current_devices.len(), target_devices.len(), i32::MAX / 2);

    // 并行计算成本矩阵
    use rayon::prelude::*;
    let costs: Vec<_> = current_devices
        .par_iter()
        .enumerate()
        .flat_map(|(i, current_device)| {
            let current_params_set = current_params.get(current_device).unwrap();

            target_devices
                .par_iter()
                .enumerate()
                .map(move |(j, target_device)| {
                    let target_params_set = target_params.get(target_device).unwrap();

                    // 计算重新分配成本：不匹配的参数数量
                    let cost = calculate_reallocation_cost(current_params_set, target_params_set);
                    (i, j, cost)
                })
        })
        .collect();

    // 填充成本矩阵
    for (i, j, cost) in costs {
        cost_matrix[(i, j)] = cost;
    }

    // 使用匈牙利算法求解
    let (_, assignment) = kuhn_munkres_min(&cost_matrix);

    // 构建分配结果
    let mut result = Vec::new();
    for (current_idx, target_idx) in assignment.into_iter().enumerate() {
        if target_idx < target_devices.len() {
            result.push((
                current_devices[current_idx],
                target_devices[target_idx],
                cost_matrix[(current_idx, target_idx)] as u32,
            ));
        }
    }

    Ok(result)
}

/// 计算重新分配成本
fn calculate_reallocation_cost<ShareParam: Clone + Eq + Hash>(
    current_params: &[ShareParam],
    target_params: &[ShareParam],
) -> i32 {
    let current_params = current_params.into_map_count();
    let target_params = target_params.into_map_count();

    current_params
        .into_iter()
        .map(|(param, count)| (param, count as i32))
        .chain(
            target_params
                .into_iter()
                .map(|(param, count)| (param, -(count as i32))),
        )
        .fold(HashMap::new(), |mut acc, (param, count)| {
            *acc.entry(param).or_insert(0) += count;
            acc
        })
        .values()
        .map(|&diff| diff.abs())
        .sum()
}

/// 步骤3: 构建分配对，优先保留现有模型
fn build_assignment_pairs<ShareParam: Clone + Eq + Hash>(
    current_allocation: &DeviceAllocationMap<ShareParam>,
    target_model_allocation: &HashMap<String, Vec<ShareParam>>,
    optimal_assignment: &[(Uuid, Uuid, u32)],
    target_device_params: &DeviceShareParamMap<ShareParam>,
) -> HashMap<Uuid, Vec<(ShareParam, String)>> {
    let mut assignment_pairs = HashMap::new();

    // 初始化模型分配计数器
    let mut remaining_model_allocation = target_model_allocation
        .clone()
        .into_iter()
        .map(|(model, params)| (model, params.into_map_count()))
        .collect::<HashMap<_, _>>();

    // 收集所有设备的分配信息，用于两步分配策略
    let mut all_device_assignments: Vec<_> = optimal_assignment
        .iter()
        .map(|(current_device_id, target_device_id, cost)| {
            let current_allocations = current_allocation
                .get(current_device_id)
                .map(|(_, allocs)| allocs)
                .unwrap();

            let target_slice_params = target_device_params.get(target_device_id).unwrap();

            (
                *cost,
                *current_device_id,
                current_allocations,
                target_slice_params,
            )
        })
        .collect();

    // 按成本排序，优先处理成本低的分配
    all_device_assignments.sort_by_key(|(cost, _, _, _)| *cost);

    // 第一步：为所有设备的目标ShareParam寻找相同的ShareParam并直接分配
    let mut device_target_allocations = HashMap::new();

    for (_cost, current_device_id, current_allocations, target_slice_params) in
        &all_device_assignments
    {
        let mut target_allocations = Vec::with_capacity(target_slice_params.len());
        let mut current_allocations = (*current_allocations).clone();

        for target_param in *target_slice_params {
            // 寻找当前设备中是否有相同的ShareParam
            if let Some(pos) = current_allocations
                .iter()
                .position(|(_, param, model)| param == target_param && model.is_some())
            {
                let model_name = &current_allocations[pos].2.as_ref().unwrap();
                // 检查该模型是否还有可用分配
                if let Some(model_params) = remaining_model_allocation.get_mut(*model_name)
                    && let Some(count) = model_params.get_mut(target_param)
                    && *count > 0
                {
                    *count -= 1;
                    if *count == 0 {
                        model_params.remove(target_param);
                    }
                    let (_, _, model_name) = current_allocations.remove(pos);
                    target_allocations.push((target_param.clone(), model_name));
                    continue;
                }
            }
            // 如果没有找到相同的ShareParam，暂时标记为未分配
            target_allocations.push((target_param.clone(), None));
        }

        device_target_allocations.insert(*current_device_id, target_allocations);
    }

    // 第二步：为所有未分配的ShareParam，按照AssignmentPair的cost顺序分配剩余模型
    for (_cost, current_device_id, _current_allocations, _target_slice_params) in
        all_device_assignments
    {
        let mut target_allocations = device_target_allocations
            .remove(&current_device_id)
            .unwrap();

        // 为未分配的ShareParam分配模型
        for (param, model_opt) in &mut target_allocations {
            if model_opt.is_none() {
                let model_name = allocate_model_for_param(param, &mut remaining_model_allocation);
                *model_opt = Some(model_name);
            }
        }

        // 转换为最终格式
        let final_target_allocations: Vec<_> = target_allocations
            .into_iter()
            .map(|(param, model_opt)| (param, model_opt.unwrap()))
            .collect();

        assignment_pairs.insert(current_device_id, final_target_allocations);
    }

    assignment_pairs
}

/// 为指定参数分配模型，并更新剩余分配计数
fn allocate_model_for_param<ShareParam: Eq + Hash>(
    param: &ShareParam,
    remaining_allocation: &mut HashMap<String, HashMap<ShareParam, usize>>,
) -> String {
    // Interchangeable empty slots must bind models consistently in experiments.
    // Keep the production model-placement rule unchanged outside this mode.
    if crate::experiment_random::enabled() {
        let name = remaining_allocation
            .iter()
            .filter(|(_, params)| params.contains_key(param))
            .map(|(name, _)| name)
            .min()
            .expect("No Param needed for allocation")
            .clone();
        let params = remaining_allocation.get_mut(&name).unwrap();
        let count = params.get_mut(param).unwrap();
        *count -= 1;
        if *count == 0 {
            params.remove(param);
        }
        return name;
    }
    // 寻找包含此参数且还有剩余数量的模型
    for (model_name, params) in remaining_allocation.iter_mut() {
        if let Some(pos) = params.get_mut(param) {
            *pos -= 1;
            if *pos == 0 {
                params.remove(param);
            }
            return model_name.clone();
        }
    }

    panic!("No Param needed for allocation");
}

/// 步骤4: 生成迁移命令
fn generate_migration_commands<ShareParam: Debug + Clone + Eq + Hash + Send + Sync>(
    assignment_pairs: &HashMap<Uuid, Vec<(ShareParam, String)>>,
    current_device_allocation_map: &DeviceAllocationMap<ShareParam>,
) -> Vec<(NodeId, Vec<BaseCommand<ShareParam>>)> {
    use rayon::prelude::*;

    // 并行处理每个设备的命令生成
    assignment_pairs
        .par_iter()
        .filter_map(|(device_id, target_allocs)| {
            let (node_id, current_allocs) = &current_device_allocation_map[device_id];

            // 比较当前状态和目标状态，生成具体的命令
            let commands_for_device =
                compare_and_generate_commands(current_allocs, target_allocs, *device_id);

            if !commands_for_device.is_empty() {
                Some((*node_id, commands_for_device))
            } else {
                None
            }
        })
        .collect()
}

/// 比较当前和目标分配，生成具体的命令
fn compare_and_generate_commands<ShareParam: Debug + Clone + Eq + Hash>(
    current_allocs: &[(WorkerId, ShareParam, Option<String>)],
    target_allocs: &[(ShareParam, String)],
    device_id: Uuid,
) -> Vec<BaseCommand<ShareParam>> {
    let mut commands = Vec::new();
    let mut target = target_allocs.to_vec();

    let mut current_workers: HashMap<_, _> = current_allocs
        .iter()
        .map(|(id, param, model)| ((param.clone(), model.clone()), id))
        .into_map_vec();

    // 1. remove matching items, find new targets which need to be added
    target.retain(|(param, model)| {
        current_workers
            .get_mut(&(param.clone(), Some(model.clone())))
            .and_then(|ids| ids.pop())
            .is_none()
    });

    let mut reusable_workers = current_workers.into_iter().fold(
        HashMap::new(),
        |mut map: HashMap<_, Vec<_>>, ((param, model), ids)| {
            map.entry(param)
                .or_default()
                .extend(ids.into_iter().map(|id| (id, model.is_some())));
            map
        },
    );

    // 2. process the targets that need to be added, prefer reuse workers.
    for (param, model_name) in target {
        if let Some((worker_id, need_unload)) = reusable_workers
            .get_mut(&param)
            .and_then(|items| items.pop())
        {
            if need_unload {
                commands.push(Command::UnloadModel { id: *worker_id });
            }
            commands.push(Command::LoadModel {
                id: *worker_id,
                name: model_name,
            });
        } else {
            let id = Uuid::new_v4();
            commands.push(Command::CreateWorker {
                id,
                device_id,
                params: param.clone(),
            });
            commands.push(Command::LoadModel {
                id,
                name: model_name,
            });
        }
    }

    let mut destroy_commands = Vec::new();

    // 4. destroy all workers that are not reused.
    for (worker, _) in reusable_workers.into_values().flatten() {
        destroy_commands.push(Command::DestroyWorker { id: *worker });
    }

    destroy_commands.extend(commands);

    trace!(
        ?current_allocs,
        ?target_allocs,
        "Generated commands for device {device_id}: {destroy_commands:?}"
    );

    destroy_commands
}
