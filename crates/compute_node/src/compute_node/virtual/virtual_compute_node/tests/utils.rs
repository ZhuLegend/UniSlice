//! 测试工具模块
//!
//! 提供用于测试VirtualComputeNode组件的通用mock结构体和工具函数

use uuid::Uuid;

use crate::{
    compute_device::{
        ComputeDevice,
        vgpu::{
            ShareableComputeDevice, ShareableComputeDeviceResource, ShareableResource,
            SharedResource, share_target::ShareTarget,
        },
    },
    compute_node::r#virtual::virtual_compute_node::components::worker_registry::{
        WorkerInfo, WorkerStatus,
    },
};

/// Mock资源类型，用于测试可共享设备
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct MockResource {
    /// 内存大小（字节）
    pub memory: u64,
    /// 计算单元数量
    pub compute_units: u32,
}

impl Default for MockResource {
    fn default() -> Self {
        Self {
            memory: 8192,
            compute_units: 16,
        }
    }
}

impl ShareableComputeDeviceResource for MockResource {
    fn can_share(&self, other: &Self) -> bool {
        self.memory >= other.memory && self.compute_units >= other.compute_units
    }

    fn borrow(&mut self, target: &Self) -> Option<Self> {
        if self.can_share(target) {
            self.memory -= target.memory;
            self.compute_units -= target.compute_units;
            Some(*target)
        } else {
            None
        }
    }

    fn refund(&mut self, target: Self) {
        self.memory += target.memory;
        self.compute_units += target.compute_units;
    }
}

impl From<&MockResource> for MockResource {
    fn from(resource: &MockResource) -> Self {
        *resource
    }
}

impl From<&String> for MockResource {
    fn from(param: &String) -> Self {
        // 解析格式: "memory,compute_units" 或 "memory"
        let parts: Vec<&str> = param.split(',').collect();
        let memory = parts[0].parse::<u64>().unwrap_or(1024);
        let compute_units = if parts.len() > 1 {
            parts[1].parse::<u32>().unwrap_or(4)
        } else {
            4
        };

        MockResource {
            memory,
            compute_units,
        }
    }
}

/// Mock原始计算设备
#[derive(Debug, Clone)]
pub struct MockRawDevice {
    /// 设备唯一标识符
    #[allow(dead_code)]
    pub id: Uuid,
    /// 逻辑索引
    pub logic_index: Option<u32>,
}

impl Default for MockRawDevice {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4(),
            logic_index: Some(0),
        }
    }
}

impl MockRawDevice {
    /// 创建新的Mock原始设备
    #[allow(dead_code)]
    pub fn new(logic_index: u32) -> Self {
        Self {
            id: Uuid::new_v4(),
            logic_index: Some(logic_index),
        }
    }
}

/// Mock响应类型，实现DeviceState
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, bincode::Encode, bincode::Decode)]
pub struct MockResponse {
    /// 响应数据
    pub data: String,
}

impl crate::compute_node::r#virtual::base::DeviceState for MockResponse {}

impl ComputeDevice for MockRawDevice {
    type Request = String;
    type Response = MockResponse;

    async fn request(&self, request: Self::Request) -> Self::Response {
        MockResponse {
            data: format!("mock_response_for_{request}"),
        }
    }

    fn logic_index(&self) -> Option<u32> {
        self.logic_index
    }
}

/// Mock共享计算设备
#[derive(Debug)]
pub struct MockSharedDevice {
    /// 设备ID
    pub device_id: Uuid,
    /// 设备参数
    #[allow(dead_code)]
    pub params: String,
    /// 共享资源
    pub _resource: SharedResource<MockResource>,
}

impl ComputeDevice for MockSharedDevice {
    type Request = String;
    type Response = MockResponse;

    async fn request(&self, request: Self::Request) -> Self::Response {
        MockResponse {
            data: format!("mock_shared_response_{}_for_{}", self.device_id, request),
        }
    }

    fn logic_index(&self) -> Option<u32> {
        Some(1)
    }
}

impl TryFrom<(&MockDevice, String, SharedResource<MockResource>)> for MockSharedDevice {
    type Error = String;

    fn try_from(
        (device, params, resource): (&MockDevice, String, SharedResource<MockResource>),
    ) -> Result<Self, Self::Error> {
        Ok(MockSharedDevice {
            device_id: device.id,
            params,
            _resource: resource,
        })
    }
}

/// Mock可共享设备
#[derive(Debug)]
pub struct MockDevice {
    /// 设备唯一标识符
    pub id: Uuid,
    /// 原始计算设备
    pub raw_device: MockRawDevice,
    /// 可共享资源
    pub shareable_resource: ShareableResource<MockResource>,
    /// 总资源
    pub total_resource: MockResource,
}

impl Default for MockDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl ShareTarget for String {
    fn name(&self) -> &str {
        self.as_str()
    }

    fn from_name(name: &str) -> Option<Self> {
        Some(name.to_string())
    }
}

impl MockDevice {
    /// 创建新的Mock设备
    pub fn new() -> Self {
        let total_resource = MockResource::default();
        Self {
            id: Uuid::new_v4(),
            raw_device: MockRawDevice::default(),
            shareable_resource: ShareableResource::from(total_resource),
            total_resource,
        }
    }
}

impl From<MockRawDevice> for MockDevice {
    fn from(raw_device: MockRawDevice) -> Self {
        let total_resource = MockResource::default();
        Self {
            id: Uuid::new_v4(),
            raw_device,
            shareable_resource: ShareableResource::from(total_resource),
            total_resource,
        }
    }
}

impl ShareableComputeDevice for MockDevice {
    type Resource = MockResource;
    type RawComputeDevice = MockRawDevice;
    type ShareParam = String;
    type SharedComputeDevice = MockSharedDevice;

    fn raw_device(&self) -> &Self::RawComputeDevice {
        &self.raw_device
    }

    fn shareable_resource(&self) -> &ShareableResource<Self::Resource> {
        &self.shareable_resource
    }

    fn total_resources(&self) -> Self::Resource {
        self.total_resource
    }

    fn remain_resource(&self) -> Self::Resource {
        self.shareable_resource.current_resource()
    }
}

/// 创建测试用的Worker信息
pub fn create_test_worker_info(
    worker_id: Uuid,
    model_name: String,
    node_id: Uuid,
    device_id: Uuid,
    device_params: String,
) -> WorkerInfo<String> {
    WorkerInfo {
        worker_id,
        model_name,
        node_id,
        device_id,
        device_params,
        created_at: std::time::Instant::now(),
        status: WorkerStatus::Running,
    }
}

/// 测试辅助函数：等待异步任务完成
pub async fn wait_for_completion() {
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
}

/// 创建测试场景的设备列表
pub fn create_test_devices(count: usize) -> Vec<MockDevice> {
    (0..count)
        .map(|i| {
            let mut device = MockDevice::new();
            device.raw_device.logic_index = Some(i as u32);
            device
        })
        .collect()
}

/// 创建测试场景的Worker状态映射
pub fn create_worker_state_map(
    scenarios: Vec<(Uuid, Vec<(String, String)>)>, // (device_id, [(worker_id, params), ...])
) -> std::collections::HashMap<Uuid, Vec<(Uuid, String, Option<String>)>> {
    use std::collections::HashMap;

    let mut state_map = HashMap::new();

    for (device_id, workers) in scenarios {
        let mut worker_infos = Vec::new();
        for (worker_id_str, params) in workers {
            let worker_id = Uuid::parse_str(&worker_id_str).unwrap_or_else(|_| Uuid::new_v4());
            let worker_info = (worker_id, params.clone(), Some(format!("model_{}", params)));
            worker_infos.push(worker_info);
        }
        state_map.insert(device_id, worker_infos);
    }

    state_map
}

/// 验证迁移命令的辅助函数
pub fn validate_migration_commands(
    commands: &[crate::compute_node::r#virtual::virtual_compute_node::MigrationCommand<
        MockDevice,
    >],
    expected_migrations: usize,
    expected_creations: usize,
    expected_deletions: usize,
) -> bool {
    use crate::compute_node::r#virtual::base::Command as BaseCommand;

    let mut migration_count = 0;
    let mut creation_count = 0;
    let mut deletion_count = 0;

    for (_, base_commands) in commands {
        for base_command in base_commands {
            match base_command {
                BaseCommand::CreateWorker { .. } => creation_count += 1,
                BaseCommand::LoadModel { .. } => migration_count += 1,
                BaseCommand::UnloadModel { .. } => migration_count += 1,
                BaseCommand::DestroyWorker { .. } => deletion_count += 1,
            }
        }
    }

    migration_count == expected_migrations
        && creation_count == expected_creations
        && deletion_count == expected_deletions
}

/// 创建复杂的测试场景数据
/// 测试场景数据结构
pub struct TestScenario {
    /// 场景名称
    pub name: String,
    /// 测试设备列表
    pub devices: Vec<MockDevice>,
    /// 当前状态映射
    pub current_state: std::collections::HashMap<Uuid, Vec<(Uuid, String, Option<String>)>>,
    /// 目标分配映射（模型 -> 设备参数）
    pub target_allocation: std::collections::HashMap<String, Vec<String>>,
    /// 期望的命令数量（迁移数，创建数，删除数）
    pub expected_commands: (usize, usize, usize),
}

impl TestScenario {
    /// 创建简单的重分配场景
    pub fn create_simple_reallocation() -> Self {
        let devices = create_test_devices(3);
        let device_ids: Vec<Uuid> = devices.iter().map(|d| d.id).collect();

        // 当前状态：所有worker在device0上
        let current_state = create_worker_state_map(vec![(
            device_ids[0],
            vec![
                ("worker1".to_string(), "1024,4".to_string()),
                ("worker2".to_string(), "2048,8".to_string()),
            ],
        )]);

        // 目标分配：分散到不同设备
        let mut target_allocation = std::collections::HashMap::new();
        target_allocation.insert("model_1024,4".to_string(), vec!["1024,4".to_string()]);
        target_allocation.insert("model_2048,8".to_string(), vec!["2048,8".to_string()]);

        TestScenario {
            name: "Simple Reallocation".to_string(),
            devices,
            current_state,
            target_allocation,
            expected_commands: (1, 0, 0), // 1个迁移，0个创建，0个删除
        }
    }

    /// 创建复杂的负载均衡场景
    pub fn create_load_balancing() -> Self {
        let devices = create_test_devices(4);
        let device_ids: Vec<Uuid> = devices.iter().map(|d| d.id).collect();

        // 当前状态：不均匀分布
        let current_state = create_worker_state_map(vec![
            (
                device_ids[0],
                vec![
                    ("worker1".to_string(), "1024,4".to_string()),
                    ("worker2".to_string(), "1024,4".to_string()),
                    ("worker3".to_string(), "2048,8".to_string()),
                ],
            ),
            (
                device_ids[1],
                vec![("worker4".to_string(), "2048,8".to_string())],
            ),
        ]);

        // 目标分配：均匀分布
        let mut target_allocation = std::collections::HashMap::new();
        target_allocation.insert(
            "model_1024,4".to_string(),
            vec!["1024,4".to_string(), "1024,4".to_string()],
        );
        target_allocation.insert(
            "model_2048,8".to_string(),
            vec!["2048,8".to_string(), "2048,8".to_string()],
        );

        TestScenario {
            name: "Load Balancing".to_string(),
            devices,
            current_state,
            target_allocation,
            expected_commands: (2, 0, 0), // 2个迁移，0个创建，0个删除
        }
    }

    /// 创建扩缩容场景
    pub fn create_scaling_scenario() -> Self {
        let devices = create_test_devices(3);
        let device_ids: Vec<Uuid> = devices.iter().map(|d| d.id).collect();

        // 当前状态：较少的worker
        let current_state = create_worker_state_map(vec![
            (
                device_ids[0],
                vec![("worker1".to_string(), "1024,4".to_string())],
            ),
            (
                device_ids[1],
                vec![
                    ("worker2".to_string(), "2048,8".to_string()),
                    ("worker3".to_string(), "4096,16".to_string()),
                ],
            ),
        ]);

        // 目标分配：增加更多worker
        let mut target_allocation = std::collections::HashMap::new();
        target_allocation.insert(
            "model_1024,4".to_string(),
            vec!["1024,4".to_string(), "1024,4".to_string()],
        );
        target_allocation.insert("model_2048,8".to_string(), vec!["2048,8".to_string()]);
        target_allocation.insert(
            "model_4096,16".to_string(),
            vec!["4096,16".to_string(), "4096,16".to_string()],
        );

        TestScenario {
            name: "Scaling Scenario".to_string(),
            devices,
            current_state,
            target_allocation,
            expected_commands: (0, 2, 0), // 0个迁移，2个创建，0个删除
        }
    }
}
