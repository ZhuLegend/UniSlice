//! 守护进程规格定义 Trait

use std::{collections::HashMap, error::Error, fmt::Debug, future::Future, pin::Pin, sync::Arc};

use bincode::{Decode, Encode};
use compute_node::{
    compute_device::vgpu::{
        ShareableComputeDeviceResource,
        share_target::{ConfigurableShareTargets, ShareTarget},
    },
    compute_node::r#virtual::{
        base::{self, DeviceState},
        virtual_compute_node::{
            RemoteBaseVirtualComputeNode, VirtualComputeNode,
            components::request_scheduler::batch_size_extractor::BatchExtractorStrategy,
        },
    },
};
use serde::{Serialize, de::DeserializeOwned};
use utils::ErrorMessage;

use crate::{
    daemon::types::{DeviceConfig, LatencyData, RemoteNodeConfig},
    schedule_system::virtual_compute_node::{
        algorithms::buddy_cell::BuddyCellResource, system::InferenceLatencyPredictor,
    },
};

#[cfg(feature = "huawei")]
pub mod huawei_spec;
#[cfg(feature = "nvidia")]
pub mod nvidia_spec;

/// 一个辅助 Trait，用于抽象不同的 VScale 类型（如 `VChipScale`, `VDeviceScale`）。
pub trait VScale:
    Clone + Copy + Send + Sync + Debug + Eq + Ord + std::hash::Hash + 'static
{
}

/// 创建 `VirtualComputeNode` 的异步函数类型定义。
pub type CreateVirtualComputeNodeFuture<Daemon> = Pin<
    Box<
        dyn Future<
                Output = Result<
                    VirtualComputeNode<
                        <Daemon as DaemonSpec>::ShareParam,
                        <Daemon as DaemonSpec>::DeviceState,
                    >,
                    Box<dyn Error + Send>,
                >,
            > + Send,
    >,
>;

/// DaemonSpec Trait，用于抽象所有与特定硬件相关的应用层配置和逻辑。
pub trait DaemonSpec: Sized + Send + Sync + 'static {
    /// 进行切分的参数
    type ShareParam: VScale + ShareTarget;

    /// 被切分的资源表示
    type Resource: ShareableComputeDeviceResource + for<'a> From<&'a Self::ShareParam>;

    /// 内部设备状态
    type DeviceState: base::DeviceState + Send + Sync + 'static;

    // ------------------------------------------------------------------
    // |                 ↓↓↓ 需要手动实现的部分 ↓↓↓                     |
    // ------------------------------------------------------------------

    /// 调度器 BuddyCell 分配器所使用的资源类型。
    type SchedulerResource: BuddyCellResource<ChoosableShareTarget = Self::ShareParam, InnerResource = Self::Resource>
        + Clone
        + Send
        + Sync
        + 'static;

    /// 返回守护进程的服务名称。
    fn service_name() -> &'static str;

    /// 从设备配置中创建资源模板实例。
    fn create_resource_template(device_config: &DeviceConfig) -> Result<Self::Resource, String>;

    /// 从设备配置创建可配置的共享目标实例。
    fn create_configurable_targets(
        device_config: &DeviceConfig,
    ) -> Result<ConfigurableShareTargets<Self::ShareParam>, String>;

    // ------------------------------------------------------------------
    // |                 ↓↓↓ 自动提供的默认实现 ↓↓↓                     |
    // ------------------------------------------------------------------

    /// 创建用于 `VirtualComputeNode` 的延迟预测器。
    fn create_vnode_latency_predictor(
        data: Arc<LatencyData>,
    ) -> InferenceLatencyPredictor<Self::ShareParam> {
        Arc::new(
            move |model: &str, resource: &Self::ShareParam, batch_size: u32| -> f32 {
                let resource_key = resource.name();
                data.get(model)
                    .and_then(|model_data| model_data.get(resource_key))
                    .and_then(|batch_data| {
                        latency_profile::interpolate(batch_data, batch_size)
                            .ok()
                            .or_else(|| {
                                // Legacy non-simulation callers keep the nearest-point fallback.
                                batch_data
                                    .iter()
                                    .min_by_key(|(b, _)| b.abs_diff(batch_size))
                                    .map(|(_, l)| *l)
                            })
                    })
                    .unwrap_or(100.0)
            },
        )
    }

    /// 创建一个虚拟计算节点 (VirtualComputeNode)。
    fn create_virtual_compute_node(
        remote_nodes: HashMap<String, RemoteNodeConfig>,
        latency_data: Arc<LatencyData>,
        batch_extractor_strategy: BatchExtractorStrategy,
    ) -> CreateVirtualComputeNodeFuture<Self>
    where
        Self::ShareParam: Serialize + DeserializeOwned + Encode + Decode<()>,
        Self::DeviceState: Decode<()> + Encode + DeviceState + Sync,
    {
        Box::pin(async move {
            let latency_predictor = Self::create_vnode_latency_predictor(latency_data);
            let mut virtual_node = VirtualComputeNode::new(
                batch_extractor_strategy,
                move |name, target, batch_size| {
                    (*latency_predictor)(name, target, batch_size as u32)
                },
            );

            for (name, config) in remote_nodes {
                let address = format!("{}:{}", config.ip, config.port);
                tracing::info!(
                    "Adding remote node for {}: {} -> {}",
                    Self::service_name(),
                    name,
                    address
                );
                let remote_node =
                    RemoteBaseVirtualComputeNode::<Self::ShareParam, Self::DeviceState>::new(
                        address,
                    )
                    .with_timeouts(
                        std::time::Duration::from_millis(config.control_timeout_ms),
                        std::time::Duration::from_millis(config.inference_timeout_ms),
                    );
                if let Some(expected) = &config.expected_simulation_identity {
                    use compute_node::compute_node::r#virtual::virtual_compute_node::DynBaseVirtualComputeNode;
                    let status = remote_node
                        .status()
                        .await
                        .map_err(|e| Box::new(ErrorMessage::new(e)) as Box<dyn Error + Send>)?;
                    let expected_gpus = serde_json::from_str::<serde_json::Value>(expected)
                        .ok()
                        .and_then(|v| v[0]["gpus"].as_u64())
                        .unwrap_or(0) as usize;
                    if status.devices.len() != expected_gpus
                        || status.devices.values().any(|state| {
                            serde_json::to_value(state)
                                .ok()
                                .and_then(|v| {
                                    v.get("config_name")
                                        .and_then(|n| n.as_str())
                                        .map(str::to_owned)
                                })
                                .as_ref()
                                != Some(expected)
                        })
                    {
                        return Err(Box::new(ErrorMessage::new(
                            "simulation node configuration does not match scheduler".into(),
                        )) as Box<dyn Error + Send>);
                    }
                }
                let node_id = uuid::Uuid::new_v4();
                virtual_node
                    .add_virtual_compute_node(node_id, Arc::new(remote_node))
                    .await
                    .map_err(|e| {
                        Box::new(ErrorMessage::from(format!(
                            "Failed to add remote node: {}",
                            e
                        ))) as Box<dyn Error + Send>
                    })?;
            }

            Ok(virtual_node)
        })
    }
}
