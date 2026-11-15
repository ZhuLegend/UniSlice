//! 从配置构建异构系统的构建器

use std::{
    collections::{BTreeSet, HashMap},
    pin::Pin,
    sync::Arc,
};

use bincode::{Decode, Encode};
#[cfg(feature = "huawei")]
use compute_node::compute_device::vgpu::resource_spec::hw::VChipScale;
#[cfg(feature = "nvidia")]
use compute_node::compute_device::vgpu::resource_spec::nv::mig::VDeviceScale;
use compute_node::{
    compute_device::vgpu::share_target::{ChoosableShareTargetContainer, ConfigurableShareTargets},
    compute_node::r#virtual::{
        base::DeviceState,
        virtual_compute_node::components::request_scheduler::batch_size_extractor::BatchExtractorStrategy,
    },
};
use futures::future::TryJoinAll;
use serde::{Serialize, de::DeserializeOwned};

use crate::{
    daemon::{
        spec::DaemonSpec,
        types::{Config, DeviceConfig, LatencyData, RemoteNodeConfig},
    },
    schedule_system::virtual_compute_node::{
        algorithms::buddy_cell::constrained::BuddyCellResourceImplHelper,
        system::heterogeneous::{
            bid::Bid,
            device::{BoxedShareableDevice, VirtualComputeNodeAdapter},
            system::HeterogeneousSystem,
        },
    },
};

/// 异构系统构建器
pub struct HeterogeneousSystemBuilder;

/// Stable logical-pool identities, shared by the builder and experiment plans.
pub(crate) fn template_pool_ids(config: &Config) -> HashMap<String, uuid::Uuid> {
    config
        .remote_nodes
        .values()
        .map(|node| node.template_name.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .enumerate()
        .map(|(index, template)| {
            (
                template,
                uuid::Uuid::from_u128(0x48534753000000000000000000000000 | index as u128),
            )
        })
        .collect()
}

impl HeterogeneousSystemBuilder {
    /// 从配置构建异构系统
    pub async fn build_from_config_and_rate_predictor(
        mut config: Config,
        model_request_rate_predictor: impl Fn() -> Pin<
            Box<dyn Future<Output = HashMap<String, f32>> + Send>,
        > + Send
        + Sync
        + 'static,
        latency_data: Arc<LatencyData>,
        template_latency_data: HashMap<String, Arc<LatencyData>>,
    ) -> Result<HeterogeneousSystem, String> {
        let request_timeout = std::time::Duration::from_millis(
            config
                .remote_nodes
                .values()
                .map(|node| node.inference_timeout_ms)
                .max()
                .unwrap_or(300_000),
        );
        let pool_ids = template_pool_ids(&config);
        // 1. 按template_name分组remote_nodes
        let template_groups =
            Self::group_nodes_by_template(std::mem::take(&mut config.remote_nodes));

        // 3. 为每个模版组创建VirtualComputeNode并转换为设备
        let mut template_groups = template_groups.into_iter().collect::<Vec<_>>();
        template_groups.sort_by(|a, b| a.0.cmp(&b.0));
        let devices = template_groups
            .into_iter()
            .map(|(template_name, remote_nodes)| {
                let config = &config;
                let pool_id = pool_ids[&template_name];
                let predictions = template_latency_data
                    .get(&template_name)
                    .cloned()
                    .unwrap_or_else(|| latency_data.clone());
                async move {
                    Self::create_devices_from_templates(
                        pool_id,
                        &config,
                        remote_nodes,
                        config
                            .device_combinations
                            .get(&template_name)
                            .unwrap_or_else(|| panic!("Template {} not found", template_name)),
                        predictions,
                    )
                    .await
                }
            })
            .collect::<TryJoinAll<_>>()
            .await?
            .into_iter()
            .map(|node| (node.id(), node))
            .collect();

        // 4. 创建异构系统
        let batch_size_range = config.batch_size_range.0..=config.batch_size_range.1;

        Ok(HeterogeneousSystem::new(
            devices,
            Arc::new(model_request_rate_predictor),
            batch_size_range,
        )
        .with_request_timeout(request_timeout))
    }

    /// 按模版名称分组远程节点
    fn group_nodes_by_template(
        remote_nodes: HashMap<String, RemoteNodeConfig>,
    ) -> HashMap<String, HashMap<String, RemoteNodeConfig>> {
        let mut template_groups: HashMap<String, HashMap<String, RemoteNodeConfig>> =
            HashMap::new();

        for (name, node) in remote_nodes {
            template_groups
                .entry(node.template_name.clone())
                .or_default()
                .insert(name, node);
        }

        template_groups
    }

    /// 从模版组创建设备
    async fn create_devices_from_templates(
        pool_id: uuid::Uuid,
        config: &Config,
        nodes: HashMap<String, RemoteNodeConfig>,
        device_config: &DeviceConfig,
        latency_data: Arc<LatencyData>,
    ) -> Result<Arc<dyn BoxedShareableDevice<Target = Bid>>, String> {
        let batch_extractor_strategy = config.batch_extractor_strategy.clone();

        Self::create_virtual_node_adapters(
            pool_id,
            nodes,
            device_config,
            latency_data.clone(),
            batch_extractor_strategy,
        )
        .await
    }

    /// 创建虚拟节点适配器
    async fn create_virtual_node_adapters(
        pool_id: uuid::Uuid,
        remote_nodes: HashMap<String, RemoteNodeConfig>,
        device_config: &DeviceConfig,
        latency_data: Arc<LatencyData>,
        batch_extractor_strategy: BatchExtractorStrategy,
    ) -> Result<Arc<dyn BoxedShareableDevice<Target = Bid>>, String> {
        match &device_config {
            #[cfg(feature = "huawei")]
            DeviceConfig::Huawei {
                share_targets,
                resource_template,
            } => {
                use crate::daemon::spec::huawei_spec::HuaweiMigDaemon;

                let resource_template = *resource_template;
                let target_container =
                    ConfigurableShareTargets::<VChipScale>::new(ConfigurableShareTargets::<
                        VChipScale,
                    >::from_names(
                        share_targets
                    ));
                Self::create_daemon_specific_virtual_node::<HuaweiMigDaemon>(
                    pool_id,
                    remote_nodes,
                    latency_data,
                    batch_extractor_strategy,
                    resource_template,
                    target_container,
                )
                .await
            }
            #[cfg(feature = "nvidia")]
            DeviceConfig::Nvidia {
                share_targets,
                resource_template,
            } => {
                use crate::daemon::spec::nvidia_spec::NvidiaMigDaemon;

                let resource_template = *resource_template;
                let target_container =
                    ConfigurableShareTargets::<VDeviceScale>::new(ConfigurableShareTargets::<
                        VDeviceScale,
                    >::from_names(
                        share_targets
                    ));
                Self::create_daemon_specific_virtual_node::<NvidiaMigDaemon>(
                    pool_id,
                    remote_nodes,
                    latency_data,
                    batch_extractor_strategy,
                    resource_template,
                    target_container,
                )
                .await
            }
            #[cfg(all(feature = "mock", feature = "huawei"))]
            DeviceConfig::MockHuawei {
                share_targets,
                resource_template,
            } => {
                use crate::daemon::spec::huawei_spec::MockHuaweiMigDaemon;

                let resource_template = *resource_template;
                let target_container =
                    ConfigurableShareTargets::<VChipScale>::new(ConfigurableShareTargets::<
                        VChipScale,
                    >::from_names(
                        share_targets
                    ));
                Self::create_daemon_specific_virtual_node::<MockHuaweiMigDaemon>(
                    pool_id,
                    remote_nodes,
                    latency_data,
                    batch_extractor_strategy,
                    resource_template,
                    target_container,
                )
                .await
            }
            #[cfg(all(feature = "mock", feature = "nvidia"))]
            DeviceConfig::MockNvidia {
                share_targets,
                resource_template,
            } => {
                use crate::daemon::spec::nvidia_spec::MockNvidiaMigDaemon;

                let resource_template = *resource_template;
                let target_container =
                    ConfigurableShareTargets::<VDeviceScale>::new(ConfigurableShareTargets::<
                        VDeviceScale,
                    >::from_names(
                        share_targets
                    ));
                Self::create_daemon_specific_virtual_node::<MockNvidiaMigDaemon>(
                    pool_id,
                    remote_nodes,
                    latency_data,
                    batch_extractor_strategy,
                    resource_template,
                    target_container,
                )
                .await
            }
        }
    }

    async fn create_daemon_specific_virtual_node<D: DaemonSpec>(
        pool_id: uuid::Uuid,
        remote_nodes: HashMap<String, RemoteNodeConfig>,
        latency_data: Arc<LatencyData>,
        batch_extractor_strategy: BatchExtractorStrategy,
        resource_template: <D as DaemonSpec>::Resource,
        target_container: ConfigurableShareTargets<D::ShareParam>,
    ) -> Result<Arc<dyn BoxedShareableDevice<Target = Bid>>, String>
    where
        <D as DaemonSpec>::ShareParam: Serialize + DeserializeOwned + Encode + Decode<()>,
        <D as DaemonSpec>::DeviceState: Decode<()> + Encode + DeviceState + Sync,
        <D as DaemonSpec>::Resource:
            BuddyCellResourceImplHelper<ShareParam = <D as DaemonSpec>::ShareParam>,
    {
        let mut virtual_node = D::create_virtual_compute_node(
            remote_nodes,
            latency_data.clone(),
            batch_extractor_strategy,
        )
        .await
        .map_err(|e| e.to_string())?;
        virtual_node.set_experiment_random_namespace(pool_id.to_string());

        let (predictor, observer) = crate::daemon::predictor::with_online_correction(
            pool_id,
            D::create_vnode_latency_predictor(latency_data),
        );
        virtual_node.set_performance_observer(observer);

        let adapter = VirtualComputeNodeAdapter::new(
            virtual_node,
            resource_template,
            target_container,
            predictor,
        )
        .with_id(pool_id);

        Ok(Arc::new(adapter))
    }
}
