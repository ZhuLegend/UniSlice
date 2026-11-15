//! A base virtual compute node implementation.
//!
//! This implementation supports creating and managing workers that run on virtual GPUs.
//! It allows for loading and unloading models, and provides a mechanism to compute tasks on the workers.

use std::{
    collections::HashMap,
    fmt::Debug,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    path::PathBuf,
    sync::Arc,
};

use bincode::{Decode, Encode};
use dashmap::DashMap;
use futures::future::JoinAll;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tokio::sync::RwLock;
use tracing::{trace, warn};
use uuid::Uuid;
use worker::model::tensor::DynTensor;

use crate::{
    ComputeNode,
    compute_device::{ComputeDevice, vgpu::ShareableComputeDevice},
    compute_node::r#virtual::BaseWork,
    runnable_on::RunnableOn,
    util::{
        Input, Output, WorkState,
        bin_compute_node::BinComputeNode,
        worker_proxy::{
            WorkerProxy,
            bin_simple_worker_proxy::{
                BinSimpleWorkerProxyImplInit, WorkerProxyCapable, WorkerProxyType,
            },
        },
    },
};

/// 计算节点工作
#[derive(Debug, Serialize, Deserialize, Encode, Decode, Clone)]
#[serde(bound = "State: WorkState")]
pub struct Work<State: WorkState> {
    /// 目标Worker
    #[bincode(with_serde)]
    pub target_worker: Uuid,
    /// 具体工作
    pub work: BaseWork<State>,
}

impl Work<Input> {
    /// generate a new [Work<Input>] which just keep the batch size and data type,
    /// and set other dimensions to dimensions param.
    ///
    /// ```
    /// # use compute_node::util::Input;
    /// # use crate::compute_node::compute_node::virtual::base::Work;
    /// #
    /// # fn example(work: Work<Input>) -> {
    ///     let batch_size = work.work.data().dimensions()[0];
    ///     work.with_batch_size_only([3, 224, 224]);
    ///     assert_eq!(work.work.data().dimensions(), vec![batch_size, 3, 224, 224]);
    /// # }
    ///
    /// ```
    pub fn with_batch_size_only(mut self, mut dimensions: Vec<usize>) -> Self {
        let batch_size = self.work.data.dimensions()[0];
        dimensions.insert(0, batch_size as usize);
        self.work.data = DynTensor::random(dimensions, self.work.data.dtype());

        self
    }
}

impl Work<Output> {
    /// set other dimensions to 0 except batch size.
    pub fn keep_batch_size_only(mut self) -> Self {
        let mut dimensions = self
            .work
            .data
            .dimensions()
            .iter()
            .map(|dim| *dim as usize)
            .collect::<Vec<_>>();
        dimensions.iter_mut().skip(1).for_each(|dim| *dim = 0);
        self.work.data = DynTensor::random(dimensions, self.work.data.dtype());

        self
    }
}

/// Worker 状态
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Encode, Decode)]
#[serde(bound(deserialize = "Param: DeserializeOwned"))]
pub struct WorkerState<Param>
where
    Param: Serialize + DeserializeOwned + Send + Sync,
{
    /// 当前加载的模型名称
    pub model_name: Option<String>,
    /// 创建 Worker 时的参数
    #[bincode(with_serde)]
    pub params: Param,
    /// GPU 的 device_id
    #[bincode(with_serde)]
    pub device_id: Uuid,
    /// External simulation stream slot, independent of process and worker UUIDs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub experiment_noise_slot: Option<String>,
}

/// 基础计算节点
#[derive(Debug)]
pub struct BaseVirtualComputeNode<Worker, Device, Proxy>
where
    Worker: worker::Worker + RunnableOn<<Device as ShareableComputeDevice>::SharedComputeDevice>,
    Device: ShareableComputeDevice,
    <Device as ShareableComputeDevice>::RawComputeDevice:
        DeviceManageHelper<<Device as ShareableComputeDevice>::RawComputeDevice>,
    <<Device as ShareableComputeDevice>::RawComputeDevice as ComputeDevice>::Response:
        Encode + Decode<()> + Send + Sync,
    <Device as ShareableComputeDevice>::ShareParam:
        Serialize + DeserializeOwned + Debug + Send + Sync,
    Proxy: WorkerProxy
        + BinSimpleWorkerProxyImplInit<<Device as ShareableComputeDevice>::SharedComputeDevice>
        + Send
        + Sync
        + 'static,
{
    /// 计算节点逻辑 id
    device_logical_id: HashMap<Uuid, u32>,
    /// 该计算节点持有的计算设备
    devices: HashMap<Uuid, Device>,
    /// 计算节点管理的虚拟GPU工作代理
    workers: DashMap<Uuid, Arc<RwLock<ProxyCleaner<Proxy>>>>,
    /// 模型目录路径
    model_dir: PathBuf,
    /// 模型配置目录路径
    model_config_dir: PathBuf,
    /// 存储 Worker 的状态信息
    worker_states: DashMap<Uuid, WorkerState<<Device as ShareableComputeDevice>::ShareParam>>,
    /// 二进制计算节点路径
    bin_worker_path: PathBuf,
    worker: PhantomData<fn() -> Worker>,
    // Serialize mutations per worker ID, including retries while creation/destruction is in flight.
    command_locks: DashMap<Uuid, Arc<tokio::sync::Mutex<()>>>,
    cleanup_tasks: utils::sync::Task,
    shutting_down: std::sync::atomic::AtomicBool,
    destroying: DashMap<Uuid, tokio::sync::watch::Receiver<Option<Result<(), String>>>>,
}

impl<Worker, Device, Proxy> BaseVirtualComputeNode<Worker, Device, Proxy>
where
    Worker: worker::Worker
        + RunnableOn<<Device as ShareableComputeDevice>::SharedComputeDevice>
        + WorkerProxyCapable<<Device as ShareableComputeDevice>::SharedComputeDevice>,
    Device: ShareableComputeDevice,
    <Device as ShareableComputeDevice>::RawComputeDevice:
        DeviceManageHelper<<Device as ShareableComputeDevice>::RawComputeDevice>,
    <<Device as ShareableComputeDevice>::RawComputeDevice as ComputeDevice>::Response:
        Encode + Decode<()> + Send + Sync,
    <Device as ShareableComputeDevice>::ShareParam:
        Serialize + DeserializeOwned + Debug + Send + Sync,
    Proxy: WorkerProxy
        + BinSimpleWorkerProxyImplInit<<Device as ShareableComputeDevice>::SharedComputeDevice>
        + Send
        + Sync
        + 'static,
{
    /// 创建一个新的基础计算节点
    #[tracing::instrument(level = "trace", skip(device_ids), fields(devices))]
    pub fn new(
        device_ids: impl IntoIterator<Item = impl Into<usize>>,
        model_dir: PathBuf,
        model_config_dir: PathBuf,
        bin_worker_path: PathBuf,
    ) -> Self {
        let devices = <Device as ShareableComputeDevice>::RawComputeDevice::get_devices(device_ids);
        tracing::Span::current().record("devices", tracing::field::debug(&devices));
        Self::from_devices(
            devices
                .into_iter()
                .map(|(id, (logical_id, device))| (id, (logical_id as u32, device.into())))
                .collect(),
            model_dir,
            model_config_dir,
            bin_worker_path,
        )
    }

    /// Inject configured devices while retaining the existing ID-based hardware constructor.
    pub fn from_devices(
        devices: HashMap<Uuid, (u32, Device)>,
        model_dir: PathBuf,
        model_config_dir: PathBuf,
        bin_worker_path: PathBuf,
    ) -> Self {
        let device_logical_id = devices
            .iter()
            .map(|(id, (logical_id, _))| (*id, *logical_id))
            .collect();
        Self {
            devices: devices
                .into_iter()
                .map(|(id, (_, device))| (id, device))
                .collect(),
            device_logical_id,
            workers: DashMap::new(),
            worker_states: DashMap::new(),
            command_locks: DashMap::new(),
            destroying: DashMap::new(),
            shutting_down: std::sync::atomic::AtomicBool::new(false),
            cleanup_tasks: utils::sync::Task::new(),
            model_dir,
            model_config_dir,
            bin_worker_path,
            worker: PhantomData,
        }
    }

    /// 计算节点设备的逻辑 id映射
    pub fn device_logical_id_map(&self) -> &HashMap<Uuid, u32> {
        &self.device_logical_id
    }
}

/// 计算节点命令
#[derive(Clone, Serialize, Deserialize, Encode, Decode, Debug)]
pub enum Command<Param>
where
    Param: Debug,
{
    /// 创建[Worker](worker::Worker)
    CreateWorker {
        /// Worker id
        #[bincode(with_serde)]
        id: Uuid,
        /// 物理GPU id
        #[bincode(with_serde)]
        device_id: Uuid,
        /// 额外参数
        params: Param,
    },
    /// 在指定[Worker](worker::Worker)上加载模型
    LoadModel {
        /// Worker id
        #[bincode(with_serde)]
        id: Uuid,
        /// 模型名称
        name: String,
    },
    /// 卸载指定[Worker](worker::Worker)上的模型
    UnloadModel {
        /// Worker id
        #[bincode(with_serde)]
        id: Uuid,
    },
    /// 销毁指定[Worker](worker::Worker)
    DestroyWorker {
        /// Worker id
        #[bincode(with_serde)]
        id: Uuid,
    },
}

/// Device 响应
pub trait DeviceState: Debug + Serialize + DeserializeOwned {}

/// 计算节点状态
#[derive(Debug, Serialize, Deserialize, Encode, Decode)]
#[serde(bound(deserialize = "Res: DeviceState"))]
pub struct Status<Res>
where
    Res: DeviceState,
{
    /// [WorkerProxyType] 类型
    pub worker_proxy_type: WorkerProxyType,
    /// Device 状态
    #[bincode(with_serde)]
    pub devices: HashMap<Uuid, Res>,
}

/// 计算节点状态
#[derive(Debug, Serialize, Deserialize, Encode, Decode)]
#[serde(bound(deserialize = "Param: Serialize + DeserializeOwned + Send + Sync"))]
pub struct State<Param>
where
    Param: Serialize + DeserializeOwned + Send + Sync,
{
    /// Worker 状态
    #[bincode(with_serde)]
    pub workers: HashMap<Uuid, WorkerState<Param>>,
}

/// 清理完成的等待句柄
pub struct CleanupHandle {
    receiver: tokio::sync::watch::Receiver<Option<Result<(), String>>>,
}

impl CleanupHandle {
    /// 等待清理完成
    pub async fn wait(mut self) -> Result<(), String> {
        loop {
            if let Some(result) = self.receiver.borrow().clone() {
                return result;
            }
            self.receiver
                .changed()
                .await
                .map_err(|_| "cleanup task ended without a result".to_string())?;
        }
    }
}

/// Proxy Cleaner
///
/// This struct used to auto clean up the proxy when it is dropped.
/// If you have reference to the proxy, then the proxy will must contain in this struct, so it can safely defer to inner Proxy
#[derive(Debug)]
struct ProxyCleaner<Proxy: 'static>
where
    Proxy: WorkerProxy + Send + Sync,
{
    proxy: Option<Proxy>,
    cleanup_tasks: utils::sync::Task,
    // 用于通知清理完成的通知器
    cleanup_receiver: tokio::sync::watch::Sender<Option<Result<(), String>>>,
}

impl<Proxy: 'static> ProxyCleaner<Proxy>
where
    Proxy: WorkerProxy + Send + Sync,
{
    /// Create a new ProxyCleaner
    pub fn new(proxy: Proxy, cleanup_tasks: utils::sync::Task) -> Self {
        Self {
            proxy: Some(proxy),
            cleanup_tasks,
            cleanup_receiver: tokio::sync::watch::channel(None).0,
        }
    }

    /// Start cleanup explicitly after exclusive access has drained current compute.
    /// The spawned task owns the proxy even if its caller is cancelled.
    pub fn begin_cleanup(&mut self) -> CleanupHandle {
        if let Some(proxy) = self.proxy.take() {
            let notifier = self.cleanup_receiver.clone();
            self.cleanup_tasks.spawn(move |_| async move {
                let result = proxy.destroy().await;
                if let Err(error) = &result {
                    tracing::error!(%error, "proxy cleanup failed");
                }
                notifier.send_replace(Some(result));
            });
        }
        self.cleanup_handle()
    }

    /// 获取一个可以等待清理完成的句柄
    /// 这样即使 ProxyCleaner 被移动或 drop，外部还能等待清理完成
    pub fn cleanup_handle(&self) -> CleanupHandle {
        CleanupHandle {
            receiver: self.cleanup_receiver.subscribe(),
        }
    }
}

impl<Proxy: 'static> Drop for ProxyCleaner<Proxy>
where
    Proxy: WorkerProxy + Send + Sync,
{
    fn drop(&mut self) {
        if self.proxy.is_some() {
            if tokio::runtime::Handle::try_current().is_ok() {
                self.begin_cleanup();
            } else {
                // Without an executor we cannot safely perform asynchronous cleanup.
                // Keep the reservation quarantined instead of claiming it is free.
                tracing::error!("proxy dropped outside runtime; resource quarantined");
                std::mem::forget(self.proxy.take());
            }
        }
    }
}

impl<Proxy: 'static> Deref for ProxyCleaner<Proxy>
where
    Proxy: WorkerProxy + Send + Sync,
{
    type Target = Proxy;

    fn deref(&self) -> &Self::Target {
        self.proxy.as_ref().expect("ProxyCleaner has been dropped")
    }
}

impl<Proxy: 'static> DerefMut for ProxyCleaner<Proxy>
where
    Proxy: WorkerProxy + Send + Sync,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.proxy.as_mut().expect("ProxyCleaner has been dropped")
    }
}

impl<Worker, Device, Proxy> BaseVirtualComputeNode<Worker, Device, Proxy>
where
    Worker: worker::Worker
        + RunnableOn<<Device as ShareableComputeDevice>::SharedComputeDevice>
        + WorkerProxyCapable<<Device as ShareableComputeDevice>::SharedComputeDevice>
        + 'static,
    Device: ShareableComputeDevice + Send + Sync,
    <Device as ShareableComputeDevice>::RawComputeDevice:
        DeviceManageHelper<<Device as ShareableComputeDevice>::RawComputeDevice>,
    <<Device as ShareableComputeDevice>::RawComputeDevice as ComputeDevice>::Response:
        Encode + Decode<()> + Send + Sync + DeviceState,
    <Device as ShareableComputeDevice>::ShareParam:
        Serialize + DeserializeOwned + Debug + Send + Sync,
    Proxy: WorkerProxy
        + BinSimpleWorkerProxyImplInit<<Device as ShareableComputeDevice>::SharedComputeDevice>
        + Send
        + Sync,
{
    /// Stop accepting mutations, drain accepted commands, and wait for every
    /// registered destruction. A failed cleanup remains an error on shutdown.
    pub async fn shutdown(&self) -> Result<(), String> {
        self.shutting_down
            .store(true, std::sync::atomic::Ordering::Release);
        let locks: Vec<_> = self
            .command_locks
            .iter()
            .map(|v| v.value().clone())
            .collect();
        for lock in locks {
            drop(lock.lock().await);
        }
        let mut ids: std::collections::HashSet<Uuid> =
            self.workers.iter().map(|v| *v.key()).collect();
        ids.extend(self.destroying.iter().map(|v| *v.key()));
        let results = futures::future::join_all(
            ids.into_iter()
                .map(|id| self.handle(Command::DestroyWorker { id })),
        )
        .await;
        self.cleanup_tasks.wait().await;
        let errors: Vec<_> = results.into_iter().filter_map(Result::err).collect();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("; "))
        }
    }
}

impl<Worker, Device, Proxy> ComputeNode for BaseVirtualComputeNode<Worker, Device, Proxy>
where
    Worker: worker::Worker
        + RunnableOn<<Device as ShareableComputeDevice>::SharedComputeDevice>
        + WorkerProxyCapable<<Device as ShareableComputeDevice>::SharedComputeDevice>
        + 'static,
    Device: ShareableComputeDevice + Send + Sync,
    <Device as ShareableComputeDevice>::RawComputeDevice:
        DeviceManageHelper<<Device as ShareableComputeDevice>::RawComputeDevice>,
    <<Device as ShareableComputeDevice>::RawComputeDevice as ComputeDevice>::Response:
        Encode + Decode<()> + Send + Sync + DeviceState,
    <Device as ShareableComputeDevice>::ShareParam:
        Serialize + DeserializeOwned + Debug + Send + Sync,
    Proxy: WorkerProxy
        + BinSimpleWorkerProxyImplInit<<Device as ShareableComputeDevice>::SharedComputeDevice>
        + Send
        + Sync,
{
    type State = State<<Device as ShareableComputeDevice>::ShareParam>;
    type Status =
        Status<<<Device as ShareableComputeDevice>::RawComputeDevice as ComputeDevice>::Response>;
    type Command = Command<<Device as ShareableComputeDevice>::ShareParam>;
    type InputData = Work<Input>;

    type OutputData = Work<Output>;

    #[tracing::instrument(level = "trace", skip(self), fields(self.workers, self.worker_states))]
    async fn handle(&self, command: Self::Command) -> Result<Self::State, String> {
        if self
            .shutting_down
            .load(std::sync::atomic::Ordering::Acquire)
            && !matches!(command, Command::DestroyWorker { .. })
        {
            return Err("node is shutting down".into());
        }
        let id = match &command {
            Command::CreateWorker { id, .. }
            | Command::DestroyWorker { id }
            | Command::LoadModel { id, .. }
            | Command::UnloadModel { id } => *id,
        };
        // Weak references would allow a concurrent retry to acquire a different lock.
        // Retain the small ID registry for this node's lifetime.
        let lock = self.command_locks.entry(id).or_default().clone();
        let _guard = lock.lock().await;
        if let Some(receiver) = self.destroying.get(&id).map(|entry| entry.clone()) {
            CleanupHandle { receiver }.wait().await?;
            self.destroying.remove(&id);
        }
        match command {
            Command::CreateWorker {
                id,
                device_id,
                params,
            } => {
                if self.workers.contains_key(&id) {
                    let worker = self.worker_states.get(&id).unwrap();
                    warn!(
                        "Worker already exists: {:?}, state: {:?}",
                        id,
                        worker.value()
                    );
                    return if worker.value().device_id == device_id
                        && worker.value().params == params
                    {
                        self.state().await
                    } else {
                        Err("Worker already exists and in unexpected state".to_string())
                    };
                }
                let device = self
                    .devices
                    .get(&device_id)
                    .ok_or("Target device not found")?;
                let v_device = device
                    .share(params.clone())
                    .ok_or("Resource not enough for share")?
                    .map_err(|e| format!("Failed to share device: {e:?}"))?;

                let proxy = Proxy::new::<Worker>(
                    id,
                    v_device,
                    &self.model_dir,
                    &self.model_config_dir,
                    &self.bin_worker_path,
                )
                .await?;

                trace!("Created worker proxy: {:?}", proxy);

                // 存储 Worker 的状态信息
                self.worker_states.insert(
                    id,
                    WorkerState {
                        model_name: None,
                        params,
                        device_id,
                        experiment_noise_slot: crate::experiment_random::worker_noise_slot(id),
                    },
                );
                self.workers.insert(
                    id,
                    Arc::new(RwLock::new(ProxyCleaner::new(
                        proxy,
                        self.cleanup_tasks.clone(),
                    ))),
                );
            }
            Command::LoadModel { id, name } => {
                let proxy = self
                    .workers
                    .get(&id)
                    .ok_or("Target worker not found")?
                    .clone();
                proxy.write().await.load_model(name.clone()).await?;
                tracing::info!(target: "allocation", event = "model_loaded", %id, model = %name);
                if let Some(mut state) = self.worker_states.get_mut(&id) {
                    state.model_name = Some(name);
                } else {
                    warn!("Worker state not found for id: {}", id);
                }
            }
            Command::UnloadModel { id } => {
                let proxy = self
                    .workers
                    .get(&id)
                    .ok_or("Target worker not found")?
                    .clone();
                proxy.write().await.unload_model().await?;
                tracing::info!(target: "allocation", event = "model_unloaded", %id);
                if let Some(mut state) = self.worker_states.get_mut(&id) {
                    state.model_name = None;
                } else {
                    warn!("Worker state not found for id: {}", id);
                }
            }
            Command::DestroyWorker { id } => {
                let Some((_, worker)) = self.workers.remove(&id) else {
                    // A repeated destroy request is already satisfied.
                    return self.state().await;
                };
                self.worker_states.remove(&id);
                let (tx, receiver) = tokio::sync::watch::channel(None);
                self.destroying.insert(id, receiver.clone());
                // Keep cleanup alive if the request is cancelled. A retry joins the
                // same completion before reporting success or reusing this ID.
                tokio::spawn(async move {
                    let unload_result = worker.write().await.unload_model().await;
                    if unload_result.is_ok() {
                        tracing::info!(target: "allocation", event = "model_unloaded", %id);
                    }
                    let handler = worker.write().await.begin_cleanup();
                    drop(worker);
                    let cleanup_result = handler.wait().await;
                    if cleanup_result.is_ok() {
                        tracing::info!(target: "allocation", event = "worker_released", %id);
                    }
                    tx.send_replace(Some(cleanup_result.and(unload_result)));
                });
                CleanupHandle { receiver }.wait().await?;
                self.destroying.remove(&id);
            }
        }
        self.state().await
    }

    async fn state(&self) -> Result<Self::State, String> {
        let proxy_state = self
            .workers
            .iter()
            .filter_map(|entry| {
                let worker_id = *entry.key();
                self.worker_states
                    .get(&worker_id)
                    .map(|state_ref| async move { (worker_id, state_ref.value().clone()) })
            })
            .collect::<JoinAll<_>>()
            .await
            .into_iter()
            .collect();

        Ok(State {
            workers: proxy_state,
        })
    }

    async fn status(&self) -> Result<Self::Status, String> {
        let devices = <Device as ShareableComputeDevice>::RawComputeDevice::get_devices_state(
            self.devices
                .iter()
                .map(|(id, device)| (*id, device.raw_device())),
        );

        Ok(Status {
            worker_proxy_type: Proxy::PROXY_TYPE,
            devices,
        })
    }

    #[tracing::instrument(level = "trace", skip(self))]
    async fn compute(&self, work: Self::InputData) -> Result<Self::OutputData, String> {
        let Work {
            target_worker,
            work,
        } = work;

        let proxy = self
            .workers
            .get(&target_worker)
            .ok_or("Target worker not found")?
            .clone();
        let guard = proxy.read().await;
        let res = guard
            .proxy
            .as_ref()
            .ok_or("worker is being destroyed")?
            .compute(work.data)
            .await?;

        let work = BaseWork::<Output> {
            id: work.id,
            data: res,
            state: PhantomData,
        };

        Ok(Work {
            target_worker,
            work,
        })
    }
}

/// Device 管理助手
pub trait DeviceManageHelper<Device = Self, Res = <Device as ComputeDevice>::Response>
where
    Device: ComputeDevice,
    Res: Send + Sync + Encode + Decode<()>,
{
    /// 根据设备id获取设备
    fn get_device(id: usize) -> (Uuid, Device);

    /// 获取设备
    fn get_devices(
        device_ids: impl IntoIterator<Item = impl Into<usize>>,
    ) -> HashMap<Uuid, (usize, Device)> {
        device_ids
            .into_iter()
            .map(Into::into)
            .map(|id| {
                let (uuid, device) = Self::get_device(id);
                (uuid, (id, device))
            })
            .collect()
    }

    /// 获取设备状态
    fn get_device_state(device: impl Deref<Target = Device>) -> Device::Response;

    /// 获取设备状态
    fn get_devices_state(
        devices: impl Iterator<Item = (Uuid, impl Deref<Target = Device>)>,
    ) -> HashMap<Uuid, Device::Response> {
        devices
            .map(|(id, device)| {
                let state = Self::get_device_state(device.deref());
                (id, state)
            })
            .collect()
    }
}

impl<Worker, Device, Proxy> BinComputeNode for BaseVirtualComputeNode<Worker, Device, Proxy>
where
    Worker: worker::Worker
        + RunnableOn<<Device as ShareableComputeDevice>::SharedComputeDevice>
        + WorkerProxyCapable<<Device as ShareableComputeDevice>::SharedComputeDevice>
        + 'static,
    Device: ShareableComputeDevice + Send + Sync + 'static,
    <Device as ShareableComputeDevice>::RawComputeDevice:
        DeviceManageHelper<<Device as ShareableComputeDevice>::RawComputeDevice>,
    <<Device as ShareableComputeDevice>::RawComputeDevice as ComputeDevice>::Response:
        Encode + Decode<()> + Send + Sync + DeviceState,
    <Device as ShareableComputeDevice>::ShareParam:
        Serialize + DeserializeOwned + Encode + Decode<()> + Debug + Send + Sync,
    Proxy: WorkerProxy
        + BinSimpleWorkerProxyImplInit<<Device as ShareableComputeDevice>::SharedComputeDevice>
        + Send
        + Sync
        + 'static,
{
    async fn shutdown(&self) -> Result<(), String> {
        BaseVirtualComputeNode::shutdown(self).await
    }
}

#[cfg(test)]
mod noise_slot_tests {
    use super::WorkerState;

    #[test]
    fn optional_experiment_slot_keeps_legacy_json_and_roundtrips_through_rpc() {
        let mut state: WorkerState<String> = serde_json::from_value(serde_json::json!({
            "model_name": "resnet50", "params": "1g.10gb", "device_id": uuid::Uuid::nil()
        }))
        .unwrap();
        assert!(state.experiment_noise_slot.is_none());
        assert!(
            serde_json::to_value(&state)
                .unwrap()
                .get("experiment_noise_slot")
                .is_none()
        );
        state.experiment_noise_slot = Some("node/0/1g.10gb/2".into());
        let bytes = bincode::encode_to_vec(&state, bincode::config::standard()).unwrap();
        let (decoded, consumed): (WorkerState<String>, usize) =
            bincode::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
        assert_eq!(decoded, state);
        assert_eq!(consumed, bytes.len());
    }
}
