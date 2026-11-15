//! 计算节点模块

#![warn(missing_docs)]

pub mod compute_device;

pub mod util;

pub mod compute_node;

#[cfg(feature = "bin_help")]
pub mod model_provider;

pub mod runnable_on;

#[cfg(feature = "vgpu")]
pub mod experiment_random;

#[cfg(feature = "mock")]
pub mod simulation;

/// 计算节点抽象
#[trait_variant::make(Send)]
pub trait ComputeNode {
    /// 计算节点可以配置的状态
    type State: Send + Sync;

    /// 计算节点无法配置的状态
    type Status: Send + Sync;

    /// 发给计算节点的命令
    type Command: Send + Sync;

    /// 计算节点用于计算的数据
    type InputData: Send + Sync;

    /// 计算节点计算结果输出
    type OutputData: Send + Sync;

    /// 处理命令
    async fn handle(&self, command: Self::Command) -> Result<Self::State, String>;

    /// 获取可配置的状态
    async fn state(&self) -> Result<Self::State, String>;

    /// 获取不可配置的状态
    async fn status(&self) -> Result<Self::Status, String>;

    /// 执行任务推理
    async fn compute(&self, work: Self::InputData) -> Result<Self::OutputData, String>;
}
