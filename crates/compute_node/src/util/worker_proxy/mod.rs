//! 工作代理

use std::{fmt::Debug, future::Future};

use bincode::{Decode, Encode};
use worker::model::tensor::DynTensor;

pub mod bin_simple_worker_proxy;

/// 工作代理
pub trait WorkerProxy: Sized + Debug {
    /// 请求类型
    type Request: Encode + Send;

    /// 响应类型
    type Response: Decode<()> + Send;

    /// 已加载的模型
    fn loaded_model(&self) -> Option<&String>;

    /// 加载模型
    fn load_model(&mut self, model_name: String)
    -> impl Future<Output = Result<(), String>> + Send;

    /// 卸载模型
    fn unload_model(&mut self) -> impl Future<Output = Result<(), String>> + Send;

    /// 执行计算
    fn compute(&self, input: DynTensor) -> impl Future<Output = Result<DynTensor, String>> + Send;

    /// 销毁工作代理
    fn destroy(self) -> impl Future<Output = Result<(), String>> + Send;
}
