//! 可绑定到特定[ComputeDevice]上运行的[Worker] trait

use worker::{Worker, model::model::InferenceModel};

use crate::compute_device::ComputeDevice;

/// 可绑定到特定[ComputeDevice]上运行的[Worker]
pub trait RunnableOn<T: ComputeDevice, W: Worker = Self> {
    /// 设备无关的[Worker]加载配置
    type DeviceIndependentConfig: IndependentModelLoadConfig<
        TargetLoadConfig: Into<<W::Model as InferenceModel>::LoadConfig>,
    >;

    /// 为[IndependentModelLoadConfig]提供目标设备信息的函数
    fn device_config_getter(
        device: &T,
    ) -> impl FnOnce() -> <Self::DeviceIndependentConfig as IndependentModelLoadConfig>::RequireDevice;

    /// 从设备无关的[Worker]加载配置生成设备相关的[Worker]加载配置
    fn generate_config(
        device: &T,
        config: Self::DeviceIndependentConfig,
    ) -> impl Into<<W::Model as InferenceModel>::LoadConfig> {
        let device_config_getter = Self::device_config_getter(device);
        config.with_device(device_config_getter)
    }
}

/// 独立于设备的模型加载配置
pub trait IndependentModelLoadConfig {
    /// 需要转换的目标设备类型
    type RequireDevice;

    /// 转换后的目标加载配置
    type TargetLoadConfig;

    /// 将当前配置转换为目标设备的加载配置
    fn with_device(self, getter: impl FnOnce() -> Self::RequireDevice) -> Self::TargetLoadConfig;
}
