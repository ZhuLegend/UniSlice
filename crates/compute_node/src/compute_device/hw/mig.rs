//! Huawei VChip 算力切分 支持

use hw_dcmi_wrapper::{
    device::{chip::Chip, vchip::VChip},
    enums::VChipCreateParam,
    error::DCMIResult,
    structs::VChipOutput,
};
use tokio::process::Command;

use crate::compute_device::vgpu::resource_spec::hw::VChipScale;

/// 检查npu-smi是否存在
#[tracing::instrument(level = "trace")]
#[allow(dead_code)]
pub(crate) async fn is_npu_smi_exist() -> bool {
    let output_absolute = Command::new("/usr/local/sbin/npu-smi")
        .arg("--version")
        .output()
        .await;
    let output_relative = Command::new("npu-smi").arg("--version").output().await;

    output_absolute.is_ok() || output_relative.is_ok()
}

/// 算力切分函数
#[tracing::instrument(level = "trace")]
pub(crate) fn create_vchip<'a, 'b, 'c>(
    chip: &'a Chip<'b, 'c>,
    config: VChipScale,
) -> DCMIResult<(VChipOutput, VChip<'a, 'b, 'c>)>
where
    'b: 'a,
    'c: 'b,
{
    let template_name = config.name();
    let param = VChipCreateParam::TemplateName(template_name.to_string());
    VChip::create(chip, param)
}
