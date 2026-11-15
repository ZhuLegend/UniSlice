//! Nvidia Multi-Instance Gpu (MIG) 支持

use std::{
    collections::HashSet,
    fs::File,
    process::{Command, Stdio},
    sync::Arc,
    thread::sleep,
    time::Duration,
};

use file_locking::FileExt;
use regex::Regex;
use tracing::{trace, warn};

use crate::compute_device::{
    ComputeDevice,
    nv::{NvidiaGpu, nvml},
    vgpu::resource_spec::nv::mig::VDeviceScale,
};

/// 检查指定GPU是否启用MIG
#[tracing::instrument(level = "trace")]
#[allow(dead_code)]
pub(crate) async fn is_mig_supported(gpu_id: u32) -> bool {
    let nvml = nvml();
    if let Ok(device) = nvml.device_by_index(gpu_id)
        && let Ok(mig_mode) = device.mig_mode()
    {
        return mig_mode.current == 1;
    }
    false
}

/// Nvidia MIG 切分输出
pub struct VDeviceOutput {
    /// GPU实例ID
    pub gpu_instance_id: u32,
    /// 计算实例ID
    pub compute_instance_id: u32,
    /// MIG UUID
    pub uuid: String,
    /// 切分大小
    pub scale: VDeviceScale,
}

/// 算力切分函数
#[tracing::instrument(level = "trace")]
pub(crate) fn create_mig_vdevice(
    host_device: Arc<NvidiaGpu>,
    scale: VDeviceScale,
) -> Result<VDeviceOutput, String> {
    let gpu_id = host_device.logic_index().unwrap();

    let lock_path = std::env::temp_dir().join(format!("hsgs.mig_gpu_{}.lock", gpu_id));
    let lock_file = File::options()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&lock_path)
        .map_err(|e| e.to_string())?;
    let lock = lock_file.lock_exclusive().map_err(|e| e.to_string())?;

    let before_output = Command::new("nvidia-smi")
        .args(["-L"])
        .output()
        .map_err(|e| e.to_string())?;

    let output = Command::new("nvidia-smi")
        .args([
            "mig",
            "-i",
            &gpu_id.to_string(),
            "-cgi",
            &match scale {
                VDeviceScale::A100(scale) => scale.id(),
            }
            .to_string(),
            "-C",
        ])
        .output()
        .map_err(|e| e.to_string())?;

    let after_output = Command::new("nvidia-smi")
        .args(["-L"])
        .output()
        .map_err(|e| e.to_string())?;

    drop(lock);

    let (gpu_instance_id, compute_instance_id) =
        parse_gi_and_ci(std::str::from_utf8(&output.stdout).map_err(|e| e.to_string())?)
            .ok_or_else(|| "Failed to parse GPU and compute instance IDs".to_string())?;

    let uuid = diff_new_mig_uuids(
        std::str::from_utf8(&before_output.stdout).map_err(|e| e.to_string())?,
        std::str::from_utf8(&after_output.stdout).map_err(|e| e.to_string())?,
        gpu_id,
    )
    .first()
    .cloned()
    .ok_or_else(|| "No new MIG UUID found".to_string())?;

    Ok(VDeviceOutput {
        gpu_instance_id,
        compute_instance_id,
        uuid,
        scale,
    })
}

/// 解析输出中的 GPU 实例 ID 和计算实例 ID
/// # Example output:
/// Successfully created GPU instance ID  6 on GPU  7 using profile MIG 1g.20gb (ID 15)
/// Successfully created compute instance ID  0 on GPU  7 GPU instance ID  6 using profile MIG 1g.20gb (ID  0)
fn parse_gi_and_ci(output: &str) -> Option<(u32, u32)> {
    let lines: Vec<_> = output.lines().collect();
    trace!("nvidia-smi mig output lines: {:?}", lines);
    let gi = lines
        .first()
        .and_then(|line| line.split_whitespace().nth(5))
        .and_then(|s| s.parse::<u32>().ok());
    let ci = lines
        .get(1)
        .and_then(|line| line.split_whitespace().nth(5))
        .and_then(|s| s.parse::<u32>().ok());
    if let (Some(gpu_instance_id), Some(compute_instance_id)) = (gi, ci) {
        Some((gpu_instance_id, compute_instance_id))
    } else {
        None
    }
}

/// 从 `nvidia-smi -L` 的输出中为特定的 GPU 提取 MIG UUIDs
fn extract_mig_uuids_for_gpu(output: &str, gpu_id: u32) -> HashSet<String> {
    let re = Regex::new(r"MIG-[a-f0-9\-]+").unwrap();
    let gpu_line_prefix = format!("GPU {}:", gpu_id);
    let mut in_gpu_block = false;
    let mut gpu_block_lines = Vec::new();

    for line in output.lines() {
        if line.starts_with(&gpu_line_prefix) {
            in_gpu_block = true;
            // MIG devices are on subsequent lines, so we don't add this line
            continue;
        }

        if in_gpu_block {
            // If we've hit the next GPU or the end of the relevant section
            if line.starts_with("GPU ") || !line.starts_with("  ") {
                break;
            }
            gpu_block_lines.push(line);
        }
    }

    let gpu_block = gpu_block_lines.join("\n");
    re.find_iter(&gpu_block)
        .map(|m| m.as_str().to_string())
        .collect()
}

/// 比较两组 nvidia-smi -L 输出，提取指定 GPU 上新增的 MIG UUID
fn diff_new_mig_uuids(before: &str, after: &str, gpu_id: u32) -> Vec<String> {
    let before_uuids = extract_mig_uuids_for_gpu(before, gpu_id);
    let after_uuids = extract_mig_uuids_for_gpu(after, gpu_id);

    after_uuids.difference(&before_uuids).cloned().collect()
}

fn try_command(command: &mut Command, max_attempt: u32) -> Result<(), String> {
    let mut attempt = 0;

    while attempt < max_attempt {
        let mut child = command.spawn().map_err(|e| e.to_string())?;
        let status = child.wait().map_err(|e| e.to_string())?;
        if status.success() {
            return Ok(());
        } else {
            if status.code() == Some(6) {
                // 6 for nvidia-smi means "No MIG devices found", which is expected in this cases
                return Ok(());
            }
            let output = child.wait_with_output().map_err(|e| e.to_string());
            trace!(?status, ?output, attempt, "Command failed");
        }
        attempt += 1;
        sleep(Duration::from_millis(500 * 2_u64.pow(attempt)));
    }
    Err("All attempts to execute command failed".to_string())
}

/// 销毁指定的 MIG 虚拟设备
#[tracing::instrument(level = "trace")]
pub(crate) fn destroy_mig_vdevice(gpu_id: String, gi: String, ci: String) -> Result<(), String> {
    // 销毁 CI
    try_command(
        Command::new("nvidia-smi")
            .args(["mig", "-dci", "-i", &gpu_id, "-gi", &gi, "-ci", &ci])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped()),
        5,
    )?;
    // 销毁 GI
    try_command(
        Command::new("nvidia-smi")
            .args(["mig", "-dgi", "-i", &gpu_id, "-gi", &gi])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped()),
        5,
    )
    .map_err(|e| format!("nvidia-smi dgi failed: {}", e))?;

    Ok(())
}
