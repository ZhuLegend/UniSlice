//! 配置文件解析模块

use serde::{Deserialize, Serialize};
use tokio::fs;

/// 主配置结构
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SenderConfig {
    pub sender: SenderSettings,
    pub metrics: MetricsSettings,
    pub rate_change: RateChangeSettings,
    pub models: Vec<ModelConfig>,
}

/// 发送器基本设置
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SenderSettings {
    /// Daemon URL
    pub daemon_url: String,
    /// 日志输出目录
    pub log_dir: String,
    /// 状态收集间隔 (秒)
    pub stats_interval: u64,
    /// 速率变化周期 (秒)
    pub cycle_duration: u64,
}

/// 指标设置
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MetricsSettings {
    /// 速率统计窗口大小 (秒)
    pub rate_window_size: u64,
    /// 时延移动平均窗口大小
    pub latency_window_size: usize,
}

/// 速率变化策略
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RateChangeStrategy {
    Sudden,
    Smooth,
    RandomWalk,
}

/// 速率变化设置
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RateChangeSettings {
    /// 变化策略: "sudden" | "smooth" | "random_walk"
    pub strategy: RateChangeStrategy,
    /// 平滑过渡步数 (仅在smooth策略下有效)
    #[serde(default = "default_transition_steps")]
    pub transition_steps: u32,
    /// 随机游走步长百分比 (仅在random_walk策略下有效)
    #[serde(default = "default_walk_step_percent")]
    pub walk_step_percent: f64,
}

fn default_transition_steps() -> u32 {
    10 // 默认值
}

fn default_walk_step_percent() -> f64 {
    0.1 // 默认10%
}

/// 模型配置
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelConfig {
    /// 模型名称
    pub name: String,
    /// 样本处理速率范围 [min, max] (samples/sec)
    /// 注意：这是样本处理速率，不是任务发送速率
    /// 任务发送速率 = 样本处理速率 / batch_size
    pub rate_range: Vec<f64>,
    /// Batch size范围 [min, max] (每个任务包含的样本数)
    pub batch_size_range: Vec<u32>,
}

/// 加载配置文件
pub async fn load_config(config_path: &str) -> Result<SenderConfig, Box<dyn std::error::Error>> {
    let config_content = fs::read_to_string(config_path)
        .await
        .map_err(|e| format!("无法读取配置文件 '{}': {}", config_path, e))?;

    let config: SenderConfig =
        toml::from_str(&config_content).map_err(|e| format!("配置文件解析失败: {}", e))?;

    Ok(config)
}

/// 验证配置文件的有效性
pub fn validate_config(config: &SenderConfig) -> Result<(), String> {
    // 验证基本设置
    if config.sender.daemon_url.is_empty() {
        return Err("daemon_url不能为空".to_string());
    }

    if config.sender.stats_interval == 0 {
        return Err("stats_interval必须大于0".to_string());
    }

    if config.sender.cycle_duration == 0 {
        return Err("cycle_duration必须大于0".to_string());
    }

    // 验证指标设置
    if config.metrics.rate_window_size == 0 {
        return Err("rate_window_size必须大于0".to_string());
    }

    if config.metrics.latency_window_size == 0 {
        return Err("latency_window_size必须大于0".to_string());
    }

    // 验证速率变化设置
    match config.rate_change.strategy {
        RateChangeStrategy::Smooth => {
            if config.rate_change.transition_steps == 0 {
                return Err("smooth策略下transition_steps必须大于0".to_string());
            }
        }
        RateChangeStrategy::RandomWalk => {
            if config.rate_change.walk_step_percent <= 0.0
                || config.rate_change.walk_step_percent > 1.0
            {
                return Err("walk_step_percent必须在(0.0, 1.0]范围内".to_string());
            }
        }
        RateChangeStrategy::Sudden => {}
    }

    // 验证模型配置
    if config.models.is_empty() {
        return Err("至少需要配置一个模型".to_string());
    }

    for model in &config.models {
        if model.name.is_empty() {
            return Err("模型名称不能为空".to_string());
        }

        if model.rate_range.len() != 2 {
            return Err(format!("模型'{}'的rate_range必须包含2个值", model.name));
        }

        if model.rate_range.iter().any(|v| !v.is_finite())
            || model.rate_range[0] < 0.0
            || model.rate_range[1] < model.rate_range[0]
        {
            return Err(format!(
                "模型'{}'的样本处理速率范围无效: [{}, {}] samples/sec",
                model.name, model.rate_range[0], model.rate_range[1]
            ));
        }

        if model.batch_size_range.len() != 2 {
            return Err(format!(
                "模型'{}'的batch_size_range必须包含2个值",
                model.name
            ));
        }

        if model.batch_size_range[0] == 0 || model.batch_size_range[1] < model.batch_size_range[0] {
            return Err(format!(
                "模型'{}'的batch_size_range无效: [{}, {}] samples/task",
                model.name, model.batch_size_range[0], model.batch_size_range[1]
            ));
        }
    }

    Ok(())
}
