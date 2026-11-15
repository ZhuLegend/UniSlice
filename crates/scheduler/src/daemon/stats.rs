//! 调度器统计管理模块

use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use tokio::{fs::OpenOptions, io::AsyncWriteExt};

use crate::daemon::utils;

/// 调度器统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerStats {
    /// 调度周期计数
    pub cycle_count: u64,
    /// 成功调度次数
    pub successful_schedules: u64,
    /// 失败调度次数
    pub failed_schedules: u64,
    /// 平均调度时间（毫秒）
    pub avg_schedule_time_ms: f64,
    /// 最后一次调度时间戳
    pub last_schedule_timestamp: Option<u64>,
}

/// 调度器统计管理器
#[derive(Clone)]
pub struct SchedulerStatsManager {
    /// 调度周期计数
    cycle_count: Arc<Mutex<u64>>,
    /// 成功调度次数
    successful_schedules: Arc<Mutex<u64>>,
    /// 失败调度次数
    failed_schedules: Arc<Mutex<u64>>,
    /// 调度时间记录
    schedule_times: Arc<Mutex<Vec<u64>>>,
    /// 最后一次调度时间戳
    last_schedule_timestamp: Arc<Mutex<Option<u64>>>,
    /// CSV文件路径
    csv_path: std::path::PathBuf,
}

impl SchedulerStatsManager {
    /// 创建新的调度器统计管理器
    pub async fn new(
        log_dir: &std::path::Path,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // 确保日志目录存在
        tokio::fs::create_dir_all(log_dir).await?;

        // 生成CSV文件路径
        let csv_path = log_dir.join("scheduler_stats_log.csv");

        // 检查文件是否存在，如果不存在则创建并写入表头
        if !csv_path.exists() {
            let mut file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&csv_path)
                .await?;

            // 写入CSV表头
            file.write_all(b"timestamp,cycle_duration_ms,success,total_cycles,success_rate\n")
                .await?;
            file.flush().await?;
        }

        Ok(Self {
            cycle_count: Arc::new(Mutex::new(0)),
            successful_schedules: Arc::new(Mutex::new(0)),
            failed_schedules: Arc::new(Mutex::new(0)),
            schedule_times: Arc::new(Mutex::new(Vec::new())),
            last_schedule_timestamp: Arc::new(Mutex::new(None)),
            csv_path,
        })
    }

    /// 记录调度周期开始
    pub fn record_schedule_start(&self) -> std::time::Instant {
        std::time::Instant::now()
    }

    /// 记录调度周期结束
    pub async fn record_schedule_end(&self, start_time: std::time::Instant, success: bool) {
        let elapsed = start_time.elapsed().as_millis() as u64;
        let timestamp = utils::current_timestamp();

        {
            let mut cycle_count = self.cycle_count.lock().unwrap();
            *cycle_count += 1;
        }

        if success {
            let mut successful = self.successful_schedules.lock().unwrap();
            *successful += 1;
        } else {
            let mut failed = self.failed_schedules.lock().unwrap();
            *failed += 1;
        }

        {
            let mut times = self.schedule_times.lock().unwrap();
            times.push(elapsed);
            // 保留最近100次记录
            if times.len() > 100 {
                let len = times.len();
                times.drain(..len - 100);
            }
        }

        {
            let mut last_timestamp = self.last_schedule_timestamp.lock().unwrap();
            *last_timestamp = Some(timestamp);
        }

        // 记录到CSV文件
        self.log_schedule_to_csv(timestamp, elapsed, success).await;

        tracing::info!(
            "调度周期完成: 耗时={}ms, 成功={}, 时间戳={}",
            elapsed,
            success,
            timestamp
        );
    }

    /// 记录调度统计到CSV文件
    async fn log_schedule_to_csv(&self, timestamp: u64, duration_ms: u64, success: bool) {
        let cycle_count = *self.cycle_count.lock().unwrap();
        let successful_schedules = *self.successful_schedules.lock().unwrap();
        let success_rate = if cycle_count > 0 {
            successful_schedules as f64 / cycle_count as f64 * 100.0
        } else {
            0.0
        };

        let log_entry = format!(
            "{},{},{},{},{:.2}\n",
            timestamp, duration_ms, success, cycle_count, success_rate
        );

        // 异步写入CSV文件
        let csv_path = self.csv_path.clone();
        tokio::spawn(async move {
            if let Ok(mut file) = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&csv_path)
                .await
            {
                if let Err(e) = file.write_all(log_entry.as_bytes()).await {
                    tracing::error!("写入调度统计CSV失败: {}", e);
                } else if let Err(e) = file.flush().await {
                    tracing::error!("刷新调度统计CSV失败: {}", e);
                }
            }
        });

        tracing::debug!(
            "记录调度统计: 时间戳={}, 耗时={}ms, 成功={}, 总周期={}",
            timestamp,
            duration_ms,
            success,
            cycle_count
        );
    }

    /// 获取调度器统计信息
    pub fn get_stats(&self) -> SchedulerStats {
        let cycle_count = *self.cycle_count.lock().unwrap();
        let successful_schedules = *self.successful_schedules.lock().unwrap();
        let failed_schedules = *self.failed_schedules.lock().unwrap();
        let last_schedule_timestamp = *self.last_schedule_timestamp.lock().unwrap();

        let avg_schedule_time_ms = {
            let times = self.schedule_times.lock().unwrap();
            if times.is_empty() {
                0.0
            } else {
                times.iter().sum::<u64>() as f64 / times.len() as f64
            }
        };

        SchedulerStats {
            cycle_count,
            successful_schedules,
            failed_schedules,
            avg_schedule_time_ms,
            last_schedule_timestamp,
        }
    }

    /// 重置统计信息
    pub fn reset_stats(&self) {
        *self.cycle_count.lock().unwrap() = 0;
        *self.successful_schedules.lock().unwrap() = 0;
        *self.failed_schedules.lock().unwrap() = 0;
        self.schedule_times.lock().unwrap().clear();
        *self.last_schedule_timestamp.lock().unwrap() = None;

        tracing::info!("调度器统计信息已重置");
    }

    /// 获取CSV文件路径
    pub fn csv_path(&self) -> &std::path::Path {
        &self.csv_path
    }
}
