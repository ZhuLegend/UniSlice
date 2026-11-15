//! CSV任务记录器模块，用于记录任务完成统计

use std::path::{Path, PathBuf};

use tokio::{
    fs::OpenOptions,
    io::AsyncWriteExt,
    sync::{mpsc, oneshot},
    time::{Duration, Instant},
};

use crate::daemon::types::{RequestPermit, TaskStats, TaskStatus};

/// CSV日志条目
#[derive(Debug)]
pub struct LogEntry {
    task_id: String,
    status: TaskStatus,
    error: Option<String>,
    // Retain admission until this terminal record has been flushed to disk.
    _permit: Option<RequestPermit>,
    /// 任务对应的模型名称
    pub model_name: String,
    /// 任务接收时间戳（单位：毫秒）
    pub received_at: u64,
    /// 任务处理耗时（单位：毫秒）
    pub processing_duration_ms: u64,
    /// 批量大小（单位：条目数）
    pub batch_size: u64,
}

/// 批量CSV任务记录器
#[derive(Clone)]
pub struct CsvTaskLogger {
    /// 日志发送器
    log_sender: mpsc::UnboundedSender<LogMessage>,
}

enum LogMessage {
    Entry(LogEntry),
    Flush(oneshot::Sender<Result<(), String>>),
}

fn csv_field(value: &str) -> String {
    if value.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}

/// 批量日志写入器
struct BatchLogWriter {
    /// CSV文件路径
    csv_path: PathBuf,
    /// 日志缓冲区
    buffer: Vec<LogEntry>,
    /// 上次刷新时间
    last_flush: Instant,
    /// 刷新间隔
    flush_interval: Duration,
    /// 最大批量大小
    max_batch_size: usize,
}

impl BatchLogWriter {
    /// 创建新的批量日志写入器
    async fn new(
        csv_path: PathBuf,
        flush_interval: Duration,
        max_batch_size: usize,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // 检查文件是否存在，如果不存在则创建并写入表头
        if !csv_path.exists() {
            let mut file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&csv_path)
                .await?;

            // 写入CSV表头
            file.write_all(
                b"model_name,received_at,processing_duration_ms,batch_size,task_id,status,error\n",
            )
            .await?;
            file.flush().await?;
        }

        Ok(Self {
            csv_path,
            buffer: Vec::with_capacity(max_batch_size),
            last_flush: Instant::now(),
            flush_interval,
            max_batch_size,
        })
    }

    /// 添加日志条目到缓冲区
    fn add_entry(&mut self, entry: LogEntry) {
        self.buffer.push(entry);
    }

    /// 检查是否需要刷新
    fn should_flush(&self) -> bool {
        self.buffer.len() >= self.max_batch_size || self.last_flush.elapsed() >= self.flush_interval
    }

    /// 刷新缓冲区到文件
    async fn flush(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.buffer.is_empty() {
            return Ok(());
        }

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.csv_path)
            .await?;

        let mut content = String::new();
        for entry in &self.buffer {
            content.push_str(&format!(
                "{},{},{},{},{},{},{}\n",
                csv_field(&entry.model_name),
                entry.received_at,
                entry.processing_duration_ms,
                entry.batch_size,
                csv_field(&entry.task_id),
                if entry.status == TaskStatus::Completed {
                    "completed"
                } else {
                    "failed"
                },
                csv_field(entry.error.as_deref().unwrap_or(""))
            ));
        }

        file.write_all(content.as_bytes()).await?;
        file.flush().await?;

        tracing::debug!("批量写入 {} 条CSV日志记录", self.buffer.len());

        self.buffer.clear();
        self.last_flush = Instant::now();

        Ok(())
    }

    /// 启动日志写入循环
    async fn run(
        mut self,
        mut log_receiver: mpsc::UnboundedReceiver<LogMessage>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut flush_timer = tokio::time::interval(self.flush_interval);

        loop {
            tokio::select! {
                // 接收新的日志条目
                entry = log_receiver.recv() => {
                    match entry {
                        Some(LogMessage::Entry(entry)) => {
                            self.add_entry(entry);

                            // 如果缓冲区满了，立即刷新
                            if self.should_flush() && let Err(e) = self.flush().await {
                                tracing::error!("批量刷新CSV日志失败: {}", e);
                            }
                        }
                        Some(LogMessage::Flush(reply)) => {
                            let result = self.flush().await.map_err(|error| error.to_string());
                            let _ = reply.send(result);
                        }
                        None => {
                            // 通道关闭，刷新剩余数据并退出
                            if let Err(e) = self.flush().await {
                                tracing::error!("最终刷新CSV日志失败: {}", e);
                            }
                            break;
                        }
                    }
                }

                // 定时刷新
                _ = flush_timer.tick() => {
                    if self.should_flush() && let Err(e) = self.flush().await {
                        tracing::error!("定时刷新CSV日志失败: {}", e);
                    }
                }
            }
        }

        tracing::info!("CSV日志写入器已停止");
        Ok(())
    }
}

impl CsvTaskLogger {
    /// 创建新的CSV任务记录器
    pub async fn new(log_dir: &Path) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // 确保日志目录存在
        tokio::fs::create_dir_all(log_dir).await?;

        // 生成CSV文件路径
        let csv_path = log_dir.join("task_terminal_log.csv");

        // 创建通道用于日志传递
        let (log_sender, log_receiver) = mpsc::unbounded_channel();

        // 配置批量写入参数
        let flush_interval = Duration::from_secs(5); // 每5秒刷新一次
        let max_batch_size = 100; // 最大批量大小

        // 创建批量日志写入器
        let writer = BatchLogWriter::new(csv_path, flush_interval, max_batch_size).await?;

        // 启动后台日志写入任务
        tokio::spawn(async move {
            if let Err(e) = writer.run(log_receiver).await {
                tracing::error!("CSV日志写入器运行失败: {}", e);
            }
        });

        Ok(Self { log_sender })
    }

    /// 记录任务完成统计
    pub async fn log_task_completion(&self, task_stats: &TaskStats) {
        self.log_terminal(task_stats, None);
    }

    /// Enqueue a terminal record together with its retained admission budget.
    pub fn log_terminal(&self, task_stats: &TaskStats, permit: Option<RequestPermit>) {
        if let Some(completed_at) = task_stats.completed_at {
            let processing_duration = completed_at.saturating_sub(task_stats.received_at);

            let log_entry = LogEntry {
                task_id: task_stats.task_id.clone(),
                status: task_stats.status.clone(),
                error: task_stats.error.clone(),
                _permit: permit,
                model_name: task_stats.model.clone(),
                received_at: task_stats.received_at,
                processing_duration_ms: processing_duration,
                batch_size: task_stats.batch_size,
            };

            // 发送到批量写入器
            if let Err(e) = self.log_sender.send(LogMessage::Entry(log_entry)) {
                tracing::error!("发送CSV日志条目失败：{}", e);
            } else {
                tracing::debug!(
                    "记录任务完成: 模型={}, 接收时间={}, 处理耗时={}ms",
                    task_stats.model,
                    task_stats.received_at,
                    processing_duration
                );
            }
        }
    }
    /// Flush all records enqueued before this call, including cancellation records.
    pub async fn flush(&self) -> Result<(), String> {
        let (reply, result) = oneshot::channel();
        self.log_sender
            .send(LogMessage::Flush(reply))
            .map_err(|e| e.to_string())?;
        result.await.map_err(|e| e.to_string())?
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;
    use crate::daemon::types::{TaskStats, TaskStatus};

    #[tokio::test]
    async fn test_csv_logger_task_completion() {
        let temp_dir = tempdir().unwrap();
        let logger = CsvTaskLogger::new(temp_dir.path()).await.unwrap();

        let task_stats = TaskStats {
            task_id: "test-task-1".to_string(),
            model: "test-model".to_string(),
            batch_size: 1,
            received_at: 1000,
            completed_at: Some(2000),
            status: TaskStatus::Completed,
            error: None,
        };

        logger.log_task_completion(&task_stats).await;

        // 等待批量写入完成 - 需要更长时间因为是批量处理
        logger.flush().await.unwrap();

        let csv_path = temp_dir.path().join("task_terminal_log.csv");
        let content = tokio::fs::read_to_string(&csv_path).await.unwrap();
        assert!(content.contains("test-model,1000,1000"));
    }

    #[tokio::test]
    async fn test_csv_logger_batch_writing() {
        let temp_dir = tempdir().unwrap();
        let logger = CsvTaskLogger::new(temp_dir.path()).await.unwrap();

        // 发送多个任务完成记录
        for i in 0..5 {
            let task_stats = TaskStats {
                task_id: format!("test-task-{}", i),
                model: format!("test-model-{}", i),
                batch_size: 1,
                received_at: 1000 + i as u64 * 100,
                completed_at: Some(2000 + i as u64 * 100),
                status: TaskStatus::Completed,
                error: None,
            };
            logger.log_task_completion(&task_stats).await;
        }

        // 等待批量写入完成
        logger.flush().await.unwrap();

        let csv_path = temp_dir.path().join("task_terminal_log.csv");
        let content = tokio::fs::read_to_string(&csv_path).await.unwrap();

        // 验证所有记录都被写入
        for i in 0..5 {
            assert!(content.contains(&format!("test-model-{}", i)));
        }
    }
}
