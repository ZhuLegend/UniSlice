//! 错误显示模块
//!
//! 提供统一的错误输出接口，确保错误信息正确显示在indicatif进度条上方

use std::sync::{LazyLock, Mutex};

use indicatif::ProgressBar;

/// 全局错误显示器
static ERROR_DISPLAY: LazyLock<ErrorDisplay> = LazyLock::new(ErrorDisplay::new);

/// 错误显示器
pub struct ErrorDisplay {
    /// 用于输出的进度条
    progress_bar: Mutex<Option<ProgressBar>>,
}

impl ErrorDisplay {
    /// 创建新的错误显示器
    pub fn new() -> Self {
        Self {
            progress_bar: Mutex::new(None),
        }
    }

    /// 设置用于输出的进度条
    pub fn set_progress_bar(&self, pb: ProgressBar) {
        if let Ok(mut guard) = self.progress_bar.lock() {
            *guard = Some(pb);
        }
    }

    /// 输出错误信息
    pub fn print_error(&self, message: &str) {
        if let Ok(guard) = self.progress_bar.lock() {
            if let Some(ref pb) = *guard {
                pb.println(format!("❌ 错误: {}", message));
            } else {
                // 如果没有进度条，回退到标准错误输出
                eprintln!("❌ 错误: {}", message);
            }
        } else {
            eprintln!("❌ 错误: {}", message);
        }
    }
}

/// 设置全局错误显示器的进度条
pub fn set_global_progress_bar(pb: ProgressBar) {
    ERROR_DISPLAY.set_progress_bar(pb);
}

/// 输出全局错误信息
pub fn print_error(message: &str) {
    ERROR_DISPLAY.print_error(message);
}

/// 便于使用的宏，类似于tracing的error!宏
#[macro_export]
macro_rules! display_error {
    ($($arg:tt)*) => {
        $crate::error_display::print_error(&format!($($arg)*))
    };
}

/// 便于使用的宏，类似于tracing的warn!宏
#[macro_export]
macro_rules! display_warning {
    ($($arg:tt)*) => {
        $crate::error_display::print_warning(&format!($($arg)*))
    };
}

/// 便于使用的宏，类似于tracing的info!宏但输出到进度条上方
#[macro_export]
macro_rules! display_info {
    ($($arg:tt)*) => {
        $crate::error_display::print_info(&format!($($arg)*))
    };
}
