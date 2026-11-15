//! 实用工具
//!
//! - [sync] 支持取消传递的[sync::Task]

#![warn(missing_docs)]

use std::fmt::{Display, Formatter};

pub mod sync;

/// 文件系统模块
pub mod fs {
    use std::{
        ops::{Deref, DerefMut},
        path::PathBuf,
    };

    #[cfg(feature = "clap")]
    pub use clap_ext::ValidDirPathParser;

    /// 已确认为为有效文件夹路径的PathBuf
    #[derive(Clone)]
    pub struct DirPathBuf(PathBuf);

    impl Deref for DirPathBuf {
        type Target = PathBuf;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl DerefMut for DirPathBuf {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.0
        }
    }

    impl From<DirPathBuf> for PathBuf {
        fn from(value: DirPathBuf) -> Self {
            value.0
        }
    }

    #[cfg(feature = "clap")]
    mod clap_ext {
        use std::{ffi::OsStr, fs::metadata, path::PathBuf};

        use clap::{
            Arg, Command, Error,
            builder::TypedValueParser,
            error::{ContextKind, ContextValue, ErrorKind},
        };

        use crate::fs::DirPathBuf;

        /// 判断命令行输入参数是否为有效的文件夹路径
        #[derive(Clone)]
        pub struct ValidDirPathParser;

        impl TypedValueParser for ValidDirPathParser {
            type Value = DirPathBuf;

            fn parse_ref(
                &self,
                cmd: &Command,
                arg: Option<&Arg>,
                value: &OsStr,
            ) -> Result<Self::Value, Error> {
                let inner = clap::builder::PathBufValueParser::new();
                let value: PathBuf = inner.parse_ref(cmd, arg, value)?;

                let metadata = metadata(&value).map_err(Error::from)?;

                if !metadata.is_dir() {
                    let mut err = Error::new(ErrorKind::ValueValidation);
                    err.insert(
                        ContextKind::Custom,
                        ContextValue::String("Given Path is not a valid dir".to_owned()),
                    );
                    return Err(err);
                }

                Ok(DirPathBuf(value))
            }
        }
    }
}

#[cfg(test)]
mod test;

/// Error Message
#[derive(Debug)]
pub struct ErrorMessage(String);

impl ErrorMessage {
    /// Create a new [ErrorMessage]
    pub fn new(msg: String) -> Self {
        ErrorMessage(msg)
    }
}

impl Display for ErrorMessage {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.0, f)
    }
}

impl std::error::Error for ErrorMessage {}

impl<T: AsRef<str>> From<T> for ErrorMessage {
    fn from(value: T) -> Self {
        ErrorMessage(value.as_ref().to_string())
    }
}
