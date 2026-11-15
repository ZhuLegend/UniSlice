//! 基础计算节点

use std::{fmt::Debug, marker::PhantomData};

use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use worker::model::tensor::DynTensor;

use crate::util::{Input, Output, WorkState};

pub mod base;

pub mod virtual_compute_node;

/// 计算节点基础工作
#[derive(Debug, Serialize, Deserialize, Encode, Decode, Clone)]
pub struct BaseWork<State: WorkState> {
    /// Work id
    #[bincode(with_serde)]
    id: Uuid,
    /// Work data
    data: DynTensor,
    /// Work state
    state: PhantomData<fn() -> State>,
}

impl<State: WorkState> BaseWork<State> {
    /// [BaseWork]'s id
    pub fn id(&self) -> Uuid {
        self.id
    }

    /// [BaseWork]'s data
    pub fn data(&self) -> &DynTensor {
        &self.data
    }
}

impl BaseWork<Input> {
    /// create a new [BaseWork<Input>]
    pub fn new(id: Uuid, data: DynTensor) -> Self {
        Self {
            id,
            data,
            state: PhantomData,
        }
    }

    /// 在指定位置切分任务
    fn split_at_batch(self, split_point: usize) -> Result<(Self, Self), String> {
        // 检查切分点是否有效
        let batch_size = self.data.dimensions().first().copied().unwrap_or(0) as usize;
        if split_point >= batch_size {
            return Err(format!(
                "Split point {} exceeds batch size {}",
                split_point, batch_size
            ));
        }

        // 使用DynTensor的split_batch方法进行切分
        match self.data.split_batch(vec![split_point]) {
            Ok(mut parts) if parts.len() == 2 => {
                let remaining_part = parts.pop().unwrap();
                let split_part = parts.pop().unwrap();

                let split_work = BaseWork {
                    id: self.id,
                    data: split_part,
                    state: PhantomData,
                };

                let remaining_work = BaseWork {
                    id: Uuid::new_v4(), // 为剩余部分生成新ID
                    data: remaining_part,
                    state: PhantomData,
                };

                Ok((split_work, remaining_work))
            }
            Ok(_) => Err("Unexpected number of parts after split".to_string()),
            Err(e) => Err(format!("Failed to split tensor: {:?}", e)),
        }
    }
}

impl BaseWork<Output> {
    /// create a new [BaseWork<Output>]
    fn new(id: Uuid, data: DynTensor) -> Self {
        Self {
            id,
            data,
            state: PhantomData,
        }
    }

    /// 合并两个输出任务的结果
    fn merge_with(self, other: Self) -> Result<Self, String> {
        // 使用DynTensor的concat_batch方法合并结果
        match DynTensor::concat_batch(vec![self.data, other.data]) {
            Ok(merged_data) => Ok(BaseWork {
                id: self.id, // 保持原始ID
                data: merged_data,
                state: PhantomData,
            }),
            Err(e) => Err(format!("Failed to merge tensors: {:?}", e)),
        }
    }

    /// take the id and data from [`BaseWork<Output>`]
    pub fn take(self) -> (Uuid, DynTensor) {
        (self.id, self.data)
    }
}
