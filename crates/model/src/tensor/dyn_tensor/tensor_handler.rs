use std::fmt::Debug;

use ndarray::{Array, ArrayD, Axis, Dimension, Slice};
use snafu::{Snafu, ensure};

use crate::tensor::{TensorOperationError::ElementTypeMismatch, dyn_tensor::EraserData};

pub type Result<T, E = TensorOperationError> = std::result::Result<T, E>;

/// 定义统一的操作Trait
pub(crate) trait TensorHandler: 'static + Send + Sync {
    fn concat(&self, tensors: Vec<Box<EraserData>>) -> Result<Box<EraserData>>;
    fn clone(&self, tensor: &EraserData) -> Box<EraserData>;
    fn split_batch(
        &self,
        tensor: Box<EraserData>,
        indexes: Vec<usize>,
    ) -> Result<Vec<Box<EraserData>>>;
}

/// 具体类型的处理器实现
#[derive(Debug)]
pub(crate) struct ConcreteArrayHandler<T: 'static + Send + Sync>(std::marker::PhantomData<T>);

impl<T: 'static + Send + Sync> Default for ConcreteArrayHandler<T> {
    fn default() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<T: 'static + Send + Sync + Clone> TensorHandler for ConcreteArrayHandler<T> {
    fn concat(&self, tensors: Vec<Box<EraserData>>) -> Result<Box<EraserData>> {
        ensure!(
            !tensors.is_empty(),
            TargetInvalidSnafu {
                message: "No tensors to concatenate".to_string()
            }
        );

        ensure!(
            tensors.iter().all(|t| t.is::<ArrayD<T>>()),
            ElementTypeMismatchSnafu {
                message: "Tensor element type mismatch".to_string()
            }
        );

        let tensors = tensors.into_iter().map(|tensor| {
            *tensor.downcast::<ArrayD<T>>().unwrap() // 由于前面已经检查过类型，因此这里可以直接unwrap
        });

        concat_batches(tensors).map(|arr| Box::new(arr) as Box<EraserData>)
    }

    fn clone(&self, tensor: &EraserData) -> Box<EraserData> {
        Box::new(tensor.downcast_ref::<ArrayD<T>>().unwrap().clone())
    }

    fn split_batch(
        &self,
        tensor: Box<EraserData>,
        indexes: Vec<usize>,
    ) -> Result<Vec<Box<EraserData>>> {
        if indexes.is_empty() {
            return Ok(vec![tensor]);
        }

        let arr = tensor
            .downcast::<ArrayD<T>>()
            .map_err(|err| ElementTypeMismatch {
                message: format!(
                    "Want tensor element type {}, but got {}",
                    std::any::type_name::<T>(),
                    std::any::type_name_of_val(&err)
                ),
            })?;
        let data_results = split_batch(*arr, indexes)?;
        let results = data_results
            .into_iter()
            .map(|result| Box::new(result) as Box<EraserData>)
            .collect();
        Ok(results)
    }
}

// Tensor操作错误
#[derive(Debug, Snafu)]
pub enum TensorOperationError {
    /// 操作目标无效
    #[snafu(display("Tensor target invalid: {}", message))]
    TargetInvalid { message: String },

    /// 元素类型不匹配
    #[snafu(display("Tensor element type mismatch: {}", message))]
    ElementTypeMismatch { message: String },

    /// 维度无效
    #[snafu(display("Invalid dimension : {}", message))]
    InvalidDimension { message: String },

    /// 索引无效
    #[snafu(display("Invalid index: {}", message))]
    InvalidIndex { message: String },
}

/// Concatenate owned contiguous buffers without cloning the first batch.
/// Non-standard layouts use ndarray's logical indexing.
pub(crate) fn concat_batches<A: Clone, D: Dimension + ndarray::RemoveAxis>(
    arrs: impl IntoIterator<Item = Array<A, D>>,
) -> Result<Array<A, D>> {
    let mut arrs = arrs.into_iter().collect::<Vec<_>>();
    if arrs.len() == 1 {
        return Ok(arrs.pop().unwrap());
    }
    if !arrs.is_empty() && arrs.iter().all(Array::is_standard_layout) {
        let mut shape = arrs[0].raw_dim();
        ensure!(
            shape.ndim() > 0,
            InvalidDimensionSnafu {
                message: "A scalar has no batch axis".to_string()
            }
        );
        let mut batch_len = 0usize;
        let mut element_len = 0usize;
        for arr in &arrs {
            ensure!(
                arr.ndim() == shape.ndim() && arr.shape()[1..] == shape.slice()[1..],
                InvalidDimensionSnafu {
                    message: "Non-batch dimensions must match".to_string()
                }
            );
            batch_len = batch_len.checked_add(arr.shape()[0]).ok_or_else(|| {
                TensorOperationError::InvalidDimension {
                    message: "Batch dimension overflow".into(),
                }
            })?;
            element_len = element_len
                .checked_add(arr.len())
                .filter(|&len| len <= isize::MAX as usize)
                .ok_or_else(|| TensorOperationError::InvalidDimension {
                    message: "Tensor size overflow".into(),
                })?;
        }
        shape[0] = batch_len;
        let mut arrs = arrs.into_iter();
        let first = arrs.next().unwrap();
        let first_len = first.len();
        let (mut first, offset) = first.into_raw_vec_and_offset();
        let offset = offset.unwrap_or(0);
        first.truncate(offset + first_len);
        let mut data = if offset == 0 && first.capacity() >= element_len {
            first
        } else {
            // A large realloc can copy the first batch before we append the
            // rest. Allocate the final buffer directly when reuse is impossible.
            let mut data = Vec::with_capacity(element_len);
            if offset == 0 {
                data.append(&mut first);
            } else {
                data.extend(first.drain(offset..));
            }
            data
        };
        for arr in arrs {
            append_logical(&mut data, arr);
        }
        return Array::from_shape_vec(shape, data).map_err(|error| {
            TensorOperationError::InvalidDimension {
                message: error.to_string(),
            }
        });
    }
    let views = arrs.iter().map(Array::view).collect::<Vec<_>>();
    ndarray::concatenate(Axis(0), &views).map_err(|error| TensorOperationError::InvalidDimension {
        message: error.to_string(),
    })
}

/// Move only the C-contiguous logical region, avoiding a prefix-removal copy.
fn append_logical<A, D: Dimension>(target: &mut Vec<A>, arr: Array<A, D>) {
    debug_assert!(arr.is_standard_layout());
    let len = arr.len();
    let (mut data, offset) = arr.into_raw_vec_and_offset();
    let offset = offset.unwrap_or(0);
    data.truncate(offset + len);
    if offset == 0 {
        target.append(&mut data);
    } else {
        target.extend(data.drain(offset..));
    }
}

/// Split logical rows, including sliced and non-standard layouts.
pub(crate) fn split_batch<A: Clone, D: Dimension>(
    arr: Array<A, D>,
    indexes: Vec<usize>,
) -> Result<Vec<Array<A, D>>> {
    let Some(&axis_len) = arr.shape().first() else {
        return Err(TensorOperationError::InvalidDimension {
            message: "A scalar has no batch axis".into(),
        });
    };
    ensure!(
        indexes.is_sorted() && indexes.iter().all(|&index| index <= axis_len),
        InvalidIndexSnafu {
            message: "Split indices must be sorted and within the batch axis".to_string()
        }
    );
    if indexes.is_empty() {
        return Ok(vec![arr]);
    }
    let first_end = indexes[0];
    let mut start = first_end;
    let mut parts = Vec::with_capacity(indexes.len() + 1);
    for end in indexes.into_iter().skip(1).chain(std::iter::once(axis_len)) {
        parts.push(arr.slice_axis(Axis(0), Slice::from(start..end)).to_owned());
        start = end;
    }
    let mut first = arr;
    first.slice_axis_inplace(Axis(0), Slice::from(..first_end));
    parts.insert(0, first);
    Ok(parts)
}
