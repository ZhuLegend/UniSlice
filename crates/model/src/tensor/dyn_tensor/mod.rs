//! [DynTensor]

use std::{
    any::Any,
    fmt::{Debug, Formatter},
    sync::Arc,
};

use ndarray::{Array, ArrayD, Dimension};
pub(crate) use tensor_handler::*;

use super::{IntoTensorElementType, Tensor, TensorElementType};

mod tensor_handler;

pub(crate) type EraserData = dyn Any + Send + Sync;

/// 类型擦除的`Tensor`
pub struct DynTensor {
    /// 数据类型
    pub(crate) dtype: TensorElementType,
    /// `Tensor`维度
    pub(crate) dimensions: Vec<i64>,
    /// 实际`Tensor`数据
    pub(crate) data: Box<EraserData>,
    /// 数据处理器
    pub(crate) handler: Arc<dyn TensorHandler>,
}

#[cfg(any(feature = "serde", feature = "bincode"))]
pub(crate) fn validate_array_dimensions(dimensions: &[i64], shape: &[usize]) -> Result<()> {
    if dimensions.len() != shape.len()
        || dimensions
            .iter()
            .zip(shape)
            .any(|(&dimension, &actual)| usize::try_from(dimension) != Ok(actual))
    {
        return Err(TensorOperationError::InvalidDimension {
            message: format!(
                "Tensor dimensions {dimensions:?} disagree with array shape {shape:?}"
            ),
        });
    }
    Ok(())
}

impl Debug for DynTensor {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynTensor")
            .field("dtype", &self.dtype)
            .field("dimensions", &self.dimensions)
            .finish()
    }
}

impl DynTensor {
    /// 获取数据类型
    pub fn dtype(&self) -> TensorElementType {
        self.dtype.clone()
    }

    /// 获取`Tensor`维度
    pub fn dimensions(&self) -> &[i64] {
        &self.dimensions
    }

    /// 用1填充指定形状和类型的`Tensor`
    pub fn ones(dimensions: Vec<usize>, dtype: TensorElementType) -> Self {
        match dtype {
            TensorElementType::UInt8 => Tensor::<u8>::ones(dimensions).into_dyn(),
            TensorElementType::Int8 => Tensor::<i8>::ones(dimensions).into_dyn(),
            TensorElementType::Int16 => Tensor::<i16>::ones(dimensions).into_dyn(),
            TensorElementType::Int => Tensor::<i32>::ones(dimensions).into_dyn(),
            TensorElementType::Int64 => Tensor::<i64>::ones(dimensions).into_dyn(),
            TensorElementType::Float => Tensor::<f32>::ones(dimensions).into_dyn(),
            TensorElementType::Double => Tensor::<f64>::ones(dimensions).into_dyn(),
            #[cfg(feature = "onnx")]
            TensorElementType::UInt16 => Tensor::<u16>::ones(dimensions).into_dyn(),
            #[cfg(feature = "onnx")]
            TensorElementType::UInt => Tensor::<u32>::ones(dimensions).into_dyn(),
            #[cfg(feature = "onnx")]
            TensorElementType::UInt64 => Tensor::<u64>::ones(dimensions).into_dyn(),
            #[cfg(feature = "torch")]
            TensorElementType::ComplexHalf
            | TensorElementType::ComplexFloat
            | TensorElementType::ComplexDouble
            | TensorElementType::QInt8
            | TensorElementType::QUInt8
            | TensorElementType::QInt32 => {
                unimplemented!()
            }
            #[cfg(any(feature = "half", feature = "torch"))]
            TensorElementType::Half | TensorElementType::BFloat16 => unimplemented!(),
            TensorElementType::Bool => Tensor::from(ndarray::ArrayD::from_elem(
                ndarray::IxDyn(&dimensions),
                true,
            ))
            .into_dyn(),
        }
    }

    /// 用0填充指定形状和类型的`Tensor`
    pub fn zeros(dimensions: Vec<usize>, dtype: TensorElementType) -> Self {
        match dtype {
            TensorElementType::UInt8 => Tensor::<u8>::zeros(dimensions).into_dyn(),
            TensorElementType::Int8 => Tensor::<i8>::zeros(dimensions).into_dyn(),
            TensorElementType::Int16 => Tensor::<i16>::zeros(dimensions).into_dyn(),
            TensorElementType::Int => Tensor::<i32>::zeros(dimensions).into_dyn(),
            TensorElementType::Int64 => Tensor::<i64>::zeros(dimensions).into_dyn(),
            TensorElementType::Float => Tensor::<f32>::zeros(dimensions).into_dyn(),
            TensorElementType::Double => Tensor::<f64>::zeros(dimensions).into_dyn(),
            #[cfg(feature = "onnx")]
            TensorElementType::UInt16 => Tensor::<u16>::zeros(dimensions).into_dyn(),
            #[cfg(feature = "onnx")]
            TensorElementType::UInt => Tensor::<u32>::zeros(dimensions).into_dyn(),
            #[cfg(feature = "onnx")]
            TensorElementType::UInt64 => Tensor::<u64>::zeros(dimensions).into_dyn(),
            #[cfg(feature = "torch")]
            TensorElementType::ComplexHalf
            | TensorElementType::ComplexFloat
            | TensorElementType::ComplexDouble
            | TensorElementType::QInt8
            | TensorElementType::QUInt8
            | TensorElementType::QInt32 => {
                unimplemented!()
            }
            #[cfg(any(feature = "half", feature = "torch"))]
            TensorElementType::Half | TensorElementType::BFloat16 => unimplemented!(),
            TensorElementType::Bool => Tensor::from(ndarray::ArrayD::from_elem(
                ndarray::IxDyn(&dimensions),
                false,
            ))
            .into_dyn(),
        }
    }

    /// 用随机值填充指定形状和类型的`Tensor`
    #[cfg(feature = "rand")]
    pub fn random(dimensions: Vec<usize>, dtype: TensorElementType) -> Self {
        match dtype {
            TensorElementType::UInt8 => Tensor::<u8>::random(dimensions).into_dyn(),
            TensorElementType::Int8 => Tensor::<i8>::random(dimensions).into_dyn(),
            TensorElementType::Int16 => Tensor::<i16>::random(dimensions).into_dyn(),
            TensorElementType::Int => Tensor::<i32>::random(dimensions).into_dyn(),
            TensorElementType::Int64 => Tensor::<i64>::random(dimensions).into_dyn(),
            TensorElementType::Float => Tensor::<f32>::random(dimensions).into_dyn(),
            TensorElementType::Double => Tensor::<f64>::random(dimensions).into_dyn(),
            #[cfg(feature = "onnx")]
            TensorElementType::UInt16 => Tensor::<u16>::random(dimensions).into_dyn(),
            #[cfg(feature = "onnx")]
            TensorElementType::UInt => Tensor::<u32>::random(dimensions).into_dyn(),
            #[cfg(feature = "onnx")]
            TensorElementType::UInt64 => Tensor::<u64>::random(dimensions).into_dyn(),
            #[cfg(feature = "torch")]
            TensorElementType::ComplexHalf
            | TensorElementType::ComplexFloat
            | TensorElementType::ComplexDouble
            | TensorElementType::QInt8
            | TensorElementType::QUInt8
            | TensorElementType::QInt32 => {
                unimplemented!()
            }
            #[cfg(any(feature = "half", feature = "torch"))]
            TensorElementType::Half | TensorElementType::BFloat16 => unimplemented!(),
            TensorElementType::Bool => Tensor::<bool>::random(dimensions).into_dyn(),
        }
    }
}

impl DynTensor {
    /// 判断[DynTensor]是否可以转换为[Tensor]
    pub fn can_downcast<T: IntoTensorElementType>(&self) -> bool {
        self.dtype == T::into_tensor_element_type()
    }

    /// 尝试将[DynTensor]转换为[Tensor]
    pub fn downcast<T: IntoTensorElementType>(self) -> Option<Tensor<T>> {
        if self.can_downcast::<T>() {
            let data = *self.data.downcast::<ArrayD<T>>().unwrap();
            Some(Tensor {
                dtype: self.dtype,
                dimensions: self.dimensions,
                data,
            })
        } else {
            None
        }
    }

    /// 将[DynTensor]沿第0轴拼接
    pub fn concat_batch(tensors: impl IntoIterator<Item = DynTensor>) -> Result<DynTensor> {
        let mut tensors = tensors.into_iter();
        let base = tensors.next().ok_or(TensorOperationError::TargetInvalid {
            message: "No tensor to concat".to_string(),
        })?;
        if base.dimensions.is_empty() {
            return Err(TensorOperationError::InvalidDimension {
                message: "A scalar has no batch axis".into(),
            });
        }
        let mut dimensions = base.dimensions.clone();
        let mut data = vec![base.data];
        for tensor in tensors {
            if tensor.dimensions.is_empty() || tensor.dimensions[1..] != dimensions[1..] {
                return Err(TensorOperationError::InvalidDimension {
                    message: "Non-batch dimensions must match".into(),
                });
            }
            dimensions[0] = dimensions[0]
                .checked_add(tensor.dimensions[0])
                .ok_or_else(|| TensorOperationError::InvalidDimension {
                    message: "Batch dimension overflow".into(),
                })?;
            data.push(tensor.data);
        }
        let data = base.handler.concat(data)?;

        Ok(DynTensor {
            dtype: base.dtype.clone(),
            dimensions,
            data,
            handler: base.handler.clone(),
        })
    }

    /// 将[DynTensor]沿第0轴分割
    pub fn split_batch(self, indexes: Vec<usize>) -> Result<Vec<DynTensor>, TensorOperationError> {
        let original_dim0 = self
            .dimensions
            .first()
            .copied()
            .and_then(|dimension| usize::try_from(dimension).ok())
            .ok_or_else(|| TensorOperationError::InvalidDimension {
                message: "Tensor has no valid batch axis".into(),
            })?;
        if !indexes.is_sorted() || indexes.iter().any(|&index| index > original_dim0) {
            return Err(TensorOperationError::InvalidIndex {
                message: format!("Index exceeds original dimension {original_dim0}"),
            });
        }

        let mut split_points = Vec::with_capacity(indexes.len() + 2);
        split_points.push(0);
        split_points.extend_from_slice(&indexes);
        split_points.push(original_dim0);

        let split_sizes = split_points
            .windows(2)
            .map(|w| w[1] - w[0])
            .collect::<Vec<_>>();

        // 调用 handler 的 split_batch 方法，并处理可能的错误
        let data_result = self.handler.split_batch(self.data, indexes)?;

        // 构建结果张量
        let result_tensors = data_result
            .into_iter()
            .zip(split_sizes)
            .map(|(part, size)| {
                let mut new_dimensions = self.dimensions.clone();
                new_dimensions[0] = size as i64; // 仅修改第0维
                DynTensor {
                    dtype: self.dtype.clone(),
                    dimensions: new_dimensions,
                    data: part,
                    handler: self.handler.clone(),
                }
            })
            .collect();

        Ok(result_tensors)
    }
}

impl<T: IntoTensorElementType, D: Dimension + 'static> From<Array<T, D>> for DynTensor {
    fn from(value: Array<T, D>) -> Self {
        DynTensor {
            dtype: T::into_tensor_element_type(),
            dimensions: value.shape().iter().map(|&x| x as i64).collect(),
            data: Box::new(value.into_dyn()),
            handler: Arc::new(ConcreteArrayHandler::<T>::default()),
        }
    }
}

impl<T: IntoTensorElementType> Tensor<T> {
    /// 将[Tensor]转换为[DynTensor]
    #[inline]
    pub fn into_dyn(self) -> DynTensor {
        DynTensor {
            dtype: self.dtype,
            dimensions: self.dimensions,
            data: Box::new(self.data),
            handler: Arc::new(ConcreteArrayHandler::<T>::default()),
        }
    }
}

impl<T: IntoTensorElementType> From<Tensor<T>> for DynTensor {
    fn from(tensor: Tensor<T>) -> Self {
        tensor.into_dyn()
    }
}

impl<T: IntoTensorElementType> TryFrom<DynTensor> for Tensor<T> {
    type Error = &'static str;

    fn try_from(dyn_tensor: DynTensor) -> Result<Self, Self::Error> {
        if dyn_tensor.can_downcast::<T>() {
            let data = *dyn_tensor.data.downcast::<ArrayD<T>>().unwrap();
            Ok(Tensor {
                dtype: dyn_tensor.dtype,
                dimensions: dyn_tensor.dimensions,
                data,
            })
        } else {
            Err("Tensor element type mismatch")
        }
    }
}

impl Clone for DynTensor {
    fn clone(&self) -> Self {
        let data = TensorHandler::clone(self.handler.as_ref(), self.data.as_ref());
        DynTensor {
            dtype: self.dtype.clone(),
            dimensions: self.dimensions.clone(),
            data,
            handler: self.handler.clone(),
        }
    }
}
