//! 为ONNX中Tensor至本crate定义Tensor转换提供方法

use std::{fmt::Debug, sync::Arc};

use ndarray::ArrayD;

use crate::tensor::{
    ConcreteArrayHandler, DynTensor, IntoTensorElementType, Tensor, TensorElementType,
};

impl From<&ort::value::ValueType> for TensorElementType {
    /// 从`&ort::value::ValueType`类型转换为当前类型。
    ///
    /// 目前只支持`&ort::value::ValueType::Tensor`类型转换，其他类型尚未实现。
    /// 转换过程中忽略`dimensions`信息，只关注`ty`。
    fn from(value: &ort::value::ValueType) -> Self {
        match value {
            // 当输入类型为Tensor时，提取其内部类型并进行转换。
            ort::value::ValueType::Tensor {
                ty,
                shape: _shape,
                dimension_symbols: _,
            } => ty.into(),
            // 其他类型尚未实现转换逻辑，抛出未实现的错误。
            _ => unimplemented!(),
        }
    }
}

impl From<&ort::tensor::TensorElementType> for TensorElementType {
    /// 将`ort::tensor::TensorElementType`转换为`TensorElementType`。
    ///
    /// 这个函数提供了一种从`&ort::tensor::TensorElementType`枚举值到`TensorElementType`枚举值的转换方法。
    /// 它处理了大多数的`&ort::tensor::TensorElementType`变体，并将它们映射到相应的`TensorElementType`变体。
    /// 对于某些类型，如字符串和半精度浮点数，转换可能会受到特性标志的控制或者未实现。
    ///
    /// 参数:
    /// - `value`: `&ort::tensor::TensorElementType`的一个实例，表示要转换的张量元素类型。
    ///
    /// 返回值:
    /// - 返回一个`TensorElementType`的实例，与输入`&ort::tensor::TensorElementType`相对应。
    fn from(value: &ort::tensor::TensorElementType) -> Self {
        match value {
            ort::tensor::TensorElementType::Float32 => TensorElementType::Float,
            ort::tensor::TensorElementType::Uint8 => TensorElementType::UInt8,
            ort::tensor::TensorElementType::Int8 => TensorElementType::Int8,
            ort::tensor::TensorElementType::Uint16 => TensorElementType::UInt16,
            ort::tensor::TensorElementType::Int16 => TensorElementType::Int16,
            ort::tensor::TensorElementType::Int32 => TensorElementType::Int,
            ort::tensor::TensorElementType::Int64 => TensorElementType::Int64,
            ort::tensor::TensorElementType::String
            | ort::tensor::TensorElementType::Complex64
            | ort::tensor::TensorElementType::Complex128
            | ort::tensor::TensorElementType::Float8E4M3FN
            | ort::tensor::TensorElementType::Float8E4M3FNUZ
            | ort::tensor::TensorElementType::Float8E5M2
            | ort::tensor::TensorElementType::Float8E5M2FNUZ
            | ort::tensor::TensorElementType::Uint4
            | ort::tensor::TensorElementType::Int4
            | ort::tensor::TensorElementType::Undefined => {
                unimplemented!()
            }
            ort::tensor::TensorElementType::Bool => TensorElementType::Bool,
            ort::tensor::TensorElementType::Float64 => TensorElementType::Double,
            ort::tensor::TensorElementType::Uint32 => TensorElementType::UInt,
            ort::tensor::TensorElementType::Uint64 => TensorElementType::UInt64,
            #[cfg(feature = "half")]
            ort::tensor::TensorElementType::Float16 => TensorElementType::Half,
            #[cfg(feature = "half")]
            ort::tensor::TensorElementType::Bfloat16 => TensorElementType::BFloat16,
        }
    }
}

impl From<TensorElementType> for ort::tensor::TensorElementType {
    /// 将自定义的张量元素类型转换为ONNX运行时的张量元素类型。
    ///
    /// 该函数提供了一种机制，通过使用匹配语句将自定义的张量元素类型（`TensorElementType`）
    /// 转换为ONNX运行时对应的元素类型（`ort::TensorElementType`）。
    /// 这种转换是必要的，因为不同框架或库可能定义了不同版本的类型枚举，
    /// 并且在跨库操作或互操作性场景中需要进行类型转换。
    ///
    /// 参数:
    /// - `value`: 自定义的张量元素类型枚举值。
    ///
    /// 返回值:
    /// - 返回对应的`ort::TensorElementType`枚举值。
    ///
    /// 注意:
    /// - 该函数支持多种基础类型和布尔类型的转换。
    /// - 如果启用了`"half"`特性，则还支持`Half`和`BFloat16`类型的转换。
    /// - 未明确处理的`TensorElementType`类型将导致编译时错误，提示该功能未实现。
    fn from(value: TensorElementType) -> Self {
        match value {
            TensorElementType::UInt8 => ort::tensor::TensorElementType::Uint8,
            TensorElementType::Int8 => ort::tensor::TensorElementType::Int8,
            TensorElementType::Int16 => ort::tensor::TensorElementType::Int16,
            TensorElementType::Int => ort::tensor::TensorElementType::Int32,
            TensorElementType::Int64 => ort::tensor::TensorElementType::Int64,
            TensorElementType::Float => ort::tensor::TensorElementType::Float32,
            TensorElementType::Double => ort::tensor::TensorElementType::Float64,
            TensorElementType::Bool => ort::tensor::TensorElementType::Bool,
            #[cfg(feature = "half")]
            TensorElementType::Half => ort::tensor::TensorElementType::Float16,
            #[cfg(feature = "half")]
            TensorElementType::BFloat16 => ort::tensor::TensorElementType::Bfloat16,
            TensorElementType::UInt16 => ort::tensor::TensorElementType::Uint16,
            TensorElementType::UInt => ort::tensor::TensorElementType::Uint32,
            TensorElementType::UInt64 => ort::tensor::TensorElementType::Uint64,
            #[cfg(feature = "torch")]
            TensorElementType::ComplexHalf => {
                unreachable!()
            }
            #[cfg(feature = "torch")]
            TensorElementType::ComplexFloat => {
                unreachable!()
            }
            #[cfg(feature = "torch")]
            TensorElementType::ComplexDouble => {
                unreachable!()
            }
            #[cfg(feature = "torch")]
            TensorElementType::QInt8 => {
                unreachable!()
            }
            #[cfg(feature = "torch")]
            TensorElementType::QUInt8 => {
                unreachable!()
            }
            #[cfg(feature = "torch")]
            TensorElementType::QInt32 => {
                unreachable!()
            }
        }
    }
}

impl<T: ort::tensor::PrimitiveTensorElementType + IntoTensorElementType + Clone>
    From<ort::value::DynTensor> for Tensor<T>
{
    fn from(value: ort::value::DynTensor) -> Self {
        let (dimension, data) = value
            .try_extract_tensor::<T>()
            .expect("Failed to extract raw tensor");

        Tensor {
            dtype: value.dtype().into(),
            dimensions: dimension.to_vec(),
            data: ArrayD::<T>::from_shape_vec(
                dimension.iter().map(|&d| d as usize).collect::<Vec<_>>(),
                Vec::from(data),
            )
            .expect("Failed to create ArrayD from Vec"),
        }
    }
}

impl<T: ort::tensor::PrimitiveTensorElementType + IntoTensorElementType + Debug + Clone>
    From<Tensor<T>> for ort::value::Tensor<T>
{
    fn from(value: Tensor<T>) -> Self {
        ort::value::Tensor::from_array((
            value.dimensions,
            value.data.iter().cloned().collect::<Vec<_>>(),
        ))
        .expect("Failed to convert Tensor to ort::Tensor")
    }
}

impl<T: ort::tensor::PrimitiveTensorElementType + IntoTensorElementType + Debug + Clone>
    From<Tensor<T>> for ort::value::DynTensor
{
    fn from(value: Tensor<T>) -> Self {
        ort::value::Tensor::from_array((
            value.dimensions,
            value.data.iter().cloned().collect::<Vec<_>>(),
        ))
        .map(|tensor| tensor.upcast())
        .expect("Failed to convert Tensor to ort::DynTensor")
    }
}

impl From<DynTensor> for ort::value::DynTensor {
    fn from(value: DynTensor) -> Self {
        macro_rules! downcast_and_convert {
            ($tensor_type:ty) => {
                ort::value::Tensor::from_array((
                    value.dimensions,
                    value
                        .data
                        .downcast::<ArrayD<$tensor_type>>()
                        .expect(concat!(
                            "Failed to downcast to ArrayD<",
                            stringify!($tensor_type),
                            ">"
                        ))
                        .iter()
                        .cloned()
                        .collect::<Vec<_>>(),
                ))
                .expect("Cannot create target ort::Tensor")
                .upcast()
            };
        }
        match value.dtype {
            TensorElementType::Int8 => downcast_and_convert!(i8),
            TensorElementType::Int16 => downcast_and_convert!(i16),
            TensorElementType::Int => downcast_and_convert!(i32),
            TensorElementType::Int64 => downcast_and_convert!(i64),
            TensorElementType::Float => downcast_and_convert!(f32),
            TensorElementType::Double => downcast_and_convert!(f64),
            TensorElementType::Bool => downcast_and_convert!(bool),
            TensorElementType::UInt8 => downcast_and_convert!(u8),
            TensorElementType::UInt16 => downcast_and_convert!(u16),
            TensorElementType::UInt => downcast_and_convert!(u32),
            TensorElementType::UInt64 => downcast_and_convert!(u64),
            #[cfg(feature = "torch")]
            TensorElementType::ComplexHalf
            | TensorElementType::ComplexFloat
            | TensorElementType::ComplexDouble
            | TensorElementType::QInt8
            | TensorElementType::QUInt8
            | TensorElementType::QInt32 => {
                unreachable!()
            }
            #[cfg(any(feature = "half", feature = "torch"))]
            TensorElementType::Half | TensorElementType::BFloat16 => {
                unimplemented!()
            }
        }
    }
}

impl From<ort::value::DynTensor> for DynTensor {
    fn from(value: ort::value::DynTensor) -> Self {
        match value.dtype() {
            ort::value::ValueType::Tensor { ty, .. } => {
                macro_rules! match_ty {
                    ($type_:ty) => {{
                        let (dimension, data) = value
                            .try_extract_tensor::<$type_>()
                            .expect("Failed to extract raw tensor");

                        DynTensor {
                            dtype: value.dtype().into(),
                            dimensions: dimension.to_vec(),
                            data: Box::new(
                                ArrayD::<$type_>::from_shape_vec(
                                    dimension.iter().map(|&d| d as usize).collect::<Vec<_>>(),
                                    Vec::from(data),
                                )
                                .expect("Failed to create ArrayD from Vec"),
                            ),
                            handler: Arc::new(ConcreteArrayHandler::<$type_>::default()),
                        }
                    }};
                }
                match ty {
                    ort::tensor::TensorElementType::Float32 => match_ty!(f32),
                    ort::tensor::TensorElementType::Float64 => match_ty!(f64),
                    ort::tensor::TensorElementType::Int8 => match_ty!(i8),
                    ort::tensor::TensorElementType::Int16 => match_ty!(i16),
                    ort::tensor::TensorElementType::Int32 => match_ty!(i32),
                    ort::tensor::TensorElementType::Int64 => match_ty!(i64),
                    ort::tensor::TensorElementType::Uint8 => match_ty!(u8),
                    ort::tensor::TensorElementType::Uint16 => match_ty!(u16),
                    ort::tensor::TensorElementType::Uint32 => match_ty!(u32),
                    ort::tensor::TensorElementType::Uint64 => match_ty!(u64),
                    ort::tensor::TensorElementType::Bool => match_ty!(bool),
                    #[cfg(feature = "half")]
                    ort::tensor::TensorElementType::Float16
                    | ort::tensor::TensorElementType::Bfloat16 => {
                        unimplemented!()
                    }
                    ort::tensor::TensorElementType::String
                    | ort::tensor::TensorElementType::Complex64
                    | ort::tensor::TensorElementType::Complex128
                    | ort::tensor::TensorElementType::Float8E4M3FN
                    | ort::tensor::TensorElementType::Float8E4M3FNUZ
                    | ort::tensor::TensorElementType::Float8E5M2
                    | ort::tensor::TensorElementType::Float8E5M2FNUZ
                    | ort::tensor::TensorElementType::Uint4
                    | ort::tensor::TensorElementType::Int4
                    | ort::tensor::TensorElementType::Undefined => {
                        unimplemented!()
                    }
                }
            }
            _ => {
                unimplemented!("Unsupported dtype")
            }
        }
    }
}
