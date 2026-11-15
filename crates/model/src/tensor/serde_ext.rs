//! 为[DynTensor]提供serde支持

use std::sync::Arc;

use ndarray::ArrayD;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::tensor::{ConcreteArrayHandler, DynTensor, TensorElementType};

#[derive(Serialize)]
enum SerDynTensor<'a> {
    UInt8(Vec<i64>, &'a ArrayD<u8>),
    Int8(Vec<i64>, &'a ArrayD<i8>),
    Int16(Vec<i64>, &'a ArrayD<i16>),
    Int(Vec<i64>, &'a ArrayD<i32>),
    Int64(Vec<i64>, &'a ArrayD<i64>),
    Float(Vec<i64>, &'a ArrayD<f32>),
    Double(Vec<i64>, &'a ArrayD<f64>),
    Bool(Vec<i64>, &'a ArrayD<bool>),
    #[cfg(any(feature = "half", feature = "torch"))]
    Half(Vec<i64>, &'a ArrayD<half::f16>),
    #[cfg(any(feature = "half", feature = "torch"))]
    BFloat16(Vec<i64>, &'a ArrayD<half::bf16>),
    #[cfg(feature = "onnx")]
    UInt16(Vec<i64>, &'a ArrayD<u16>),
    #[cfg(feature = "onnx")]
    UInt(Vec<i64>, &'a ArrayD<u32>),
    #[cfg(feature = "onnx")]
    UInt64(Vec<i64>, &'a ArrayD<u64>),
}

impl<'a> From<&'a DynTensor> for SerDynTensor<'a> {
    fn from(dyn_tensor: &'a DynTensor) -> Self {
        macro_rules! match_tensor {
            ($type_:ty, $variant:ident) => {{
                let data: &ArrayD<$type_> = dyn_tensor.data.downcast_ref().unwrap();
                SerDynTensor::$variant(dyn_tensor.dimensions.clone(), data)
            }};
        }
        match dyn_tensor.dtype {
            TensorElementType::UInt8 => match_tensor!(u8, UInt8),
            TensorElementType::Int8 => match_tensor!(i8, Int8),
            TensorElementType::Int16 => match_tensor!(i16, Int16),
            TensorElementType::Int => match_tensor!(i32, Int),
            TensorElementType::Int64 => match_tensor!(i64, Int64),
            TensorElementType::Float => match_tensor!(f32, Float),
            TensorElementType::Double => match_tensor!(f64, Double),
            TensorElementType::Bool => match_tensor!(bool, Bool),
            #[cfg(any(feature = "half", feature = "torch"))]
            TensorElementType::Half => match_tensor!(half::f16, Half),
            #[cfg(any(feature = "half", feature = "torch"))]
            TensorElementType::BFloat16 => match_tensor!(half::bf16, BFloat16),
            #[cfg(feature = "onnx")]
            TensorElementType::UInt16 => match_tensor!(u16, UInt16),
            #[cfg(feature = "onnx")]
            TensorElementType::UInt => match_tensor!(u32, UInt),
            #[cfg(feature = "onnx")]
            TensorElementType::UInt64 => match_tensor!(u64, UInt64),
            #[cfg(feature = "torch")]
            TensorElementType::ComplexHalf
            | TensorElementType::ComplexFloat
            | TensorElementType::ComplexDouble
            | TensorElementType::QInt8
            | TensorElementType::QUInt8
            | TensorElementType::QInt32 => {
                unimplemented!()
            }
        }
    }
}

#[derive(Deserialize)]
enum DeDynTensor {
    UInt8(Vec<i64>, ArrayD<u8>),
    Int8(Vec<i64>, ArrayD<i8>),
    Int16(Vec<i64>, ArrayD<i16>),
    Int(Vec<i64>, ArrayD<i32>),
    Int64(Vec<i64>, ArrayD<i64>),
    Float(Vec<i64>, ArrayD<f32>),
    Double(Vec<i64>, ArrayD<f64>),
    Bool(Vec<i64>, ArrayD<bool>),
    #[cfg(any(feature = "half", feature = "torch"))]
    Half(Vec<i64>, ArrayD<half::f16>),
    #[cfg(any(feature = "half", feature = "torch"))]
    BFloat16(Vec<i64>, ArrayD<half::bf16>),
    #[cfg(feature = "onnx")]
    UInt16(Vec<i64>, ArrayD<u16>),
    #[cfg(feature = "onnx")]
    UInt(Vec<i64>, ArrayD<u32>),
    #[cfg(feature = "onnx")]
    UInt64(Vec<i64>, ArrayD<u64>),
}

impl TryFrom<DeDynTensor> for DynTensor {
    type Error = super::TensorOperationError;

    fn try_from(de_dyn_tensor: DeDynTensor) -> Result<Self, Self::Error> {
        macro_rules! create_dyn_tensor {
            ($variant:ident, $dimensions:expr, $data:expr, $($ty:ty)*) => {{
                super::validate_array_dimensions(&$dimensions, $data.shape())?;
                Ok(DynTensor {
                    dtype: TensorElementType::$variant,
                    dimensions: $dimensions,
                    data: Box::new($data),
                    handler: Arc::new(ConcreteArrayHandler::<$($ty)*>::default()),
                })
            }};
        }
        match de_dyn_tensor {
            DeDynTensor::UInt8(dimensions, data) => create_dyn_tensor!(UInt8, dimensions, data, u8),
            DeDynTensor::Int8(dimensions, data) => create_dyn_tensor!(Int8, dimensions, data, i8),
            DeDynTensor::Int16(dimensions, data) => {
                create_dyn_tensor!(Int16, dimensions, data, i16)
            }
            DeDynTensor::Int(dimensions, data) => create_dyn_tensor!(Int, dimensions, data, i32),
            DeDynTensor::Int64(dimensions, data) => {
                create_dyn_tensor!(Int64, dimensions, data, i64)
            }
            DeDynTensor::Float(dimensions, data) => {
                create_dyn_tensor!(Float, dimensions, data, f32)
            }
            DeDynTensor::Double(dimensions, data) => {
                create_dyn_tensor!(Double, dimensions, data, f64)
            }
            DeDynTensor::Bool(dimensions, data) => create_dyn_tensor!(Bool, dimensions, data, bool),
            #[cfg(any(feature = "half", feature = "torch"))]
            DeDynTensor::Half(dimensions, data) => {
                create_dyn_tensor!(Half, dimensions, data, half::f16)
            }
            #[cfg(any(feature = "half", feature = "torch"))]
            DeDynTensor::BFloat16(dimensions, data) => {
                create_dyn_tensor!(BFloat16, dimensions, data, half::bf16)
            }
            #[cfg(feature = "onnx")]
            DeDynTensor::UInt16(dimensions, data) => {
                create_dyn_tensor!(UInt16, dimensions, data, u16)
            }
            #[cfg(feature = "onnx")]
            DeDynTensor::UInt(dimensions, data) => create_dyn_tensor!(UInt, dimensions, data, u32),
            #[cfg(feature = "onnx")]
            DeDynTensor::UInt64(dimensions, data) => {
                create_dyn_tensor!(UInt64, dimensions, data, u64)
            }
        }
    }
}

impl Serialize for DynTensor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let serde_dyn_tensor: SerDynTensor = self.into();
        serde_dyn_tensor.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for DynTensor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let de_dyn_tensor: DeDynTensor = Deserialize::deserialize(deserializer)?;
        de_dyn_tensor.try_into().map_err(serde::de::Error::custom)
    }
}

// Keep the existing representation, validating both metadata and the typed payload.
impl<'de, T: serde::Deserialize<'de> + super::IntoTensorElementType> serde::Deserialize<'de>
    for super::Tensor<T>
{
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        struct Fields<T> {
            dtype: super::TensorElementType,
            dimensions: Vec<i64>,
            data: ndarray::ArrayD<T>,
        }
        let fields = Fields::<T>::deserialize(deserializer)?;
        if fields.dtype != T::into_tensor_element_type() {
            return Err(serde::de::Error::custom(
                "Tensor dtype does not match its element type",
            ));
        }
        super::validate_array_dimensions(&fields.dimensions, fields.data.shape())
            .map_err(serde::de::Error::custom)?;
        Ok(Self {
            dtype: fields.dtype,
            dimensions: fields.dimensions,
            data: fields.data,
        })
    }
}
