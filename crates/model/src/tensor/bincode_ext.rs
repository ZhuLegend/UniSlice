//! 为[DynTensor]提供bincode支持

use std::sync::Arc;

#[cfg(feature = "bincode")]
use bincode::{
    Decode, Encode,
    de::Decoder,
    enc::Encoder,
    error::{DecodeError, EncodeError},
    serde::Compat,
};
use ndarray::ArrayD;
#[cfg(all(feature = "bincode", feature = "serde"))]
use serde::{de::DeserializeOwned, ser::Serialize};

use crate::tensor::{
    ConcreteArrayHandler, DynTensor, EraserData, IntoTensorElementType, Tensor, TensorElementType,
    TensorHandler,
};

#[cfg(all(feature = "bincode", feature = "serde"))]
impl<T: Serialize + IntoTensorElementType> Encode for Tensor<T> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        self.dtype.encode(encoder)?;
        self.dimensions.encode(encoder)?;
        Compat(&self.data).encode(encoder)?;
        Ok(())
    }
}

#[cfg(all(feature = "bincode", feature = "serde"))]
impl<T: DeserializeOwned + IntoTensorElementType, Context> Decode<Context> for Tensor<T> {
    fn decode<D: Decoder>(decoder: &mut D) -> Result<Self, DecodeError> {
        let dtype = TensorElementType::decode(decoder)?;
        let dimensions = Vec::<i64>::decode(decoder)?;
        let data: Compat<ArrayD<T>> = Decode::decode(decoder)?;
        if dtype != T::into_tensor_element_type() {
            return Err(DecodeError::Other(
                "Tensor dtype does not match its element type",
            ));
        }
        super::validate_array_dimensions(&dimensions, data.0.shape())
            .map_err(|e| DecodeError::OtherString(e.to_string()))?;
        Ok(Tensor {
            dtype,
            dimensions,
            data: data.0,
        })
    }
}

#[cfg(feature = "bincode")]
impl Encode for DynTensor {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        self.dtype.encode(encoder)?;
        self.dimensions.encode(encoder)?;
        macro_rules! encode_dyn_tensor {
            ($type_:ty) => {{
                let data = self.data.downcast_ref::<ArrayD<$type_>>().unwrap();
                Compat(data).encode(encoder)
            }};
        }

        match self.dtype {
            TensorElementType::UInt8 => encode_dyn_tensor!(u8),
            TensorElementType::Int8 => encode_dyn_tensor!(i8),
            TensorElementType::Int16 => encode_dyn_tensor!(i16),
            TensorElementType::Int => encode_dyn_tensor!(i32),
            TensorElementType::Int64 => encode_dyn_tensor!(i64),
            TensorElementType::Float => encode_dyn_tensor!(f32),
            TensorElementType::Double => encode_dyn_tensor!(f64),
            TensorElementType::Bool => encode_dyn_tensor!(bool),
            #[cfg(any(feature = "half", feature = "torch"))]
            TensorElementType::Half => encode_dyn_tensor!(half::f16),
            #[cfg(any(feature = "half", feature = "torch"))]
            TensorElementType::BFloat16 => encode_dyn_tensor!(half::bf16),
            #[cfg(feature = "onnx")]
            TensorElementType::UInt16 => encode_dyn_tensor!(u16),
            #[cfg(feature = "onnx")]
            TensorElementType::UInt => encode_dyn_tensor!(u32),
            #[cfg(feature = "onnx")]
            TensorElementType::UInt64 => encode_dyn_tensor!(u64),
            #[cfg(feature = "torch")]
            TensorElementType::ComplexHalf
            | TensorElementType::ComplexFloat
            | TensorElementType::ComplexDouble
            | TensorElementType::QInt8
            | TensorElementType::QUInt8
            | TensorElementType::QInt32 => {
                Err(EncodeError::Other("Unsupported tensor element type"))
            }
        }
    }
}

#[cfg(feature = "bincode")]
impl<Context> Decode<Context> for DynTensor {
    fn decode<D: Decoder>(decoder: &mut D) -> Result<Self, DecodeError> {
        let dtype: TensorElementType = Decode::decode(decoder)?;
        let dimensions: Vec<i64> = Decode::decode(decoder)?;
        macro_rules! decode_and_box {
            ($($type_:ty)*) => {{
                let data: Compat<ArrayD<$($type_)*>> = Decode::decode(decoder)?;
                super::validate_array_dimensions(&dimensions, data.0.shape())
                    .map_err(|error| DecodeError::OtherString(error.to_string()))?;
                (Box::new(data.0), Arc::new(ConcreteArrayHandler::<$($type_)*>::default()))
            }};
        }
        let (data, handler): (Box<EraserData>, Arc<dyn TensorHandler>) = match dtype {
            TensorElementType::UInt8 => decode_and_box!(u8),
            TensorElementType::Int8 => decode_and_box!(i8),
            TensorElementType::Int16 => decode_and_box!(i16),
            TensorElementType::Int => decode_and_box!(i32),
            TensorElementType::Int64 => decode_and_box!(i64),
            TensorElementType::Float => decode_and_box!(f32),
            TensorElementType::Double => decode_and_box!(f64),
            TensorElementType::Bool => decode_and_box!(bool),
            #[cfg(any(feature = "half", feature = "torch"))]
            TensorElementType::Half => decode_and_box!(half::f16),
            #[cfg(any(feature = "half", feature = "torch"))]
            TensorElementType::BFloat16 => decode_and_box!(half::bf16),
            #[cfg(feature = "onnx")]
            TensorElementType::UInt16 => decode_and_box!(u16),
            #[cfg(feature = "onnx")]
            TensorElementType::UInt => decode_and_box!(u32),
            #[cfg(feature = "onnx")]
            TensorElementType::UInt64 => decode_and_box!(u64),
            #[cfg(feature = "torch")]
            TensorElementType::ComplexHalf
            | TensorElementType::ComplexFloat
            | TensorElementType::ComplexDouble
            | TensorElementType::QInt8
            | TensorElementType::QUInt8
            | TensorElementType::QInt32 => {
                return Err(DecodeError::Other("Unsupported tensor element type"));
            }
        };

        Ok(DynTensor {
            dtype,
            dimensions,
            data,
            handler,
        })
    }
}

#[cfg(feature = "bincode")]
bincode::impl_borrow_decode!(DynTensor);

// Backend features must not shift the dtype tags on the wire.
impl Encode for TensorElementType {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        (self.clone() as u32).encode(encoder)
    }
}
impl<Context> Decode<Context> for TensorElementType {
    fn decode<D: Decoder>(decoder: &mut D) -> Result<Self, DecodeError> {
        Ok(match u32::decode(decoder)? {
            0 => Self::UInt8,
            1 => Self::Int8,
            2 => Self::Int16,
            3 => Self::Int,
            4 => Self::Int64,
            5 => Self::Float,
            6 => Self::Double,
            7 => Self::Bool,
            #[cfg(any(feature = "half", feature = "torch"))]
            8 => Self::Half,
            #[cfg(any(feature = "half", feature = "torch"))]
            9 => Self::BFloat16,
            #[cfg(feature = "onnx")]
            10 => Self::UInt16,
            #[cfg(feature = "onnx")]
            11 => Self::UInt,
            #[cfg(feature = "onnx")]
            12 => Self::UInt64,
            #[cfg(feature = "torch")]
            13 => Self::ComplexHalf,
            #[cfg(feature = "torch")]
            14 => Self::ComplexFloat,
            #[cfg(feature = "torch")]
            15 => Self::ComplexDouble,
            #[cfg(feature = "torch")]
            16 => Self::QInt8,
            #[cfg(feature = "torch")]
            17 => Self::QUInt8,
            #[cfg(feature = "torch")]
            18 => Self::QInt32,
            _ => return Err(DecodeError::Other("Unsupported tensor dtype tag")),
        })
    }
}
bincode::impl_borrow_decode!(TensorElementType);
