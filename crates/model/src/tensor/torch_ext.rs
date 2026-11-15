//! 为Torch至Tensor转换提供便捷方法

use ndarray::ArrayD;

use crate::tensor::{Device, DynTensor, IntoTensorElementType, Tensor, TensorElementType};

impl From<tch::Kind> for TensorElementType {
    #[inline(always)]
    fn from(value: tch::Kind) -> Self {
        match value {
            tch::Kind::Uint8 => TensorElementType::UInt8,
            tch::Kind::Int8 => TensorElementType::Int8,
            tch::Kind::Int16 => TensorElementType::Int16,
            tch::Kind::Int => TensorElementType::Int,
            tch::Kind::Int64 => TensorElementType::Int64,
            tch::Kind::Half => TensorElementType::Half,
            tch::Kind::Float => TensorElementType::Float,
            tch::Kind::Double => TensorElementType::Double,
            tch::Kind::ComplexHalf => TensorElementType::ComplexHalf,
            tch::Kind::ComplexFloat => TensorElementType::ComplexFloat,
            tch::Kind::ComplexDouble => TensorElementType::ComplexDouble,
            tch::Kind::Bool => TensorElementType::Bool,
            tch::Kind::QInt8 => TensorElementType::QInt8,
            tch::Kind::QUInt8 => TensorElementType::QUInt8,
            tch::Kind::QInt32 => TensorElementType::QInt32,
            tch::Kind::BFloat16 => TensorElementType::BFloat16,
            tch::Kind::Float8e5m2
            | tch::Kind::Float8e4m3fn
            | tch::Kind::Float8e5m2fnuz
            | tch::Kind::Float8e4m3fnuz => {
                unimplemented!("This type of tensor element type is not support yet")
            }
        }
    }
}

impl From<TensorElementType> for tch::Kind {
    #[inline(always)]
    fn from(value: TensorElementType) -> Self {
        match value {
            TensorElementType::UInt8 => tch::Kind::Uint8,
            TensorElementType::Int8 => tch::Kind::Int8,
            TensorElementType::Int16 => tch::Kind::Int16,
            TensorElementType::Int => tch::Kind::Int,
            TensorElementType::Int64 => tch::Kind::Int64,
            TensorElementType::Float => tch::Kind::Float,
            TensorElementType::Double => tch::Kind::Double,
            TensorElementType::Bool => tch::Kind::Bool,
            TensorElementType::Half => tch::Kind::Half,
            TensorElementType::BFloat16 => tch::Kind::BFloat16,
            TensorElementType::ComplexHalf => tch::Kind::ComplexHalf,
            TensorElementType::ComplexFloat => tch::Kind::ComplexFloat,
            TensorElementType::ComplexDouble => tch::Kind::ComplexDouble,
            TensorElementType::QInt8 => tch::Kind::QInt8,
            TensorElementType::QUInt8 => tch::Kind::QUInt8,
            TensorElementType::QInt32 => tch::Kind::QInt32,
            #[cfg(feature = "onnx")]
            TensorElementType::UInt16 | TensorElementType::UInt | TensorElementType::UInt64 => {
                unreachable!()
            }
        }
    }
}

impl From<tch::Device> for Device {
    #[inline(always)]
    fn from(value: tch::Device) -> Self {
        match value {
            tch::Device::Cpu => Device::Cpu,
            tch::Device::Cuda(id) => Device::Gpu(id),
            _ => unimplemented!("This type of device is not support yet"),
        }
    }
}

impl From<Device> for tch::Device {
    #[inline(always)]
    fn from(value: Device) -> Self {
        match value {
            Device::Cpu => tch::Device::Cpu,
            Device::Gpu(id) => tch::Device::Cuda(id),
        }
    }
}
impl<T: tch::kind::Element + IntoTensorElementType + Default> From<tch::Tensor> for Tensor<T> {
    #[inline(always)]
    fn from(value: tch::Tensor) -> Self {
        let dtype = value.kind().into();
        let size = value.numel();

        let mut vec = vec![T::default(); size];
        value.copy_data(&mut vec, size);
        let array_d = ArrayD::from_shape_vec(
            value.size().iter().map(|&d| d as usize).collect::<Vec<_>>(),
            vec,
        )
        .unwrap();

        Tensor {
            dtype,
            dimensions: value.size(),
            data: array_d,
        }
    }
}

impl<T: tch::kind::Element + IntoTensorElementType> From<Tensor<T>> for tch::Tensor {
    #[inline(always)]
    fn from(value: Tensor<T>) -> Self {
        tch::Tensor::from_slice(value.data.as_standard_layout().as_slice().unwrap())
            .view(value.dimensions.as_slice())
    }
}

impl From<tch::Tensor> for DynTensor {
    #[inline(always)]
    fn from(value: tch::Tensor) -> Self {
        let dtype = value.kind().into();
        macro_rules! tensor_into_dyn {
            ($t:ty) => {{
                let tensor: Tensor<$t> = value.into();
                tensor.into_dyn()
            }};
        }
        match dtype {
            TensorElementType::UInt8 => tensor_into_dyn!(u8),
            TensorElementType::Int8 => tensor_into_dyn!(i8),
            TensorElementType::Int16 => tensor_into_dyn!(i16),
            TensorElementType::Int => tensor_into_dyn!(i32),
            TensorElementType::Int64 => tensor_into_dyn!(i64),
            TensorElementType::Float => tensor_into_dyn!(f32),
            TensorElementType::Double => tensor_into_dyn!(f64),
            TensorElementType::Bool => tensor_into_dyn!(bool),
            TensorElementType::Half
            | TensorElementType::BFloat16
            | TensorElementType::ComplexHalf
            | TensorElementType::ComplexFloat
            | TensorElementType::ComplexDouble
            | TensorElementType::QInt8
            | TensorElementType::QUInt8
            | TensorElementType::QInt32 => {
                unimplemented!()
            }
            #[cfg(feature = "onnx")]
            TensorElementType::UInt16 | TensorElementType::UInt | TensorElementType::UInt64 => {
                unreachable!()
            }
        }
    }
}

impl From<DynTensor> for tch::Tensor {
    fn from(value: DynTensor) -> Self {
        macro_rules! dyn_into_tensor {
            ($t:ty) => {{
                let tensor: Tensor<$t> = value.downcast().unwrap();
                tch::Tensor::from(tensor)
            }};
        }
        match value.dtype {
            TensorElementType::UInt8 => dyn_into_tensor!(u8),
            TensorElementType::Int8 => dyn_into_tensor!(i8),
            TensorElementType::Int16 => dyn_into_tensor!(i16),
            TensorElementType::Int => dyn_into_tensor!(i32),
            TensorElementType::Int64 => dyn_into_tensor!(i64),
            TensorElementType::Float => dyn_into_tensor!(f32),
            TensorElementType::Double => dyn_into_tensor!(f64),
            TensorElementType::Bool => dyn_into_tensor!(bool),
            #[cfg(feature = "onnx")]
            TensorElementType::UInt16 | TensorElementType::UInt | TensorElementType::UInt64 => {
                unreachable!()
            }
            TensorElementType::Half
            | TensorElementType::BFloat16
            | TensorElementType::ComplexHalf
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
