//! 不同推理框架模型输入输出Tensor的抽象

#[cfg(feature = "bincode")]
use bincode::{Decode, Encode};
pub use dyn_tensor::*;
use ndarray::{ArrayD, ArrayView, IxDyn};
pub use num_traits;
#[cfg(feature = "rand")]
use rand::random_iter;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

mod dyn_tensor;

#[cfg(feature = "torch")]
mod torch_ext;

#[cfg(feature = "onnx")]
mod onnx_ext;

#[cfg(feature = "serde")]
mod serde_ext;

#[cfg(feature = "bincode")]
mod bincode_ext;

/// 动态维度的`ArrayView`
pub type ArrayViewD<'a, T> = ArrayView<'a, T, IxDyn>;
/// 动态维度的`ArrayViewMut`
pub type ArrayViewMutD<'a, T> = ndarray::ArrayViewMut<'a, T, IxDyn>;

/// 对不同框架[Tensor]数据的抽象
///
/// 主要用于在系统内部传输不同框架下[Tensor]数据
/// 内部存储:
/// - 对应底层数据类型[Tensor::dtype]
/// - `Tensor`维度[Tensor::dimensions]
/// - 实际`Tensor`数据[Tensor::data]
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct Tensor<T: IntoTensorElementType> {
    /// 数据类型
    dtype: TensorElementType,
    /// `Tensor`维度
    dimensions: Vec<i64>,
    /// 实际`Tensor`数据
    data: ArrayD<T>,
}

impl<T: IntoTensorElementType> Tensor<T> {
    /// 获取数据类型
    pub fn dtype(&self) -> TensorElementType {
        self.dtype.clone()
    }

    /// 获取`Tensor`维度
    pub fn dimensions(&self) -> &[i64] {
        &self.dimensions
    }

    /// 获取实际`Tensor`数据
    pub fn data(&self) -> ArrayViewD<'_, T> {
        self.data.view()
    }

    /// 创建一个全1的`Tensor`
    pub fn ones(shape: Vec<usize>) -> Self
    where
        T: num_traits::identities::One,
    {
        let data = ArrayD::<T>::ones(IxDyn(&shape));
        Tensor::from(data)
    }

    /// 创建一个全0的`Tensor`
    pub fn zeros(shape: Vec<usize>) -> Self
    where
        T: num_traits::identities::Zero,
    {
        let data = ArrayD::<T>::zeros(IxDyn(&shape));
        Tensor::from(data)
    }

    /// 创建一个随机值的`Tensor`
    #[cfg(feature = "rand")]
    pub fn random(shape: Vec<usize>) -> Self
    where
        rand::distr::StandardUniform: rand::distr::Distribution<T>,
    {
        let element = random_iter()
            .take(shape.iter().product())
            .collect::<Vec<_>>();

        let data = ArrayD::<T>::from_shape_vec(IxDyn(&shape), element).unwrap();
        Tensor::from(data)
    }
}

impl<T: IntoTensorElementType> From<ArrayD<T>> for Tensor<T> {
    fn from(data: ArrayD<T>) -> Self {
        Tensor {
            dtype: T::into_tensor_element_type(),
            dimensions: data.shape().iter().map(|&x| x as i64).collect(),
            data,
        }
    }
}

/// 模型输入输出数据类型
#[derive(Eq, PartialEq, Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u32)]
pub enum TensorElementType {
    /// uint8
    UInt8 = 0,
    /// int8
    Int8 = 1,
    /// int16
    Int16 = 2,
    /// int32
    Int = 3,
    /// int64
    Int64 = 4,
    /// float32
    Float = 5,
    /// float64
    Double = 6,
    /// bool
    Bool = 7,

    /// float16
    #[cfg(any(feature = "half", feature = "torch"))]
    Half = 8,
    /// bfloat16
    #[cfg(any(feature = "half", feature = "torch"))]
    BFloat16 = 9,

    /// uint16
    #[cfg(feature = "onnx")]
    UInt16 = 10,
    /// uint32
    #[cfg(feature = "onnx")]
    UInt = 11,
    /// uint64
    #[cfg(feature = "onnx")]
    UInt64 = 12,

    /// complex64
    #[cfg(feature = "torch")]
    ComplexHalf = 13,
    /// complex128
    #[cfg(feature = "torch")]
    ComplexFloat = 14,
    /// complex256
    #[cfg(feature = "torch")]
    ComplexDouble = 15,
    /// qint8
    #[cfg(feature = "torch")]
    QInt8 = 16,
    /// quint8
    #[cfg(feature = "torch")]
    QUInt8 = 17,
    /// qint32
    #[cfg(feature = "torch")]
    QInt32 = 18,
}

impl TensorElementType {
    /// Types supported by the common Torch/ONNX dynamic conversion path.
    pub fn supports_common_io(&self) -> bool {
        matches!(
            self,
            Self::UInt8
                | Self::Int8
                | Self::Int16
                | Self::Int
                | Self::Int64
                | Self::Float
                | Self::Double
                | Self::Bool
        )
    }
}

/// 模型运行设备或Tensor所在设备
#[derive(Eq, PartialEq, Debug, Clone)]
#[cfg_attr(feature = "bincode", derive(Encode, Decode))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Device {
    /// 位于CPU上
    Cpu,
    /// 位于GPU(GPUID)上
    Gpu(usize),
}

/// 可以转换到[TensorElementType]的类型
///
/// 为rust标准库类型和half库类型提供了实现，
/// 部分框架自定类型未提供实现
pub trait IntoTensorElementType: 'static + Send + Sync + Clone {
    /// 转换到[TensorElementType]
    #[allow(clippy::wrong_self_convention)]
    fn into_tensor_element_type() -> TensorElementType;
}

macro_rules! impl_type_trait {
    ($type_:ty, $variant:ident) => {
        impl IntoTensorElementType for $type_ {
            fn into_tensor_element_type() -> TensorElementType {
                TensorElementType::$variant
            }
        }
    };
}

impl_type_trait!(f32, Float);
impl_type_trait!(u8, UInt8);
impl_type_trait!(i8, Int8);
impl_type_trait!(i16, Int16);
impl_type_trait!(i32, Int);
impl_type_trait!(i64, Int64);
impl_type_trait!(bool, Bool);
impl_type_trait!(f64, Double);

#[cfg(feature = "onnx")]
impl_type_trait!(u16, UInt16);
#[cfg(feature = "onnx")]
impl_type_trait!(u32, UInt);
#[cfg(feature = "onnx")]
impl_type_trait!(u64, UInt64);

#[cfg(any(feature = "half", feature = "torch"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "half", feature = "torch"))))]
impl_type_trait!(half::f16, Half);
#[cfg(any(feature = "half", feature = "torch"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "half", feature = "torch"))))]
impl_type_trait!(half::bf16, BFloat16);
