use ndarray::ArrayD;

use crate::tensor::{DynTensor, Tensor, TensorElementType};

mod test_tensor_operation;

#[test]
fn test_tensor_creation() {
    let data = ArrayD::from_elem(vec![2, 2], 1.0f32);
    let tensor = Tensor::from(data);

    assert_eq!(tensor.dtype(), TensorElementType::Float);
    assert_eq!(tensor.dimensions(), vec![2, 2]);
}

#[test]
fn test_batch_only_creation() {
    let tensor = Tensor::<f32>::ones(vec![12, 0, 0, 0]);
    assert_eq!(tensor.dtype(), TensorElementType::Float);
    assert_eq!(tensor.dimensions(), vec![12, 0, 0, 0]);
}

#[test]
fn test_dyn_tensor_creation() {
    let data = ArrayD::from_elem(vec![2, 2], 1.0f32);
    let dyn_tensor = DynTensor::from(data.clone());
    let tensor = dyn_tensor.downcast::<f32>().unwrap();

    assert_eq!(tensor.dtype(), TensorElementType::Float);
    assert_eq!(tensor.dimensions(), vec![2, 2]);
    assert_eq!(tensor.data(), data);
}

#[test]
fn test_dyn_tensor_downcast() {
    let data = ArrayD::from_elem(vec![2, 2], 1.0f32);
    let tensor = Tensor::from(data);
    let dyn_tensor = tensor.clone().into_dyn();
    let downcasted_tensor: Option<Tensor<f32>> = dyn_tensor.downcast();

    assert!(downcasted_tensor.is_some());
    let downcasted_tensor = downcasted_tensor.unwrap();
    assert_eq!(downcasted_tensor.dtype(), TensorElementType::Float);
    assert_eq!(downcasted_tensor.dimensions(), vec![2, 2]);
}

#[test]
#[cfg(any(feature = "half", feature = "torch"))]
fn test_half() {
    let data = ArrayD::from_elem(vec![2, 2], half::f16::from_f32(1.0));
    let tensor = Tensor::from(data);

    assert_eq!(tensor.dtype(), TensorElementType::Half);
    assert_eq!(tensor.dimensions(), vec![2, 2]);
}

#[test]
#[cfg(feature = "onnx")]
fn test_uint16() {
    let data = ArrayD::from_elem(vec![2, 2], 1u16);
    let tensor = Tensor::from(data);

    assert_eq!(tensor.dtype(), TensorElementType::UInt16);
    assert_eq!(tensor.dimensions(), vec![2, 2]);
}
