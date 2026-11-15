use ndarray::ArrayD;

use crate::tensor::{DynTensor, Tensor, TensorElementType};
#[cfg(feature = "torch")]
#[test]
fn test_tch_tensor_and_tensor_i32() {
    let data = ArrayD::from_elem(vec![2, 2], 1i32);
    let tensor = Tensor::from(data.clone());
    let tch_tensor = tch::Tensor::from(tensor);
    let new_tensor = Tensor::<i32>::from(tch_tensor);

    assert_eq!(new_tensor.dtype(), TensorElementType::Int);
    assert_eq!(new_tensor.dimensions(), vec![2, 2]);
    assert_eq!(new_tensor.data(), ArrayD::from_elem(vec![2, 2], 1i32));
}

#[cfg(any(feature = "half", feature = "torch"))]
#[test]
fn test_tch_tensor_and_tensor_half() {
    let data = ArrayD::from_elem(vec![2, 2], half::f16::from_f32(1.0));
    let tensor = Tensor::from(data);
    let tch_tensor = tch::Tensor::from(tensor.clone());
    let new_tensor = Tensor::<half::f16>::from(tch_tensor);

    assert_eq!(new_tensor.dtype(), TensorElementType::Half);
    assert_eq!(new_tensor.dimensions(), vec![2, 2]);
    assert_eq!(
        new_tensor.data(),
        ArrayD::from_elem(vec![2, 2], half::f16::from_f32(1.0))
    );
}

#[cfg(feature = "torch")]
#[test]
fn test_tch_dyn_tensor_and_dyn_tensor_i32() {
    let data = ArrayD::from_elem(vec![2, 2], 1i32);
    let dyn_tensor = DynTensor::from(data.clone());
    let tch_dyn_tensor = tch::Tensor::from(dyn_tensor);
    let new_dyn_tensor = DynTensor::from(tch_dyn_tensor);

    assert_eq!(new_dyn_tensor.dtype, TensorElementType::Int);
    assert_eq!(new_dyn_tensor.dimensions, vec![2, 2]);
    let data1 = new_dyn_tensor.data.downcast_ref::<ArrayD<i32>>().unwrap();
    assert_eq!(data1, data);
}
