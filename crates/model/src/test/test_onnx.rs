use ndarray::ArrayD;

use crate::tensor::{DynTensor, Tensor, TensorElementType};
#[test]
fn test_ort_dyn_tensor_and_tensor() {
    let data = ArrayD::from_shape_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let tensor = Tensor::from(data);
    let ort_dyn_tensor = ort::value::DynTensor::from(tensor);
    let new_tensor = Tensor::<f64>::from(ort_dyn_tensor);
    assert_eq!(new_tensor.dtype(), TensorElementType::Double);
    assert_eq!(new_tensor.dimensions(), vec![2, 2]);
    assert_eq!(
        new_tensor.data(),
        ArrayD::from_shape_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap()
    );
}

#[test]
fn test_tensor_to_ort_tensor() {
    let data = ArrayD::from_shape_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let tensor = Tensor::from(data);
    let ort_tensor = ort::value::Tensor::from(tensor);
    assert_eq!(
        TensorElementType::from(ort_tensor.dtype()),
        TensorElementType::Double
    );
    assert_eq!(ort_tensor.shape().to_vec(), &[2, 2]);
    assert_eq!(ort_tensor.extract_tensor().1, &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_ort_dyn_tensor_and_dyn_tensor() {
    let data = ArrayD::from_shape_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    let dyn_tensor = DynTensor::from(data);

    let ort_dyn_tensor = ort::value::DynTensor::from(dyn_tensor);
    assert_eq!(
        TensorElementType::from(ort_dyn_tensor.dtype()),
        TensorElementType::Double
    );
    assert_eq!(ort_dyn_tensor.shape().to_vec(), &[2, 2]);
    let new_dyn_tensor = DynTensor::from(ort_dyn_tensor);
    assert_eq!(new_dyn_tensor.dtype, TensorElementType::Double);
    assert_eq!(new_dyn_tensor.dimensions, vec![2, 2]);
    let data = new_dyn_tensor.data.downcast_ref::<ArrayD<f64>>().unwrap();
    assert_eq!(
        data.clone().into_raw_vec_and_offset().0,
        vec![1.0, 2.0, 3.0, 4.0]
    );
}
