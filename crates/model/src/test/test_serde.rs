use ndarray::ArrayD;

use crate::tensor::DynTensor;

#[test]
fn test_dyn_tensor_serde_i32() {
    let data = ArrayD::from_shape_vec(vec![2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
    let dyn_tensor = DynTensor::from(data.clone());
    let serialized = serde_json::to_string(&dyn_tensor).unwrap();
    let deserialized: DynTensor = serde_json::from_str(&serialized).unwrap();
    assert_eq!(dyn_tensor.dtype, deserialized.dtype);
    assert_eq!(dyn_tensor.dimensions, deserialized.dimensions);
    let data0 = dyn_tensor.data.downcast_ref::<ArrayD<i32>>().unwrap();
    let data1 = deserialized.data.downcast_ref::<ArrayD<i32>>().unwrap();
    assert_eq!(data0, data1);
}

#[test]
#[cfg(feature = "onnx")]
fn test_dyn_tensor_serde_u64() {
    let data = ArrayD::from_shape_vec(vec![2, 3], vec![1u64, 2, 3, 4, 5, 6]).unwrap();
    let dyn_tensor = DynTensor::from(data);
    let serialized = serde_json::to_string(&dyn_tensor).unwrap();
    let deserialized: DynTensor = serde_json::from_str(&serialized).unwrap();
    assert_eq!(dyn_tensor.dtype, deserialized.dtype);
    assert_eq!(dyn_tensor.dimensions, deserialized.dimensions);
    let data0 = dyn_tensor.data.downcast_ref::<ArrayD<u64>>().unwrap();
    let data1 = deserialized.data.downcast_ref::<ArrayD<u64>>().unwrap();
    assert_eq!(data0, data1);
}

#[cfg(any(feature = "half", feature = "torch"))]
#[test]
fn test_dyn_tensor_serde_half() {
    let data = ArrayD::from_elem(vec![2, 2], half::f16::from_f32(1.0));
    let dyn_tensor = DynTensor::from(data);

    let serialized = serde_json::to_string(&dyn_tensor).unwrap();
    let deserialized: DynTensor = serde_json::from_str(&serialized).unwrap();
    assert_eq!(dyn_tensor.dtype, deserialized.dtype);
    assert_eq!(dyn_tensor.dimensions, deserialized.dimensions);
    let data0 = dyn_tensor.data.downcast_ref::<ArrayD<half::f16>>().unwrap();
    let data1 = deserialized
        .data
        .downcast_ref::<ArrayD<half::f16>>()
        .unwrap();
    assert_eq!(data0, data1);
}
