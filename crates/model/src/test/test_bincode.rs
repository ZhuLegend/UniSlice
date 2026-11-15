use bincode::config;
use ndarray::ArrayD;

use crate::tensor::{DynTensor, Tensor};

#[test]
fn test_tensor_i32() {
    let config = config::standard();
    let data = ArrayD::from_elem(vec![2, 2], 42);

    let tensor = Tensor::from(data);

    let encoded: Vec<u8> = bincode::encode_to_vec(&tensor, config).unwrap();

    let (decoded, len): (Tensor<i32>, usize) =
        bincode::decode_from_slice(&encoded[..], config).unwrap();

    assert_eq!(tensor, decoded);
    assert_eq!(len, encoded.len());
}

#[cfg(feature = "onnx")]
#[test]
fn test_tensor_u64() {
    let config = config::standard();
    let data = ArrayD::from_elem(vec![2, 2], 42u64);

    let tensor = Tensor::from(data);

    let encoded: Vec<u8> = bincode::encode_to_vec(&tensor, config).unwrap();

    let (decoded, len): (Tensor<u64>, usize) =
        bincode::decode_from_slice(&encoded[..], config).unwrap();

    assert_eq!(tensor, decoded);
    assert_eq!(len, encoded.len());
}

#[cfg(any(feature = "half", feature = "torch"))]
#[test]
fn test_dyn_tensor_half() {
    let config = config::standard();

    let data = ArrayD::from_elem(vec![2, 2], half::f16::from_f32(1.0));
    let dyn_tensor = DynTensor::from(data);

    let encoded: Vec<u8> = bincode::encode_to_vec(&dyn_tensor, config).unwrap();

    let (decoded, len): (DynTensor, usize) =
        bincode::decode_from_slice(&encoded[..], config).unwrap();

    assert_eq!(dyn_tensor.dtype, decoded.dtype);
    assert_eq!(dyn_tensor.dimensions, decoded.dimensions);
    assert_eq!(len, encoded.len());
}

#[test]
fn test_dyn_tensor_i32() {
    let config = config::standard();

    let data = ArrayD::from_elem(vec![2, 2], 42);
    let dyn_tensor = DynTensor::from(data);

    let encoded: Vec<u8> = bincode::encode_to_vec(&dyn_tensor, config).unwrap();

    let (decoded, len): (DynTensor, usize) =
        bincode::decode_from_slice(&encoded[..], config).unwrap();

    assert_eq!(dyn_tensor.dtype, decoded.dtype);
    assert_eq!(dyn_tensor.dimensions, decoded.dimensions);
    assert_eq!(len, encoded.len());
}
