use model::tensor::{DynTensor, TensorElementType};
use ndarray::{Array2, Axis, Slice};

#[test]
fn sliced_batches_preserve_only_logical_elements() {
    let mut source = Array2::from_shape_vec((4, 2), (0..8).collect::<Vec<i32>>()).unwrap();
    source.slice_axis_inplace(Axis(0), Slice::from(1..2));
    let tail = Array2::from_shape_vec((1, 2), vec![8, 9]).unwrap();
    let result = DynTensor::concat_batch([source.into(), tail.into()]).unwrap();
    assert_eq!(result.dimensions(), &[2, 2]);
    let parts = result.split_batch(vec![1]).unwrap();
    let values = parts
        .into_iter()
        .map(|part| {
            part.downcast::<i32>()
                .unwrap()
                .data()
                .iter()
                .copied()
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(values, vec![vec![2, 3], vec![8, 9]]);
}

#[test]
fn contiguous_buffers_reuse_available_capacity_and_keep_the_first_split_buffer() {
    let mut values = Vec::with_capacity(8);
    values.extend(0..4i32);
    let pointer = values.as_ptr();
    let first = Array2::from_shape_vec((2, 2), values).unwrap();
    let second = Array2::from_shape_vec((2, 2), (4..8i32).collect()).unwrap();
    let joined = DynTensor::concat_batch([first.into(), second.into()]).unwrap();
    let parts = joined.split_batch(vec![2]).unwrap();
    let first = parts.into_iter().next().unwrap().downcast::<i32>().unwrap();
    assert_eq!(first.data().as_ptr(), pointer);
    assert_eq!(
        first.data().iter().copied().collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
}

#[test]
fn contiguous_offset_need_not_be_a_multiple_of_the_logical_row_size() {
    let mut array = Array2::from_shape_vec((1, 6), (0..6i32).collect()).unwrap();
    array.slice_axis_inplace(Axis(1), Slice::from(1..5));
    assert!(array.is_standard_layout());
    let parts = DynTensor::from(array).split_batch(vec![1]).unwrap();
    assert_eq!(parts[0].dimensions(), &[1, 4]);
    assert_eq!(parts[1].dimensions(), &[0, 4]);
    assert_eq!(
        parts
            .into_iter()
            .next()
            .unwrap()
            .downcast::<i32>()
            .unwrap()
            .data()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![1, 2, 3, 4]
    );
}

#[test]
fn invalid_split_returns_error_and_empty_non_batch_axes_are_valid() {
    let input: DynTensor = Array2::<f32>::zeros((5, 2)).into();
    assert!(input.clone().split_batch(vec![6]).is_err());
    assert!(input.split_batch(vec![4, 2]).is_err());
    let input: DynTensor = Array2::<f32>::zeros((5, 0)).into();
    let parts = input.split_batch(vec![2]).unwrap();
    assert_eq!(parts[0].dimensions(), &[2, 0]);
    assert_eq!(parts[1].dimensions(), &[3, 0]);
}

#[test]
fn split_cropped_and_strided_arrays_preserves_logical_values() {
    for step in [1, 2, -1] {
        let mut array = Array2::from_shape_vec((6, 2), (0..12).collect::<Vec<i32>>()).unwrap();
        array.slice_axis_inplace(Axis(0), ndarray::Slice::new(1, Some(5), step));
        let expected = array.iter().copied().collect::<Vec<_>>();
        let len = array.shape()[0];
        let parts = DynTensor::from(array)
            .split_batch(vec![0, 1, 1, len])
            .unwrap();
        assert_eq!(parts.len(), 5);
        let actual = parts
            .into_iter()
            .flat_map(|part| {
                part.downcast::<i32>()
                    .unwrap()
                    .data()
                    .iter()
                    .copied()
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
}

#[test]
fn transposed_concat_uses_logical_rows_and_rejects_mismatched_shapes() {
    let array = Array2::from_shape_vec((2, 3), (0..6).collect::<Vec<i32>>())
        .unwrap()
        .reversed_axes();
    let expected = array
        .iter()
        .copied()
        .chain(array.iter().copied())
        .collect::<Vec<_>>();
    let joined = DynTensor::concat_batch([array.clone().into(), array.into()]).unwrap();
    assert_eq!(
        joined
            .downcast::<i32>()
            .unwrap()
            .data()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        expected
    );
    assert!(
        DynTensor::concat_batch([
            Array2::<i32>::zeros((1, 2)).into(),
            Array2::<i32>::zeros((1, 3)).into()
        ])
        .is_err()
    );
    assert!(
        DynTensor::concat_batch([
            Array2::<i32>::zeros((1, 2)).into(),
            Array2::<f32>::zeros((1, 2)).into()
        ])
        .is_err()
    );
    let empty = DynTensor::from(Array2::<i32>::zeros((0, 2)))
        .split_batch(vec![0])
        .unwrap();
    assert_eq!(empty.len(), 2);
    assert_eq!(empty[1].dimensions(), &[0, 2]);
}

#[cfg(all(feature = "bincode", feature = "serde"))]
#[test]
fn wire_dimensions_must_match_the_actual_array() {
    let config = bincode::config::standard();
    for dimensions in [vec![32i64, 2], vec![-1, 2], vec![1]] {
        let bytes = bincode::encode_to_vec(
            (
                TensorElementType::Float,
                dimensions,
                bincode::serde::Compat(Array2::<f32>::zeros((1, 2)).into_dyn()),
            ),
            config,
        )
        .unwrap();
        assert!(bincode::decode_from_slice::<DynTensor, _>(&bytes, config).is_err());
    }
    let encoded =
        serde_json::json!({"Float": [[32, 2], {"v": 1, "dim": [1, 2], "data": [0.0, 0.0]}]});
    assert!(serde_json::from_value::<DynTensor>(encoded).is_err());
}

#[cfg(all(feature = "bincode", feature = "serde", feature = "onnx"))]
#[test]
fn uint16_round_trip_preserves_type_and_values() {
    let input: DynTensor = Array2::<u16>::from_elem((1, 2), 500).into();
    let config = bincode::config::standard();
    let bytes = bincode::encode_to_vec(input, config).unwrap();
    let (decoded, _): (DynTensor, usize) = bincode::decode_from_slice(&bytes, config).unwrap();
    assert_eq!(
        decoded
            .downcast::<u16>()
            .unwrap()
            .data()
            .as_slice()
            .unwrap(),
        &[500, 500]
    );
}

#[cfg(all(feature = "bincode", feature = "serde"))]
#[test]
fn typed_decode_rejects_wrong_dtype_and_wire_tags_are_stable() {
    let json = serde_json::json!({"dtype":"Int64", "dimensions":[1,2],
        "data":{"v":1,"dim":[1,2],"data":[1,2]}});
    assert!(serde_json::from_value::<model::tensor::Tensor<i32>>(json).is_err());
    let wire = bincode::encode_to_vec(
        (
            TensorElementType::Int64,
            vec![1i64, 2],
            bincode::serde::Compat(Array2::<i32>::zeros((1, 2)).into_dyn()),
        ),
        bincode::config::standard(),
    )
    .unwrap();
    assert!(
        bincode::decode_from_slice::<model::tensor::Tensor<i32>, _>(
            &wire,
            bincode::config::standard()
        )
        .is_err()
    );
    let config = bincode::config::standard();
    assert_eq!(
        bincode::encode_to_vec(TensorElementType::Float, config).unwrap(),
        vec![5]
    );
    #[cfg(feature = "onnx")]
    assert_eq!(
        bincode::encode_to_vec(TensorElementType::UInt16, config).unwrap(),
        vec![10]
    );
}

#[test]
fn common_boolean_type_supports_batch_only_construction() {
    assert!(
        DynTensor::ones(vec![2, 1], TensorElementType::Bool)
            .downcast::<bool>()
            .unwrap()
            .data()
            .iter()
            .all(|value| *value)
    );
    assert!(
        DynTensor::zeros(vec![2, 1], TensorElementType::Bool)
            .downcast::<bool>()
            .unwrap()
            .data()
            .iter()
            .all(|value| !*value)
    );
    #[cfg(feature = "rand")]
    assert_eq!(
        DynTensor::random(vec![2, 1], TensorElementType::Bool).dimensions(),
        &[2, 1]
    );
}

#[cfg(feature = "onnx")]
#[test]
fn onnx_conversion_uses_only_the_logical_slice() {
    let mut array = Array2::from_shape_vec((4, 2), (0..8).collect::<Vec<i32>>()).unwrap();
    array.slice_axis_inplace(Axis(0), Slice::from(1..2));
    let tensor = ort::value::DynTensor::from(DynTensor::from(array));
    let (shape, values) = tensor.try_extract_tensor::<i32>().unwrap();
    assert_eq!(shape.as_ref(), &[1, 2]);
    assert_eq!(values, &[2, 3]);
}
