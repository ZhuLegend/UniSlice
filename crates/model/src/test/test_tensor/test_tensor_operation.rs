use ndarray::{Array, Array2, Axis, array, s};

use crate::tensor::{
    DynTensor, TensorElementType, TensorOperationError, concat_batches, split_batch,
};

#[test]
fn test_ndarray_split() {
    // 创建 C-连续数组 [ [0, 1], [2, 3], [4, 5], [6, 7] ]
    let arr = Array::linspace(0.0, 7.0, 8)
        .to_shape((4, 2))
        .unwrap()
        .to_owned();
    let vec = vec![1, 2];
    let result = split_batch(arr, vec).unwrap();

    // 验证形状
    assert_eq!(result[0].shape(), &[1, 2]);
    assert_eq!(result[1].shape(), &[1, 2]);
    assert_eq!(result[2].shape(), &[2, 2]);

    // 验证数据
    assert_eq!(result[0], array![[0.0, 1.0]]);
    assert_eq!(result[1], array![[2.0, 3.0]]);
    assert_eq!(result[2], array![[4.0, 5.0], [6.0, 7.0]]);
}

#[test]
fn test_slice_ndarray_split() {
    let mut arr = Array::linspace(0.0, 7.0, 8)
        .to_shape((4, 2))
        .unwrap()
        .to_owned();
    arr.slice_axis_inplace(Axis(0), (1..).into());
    let mut view = arr.slice_mut(s![1..3, ..]);
    view += 1.0;

    let vec = vec![1];
    let result = split_batch(arr, vec).unwrap();

    assert_eq!(result[0].shape(), [1, 2]);
    assert_eq!(result[1].shape(), [2, 2]);

    assert_eq!(result[0], array![[2.0, 3.0]]);
    assert_eq!(result[1], array![[5.0, 6.0], [7.0, 8.0]]);
}

/// 测试拆分索引为0的边界情况
#[test]
fn test_split_at_zero() {
    let arr: Array2<u32> = Array::zeros((5, 3));
    let vec = vec![0];
    let result = split_batch(arr, vec).unwrap();

    assert_eq!(result[0].shape(), &[0, 3]); // 空数组
    assert_eq!(result[1].shape(), &[5, 3]);
}

/// 测试拆分索引等于轴长度的边界情况
#[test]
fn test_split_at_max_index() {
    let arr: Array2<u32> = Array::zeros((5, 3));
    let vec = vec![5];
    let result = split_batch(arr, vec).unwrap();

    assert_eq!(result[0].shape(), &[5, 3]);
    assert_eq!(result[1].shape(), &[0, 3]); // 空数组
}

/// 测试输入非C-连续数组时正确拆分
#[test]
fn test_non_contiguous_input() {
    // 创建一个非C-连续的数组（转置后的数组）
    let arr = Array::linspace(0.0, 5.0, 6)
        .to_shape((3, 2))
        .unwrap()
        .reversed_axes()
        .to_owned();
    let vec = vec![1];
    let result = split_batch(arr, vec).unwrap();

    assert_eq!(result[0].shape(), [1, 3]);
    assert_eq!(result[1].shape(), [1, 3]);

    assert_eq!(result[0], array![[0.0, 2.0, 4.0]]);
    assert_eq!(result[1], array![[1.0, 3.0, 5.0]]);
}

/// 测试索引越界时触发panic
#[test]
fn test_index_out_of_bounds() {
    let arr: Array2<u32> = Array::zeros((5, 3));
    let vec = vec![6];
    if let Err(TensorOperationError::InvalidIndex { .. }) = split_batch(arr, vec) {
    } else {
        panic!("Expected InvalidIndex error")
    }
}

/// 验证内存无复制（通过指针地址）
#[test]
fn test_no_memory_copy() {
    let arr = Array::linspace(0.0, 5.0, 6)
        .to_shape((3, 2))
        .unwrap()
        .to_owned();
    let ptr_before = dbg!(arr.as_ptr());

    let vec = vec![1];
    let result = split_batch(arr, vec).unwrap();

    // 检查子数组的指针是否指向原始内存区域
    assert_eq!(result[0].as_ptr(), ptr_before);
}

/// 测试[DynTensor]拆分
#[test]
fn test_dyn_tensor_split() {
    let arr = Array::linspace(0.0, 7.0, 8)
        .to_shape((4, 2))
        .unwrap()
        .to_owned();
    let dyn_tensor = DynTensor::from(arr);

    let vec = vec![2];
    let dyn_tensors = dyn_tensor.split_batch(vec).unwrap();
    let arr1 = dyn_tensors[0].clone().downcast::<f64>().unwrap();
    let arr2 = dyn_tensors[1].clone().downcast::<f64>().unwrap();

    assert_eq!(arr1.dimensions(), [2, 2]);
    assert_eq!(arr2.dimensions(), [2, 2]);

    assert_eq!(arr1.data(), array![[0.0, 1.0], [2.0, 3.0]].into_dyn());
    assert_eq!(arr2.data(), array![[4.0, 5.0], [6.0, 7.0]].into_dyn());

    let empty_dyn_tensor = DynTensor::zeros(vec![3, 0], TensorElementType::Double);
    let tensors = empty_dyn_tensor.split_batch(vec![1]).unwrap();
    let tensor1 = tensors[0].clone().downcast::<f64>().unwrap();
    let tensor2 = tensors[1].clone().downcast::<f64>().unwrap();
    assert_eq!(tensor1.dimensions(), vec![1, 0]);
    assert_eq!(tensor2.dimensions(), vec![2, 0]);
    assert_eq!(tensor1.data(), Array2::<f64>::zeros((1, 0)).into_dyn());
    assert_eq!(tensor2.data(), Array2::<f64>::zeros((2, 0)).into_dyn());
}

/// 测试拼接两个数组
#[test]
fn test_ndarray_concat() {
    // 创建两个数组
    let arr1 = Array::linspace(0.0, 3.0, 4)
        .to_shape((2, 2))
        .unwrap()
        .to_owned();
    let arr2 = Array::linspace(4.0, 7.0, 4)
        .to_shape((2, 2))
        .unwrap()
        .to_owned();

    // 拼接两个数组
    let arr = concat_batches([arr1, arr2]).unwrap();

    // 验证形状
    assert_eq!(arr.shape(), &[4, 2]);

    // 验证数据
    assert_eq!(arr, array![[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]);
}

/// 测试切片后拼接
#[test]
fn test_slice_ndarray_concat() {
    let mut arr1 = Array::linspace(0.0, 3.0, 4)
        .to_shape((2, 2))
        .unwrap()
        .to_owned();
    arr1.slice_axis_inplace(Axis(0), (1..).into());

    let mut arr2 = Array::linspace(4.0, 7.0, 4)
        .to_shape((2, 2))
        .unwrap()
        .to_owned();
    arr2.slice_axis_inplace(Axis(0), (1..).into());

    let arr = concat_batches([arr1, arr2]).unwrap();

    assert_eq!(arr.shape(), [2, 2]);
    assert_eq!(arr, array![[2.0, 3.0], [6.0, 7.0]]);
}

/// 测试拼接空数组
#[test]
fn test_concat_empty() {
    let arr: Array2<u32> = Array::zeros((0, 3));
    let arr = concat_batches([arr.clone(), arr.clone()]).unwrap();

    assert_eq!(arr.shape(), [0, 3]);
}

/// 测试[DynTensor]拼接
#[test]
fn test_dyn_tensor_concat() {
    let arr1 = Array::linspace(0.0, 3.0, 4)
        .to_shape((2, 2))
        .unwrap()
        .to_owned();
    let arr2 = Array::linspace(4.0, 7.0, 4)
        .to_shape((2, 2))
        .unwrap()
        .to_owned();
    let arr3 = Array::linspace(8.0, 11.0, 4)
        .to_shape((2, 2))
        .unwrap()
        .to_owned();

    let dyn_tensor1 = DynTensor::from(arr1);
    let dyn_tensor2 = DynTensor::from(arr2);
    let dyn_tensor3 = DynTensor::from(arr3);

    let dyn_tensor = DynTensor::concat_batch([dyn_tensor1, dyn_tensor2, dyn_tensor3]).unwrap();
    let arr = dyn_tensor.downcast::<f64>().unwrap();

    assert_eq!(arr.dimensions(), [6, 2]);
    assert_eq!(
        arr.data(),
        array![
            [0.0, 1.0],
            [2.0, 3.0],
            [4.0, 5.0],
            [6.0, 7.0],
            [8.0, 9.0],
            [10.0, 11.0]
        ]
        .into_dyn()
    );

    let empty_dyn_tensor1 = DynTensor::zeros(vec![2, 0], TensorElementType::Double);
    let empty_dyn_tensor2 = DynTensor::zeros(vec![3, 0], TensorElementType::Double);
    let tensor = DynTensor::concat_batch([empty_dyn_tensor1, empty_dyn_tensor2])
        .unwrap()
        .downcast::<f64>()
        .unwrap();
    assert_eq!(tensor.dimensions(), vec![5, 0]);
    assert_eq!(
        tensor.data(),
        Array2::<f64>::zeros((5, 0)).view().into_dyn()
    );
}
