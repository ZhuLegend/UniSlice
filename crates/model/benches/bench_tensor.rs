pub use divan;
use divan::AllocProfiler;
use model::tensor::{DynTensor, IntoTensorElementType};
use ndarray::Array;

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    // Run registered benchmarks.
    divan::main();
}

fn gen_array<T: Default>(size: usize) -> impl Fn() -> ndarray::Array3<T> {
    move || Array::default((1000, size, size))
}

#[divan::bench(
    types = [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64],
    args = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    max_time = 1
)]
fn into_dyn<T: Default + IntoTensorElementType>(bench: divan::Bencher, size: usize) {
    bench
        .with_inputs(gen_array::<T>(size))
        .bench_values(DynTensor::from);
}

#[divan::bench(
    types = [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64],
    args = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    max_time = 1
)]
fn dyn_tensor_concat<T: Default + IntoTensorElementType>(bench: divan::Bencher, size: usize) {
    bench
        .with_inputs(|| {
            let arr1 = gen_array::<T>(size)();
            let arr2 = gen_array::<T>(size)();

            (DynTensor::from(arr1), DynTensor::from(arr2))
        })
        .bench_values(|(arr1, arr2)| DynTensor::concat_batch([arr1, arr2]).unwrap());
}

#[divan::bench(
    types = [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64],
    args = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    max_time = 1
)]
fn dyn_tensor_split<T: Default + IntoTensorElementType>(bench: divan::Bencher, size: usize) {
    bench
        .with_inputs(|| {
            let tensor = DynTensor::from(gen_array::<T>(size)());
            let indices = vec![size / 4, size / 2, size * 3 / 4];
            (tensor, indices)
        })
        .bench_values(|(tensor, indices)| tensor.split_batch(indices).unwrap());
}

fn gen_image_tensor<T: Default + IntoTensorElementType>(batch_size: usize) -> DynTensor {
    let arr: ndarray::Array4<T> = Array::default((batch_size, 3, 224, 224));
    DynTensor::from(arr)
}

#[cfg(feature = "serde")]
#[divan::bench(
    types = [f32, f64, i32, u8],
    args = [1, 2, 4, 8, 16, 32],
    max_time = 1
)]
fn serde_json_serialize<T: Default + IntoTensorElementType + serde::Serialize>(
    bench: divan::Bencher,
    batch_size: usize,
) {
    bench
        .with_inputs(|| gen_image_tensor::<T>(batch_size))
        .bench_values(|tensor| serde_json::to_string(&tensor).unwrap());
}

#[cfg(feature = "serde")]
#[divan::bench(
    types = [f32, f64, i32, u8],
    args = [1, 2, 4, 8, 16, 32],
    max_time = 1
)]
fn serde_json_deserialize<
    T: Default + IntoTensorElementType + serde::Serialize + for<'de> serde::Deserialize<'de>,
>(
    bench: divan::Bencher,
    batch_size: usize,
) {
    bench
        .with_inputs(|| {
            let tensor = gen_image_tensor::<T>(batch_size);
            serde_json::to_string(&tensor).unwrap()
        })
        .bench_values(|json_str| {
            let _: DynTensor = serde_json::from_str(&json_str).unwrap();
        });
}

#[cfg(feature = "serde")]
#[divan::bench(
    types = [f32, f64, i32, u8],
    args = [1, 2, 4, 8, 16, 32],
    max_time = 1
)]
fn serde_json_roundtrip<
    T: Default + IntoTensorElementType + serde::Serialize + for<'de> serde::Deserialize<'de>,
>(
    bench: divan::Bencher,
    batch_size: usize,
) {
    bench
        .with_inputs(|| gen_image_tensor::<T>(batch_size))
        .bench_values(|tensor| {
            let json_str = serde_json::to_string(&tensor).unwrap();
            let _: DynTensor = serde_json::from_str(&json_str).unwrap();
        });
}

#[cfg(feature = "bincode")]
#[divan::bench(
    types = [f32, f64, i32, u8],
    args = [1, 2, 4, 8, 16, 32],
    max_time = 1
)]
fn bincode_serialize<T: Default + IntoTensorElementType + serde::Serialize>(
    bench: divan::Bencher,
    batch_size: usize,
) {
    bench
        .with_inputs(|| gen_image_tensor::<T>(batch_size))
        .bench_values(|tensor| {
            bincode::encode_to_vec(&tensor, bincode::config::standard()).unwrap()
        });
}

#[cfg(feature = "bincode")]
#[divan::bench(
    types = [f32, f64, i32, u8],
    args = [1, 2, 4, 8, 16, 32],
    max_time = 1
)]
fn bincode_deserialize<
    T: Default + IntoTensorElementType + serde::Serialize + for<'de> serde::Deserialize<'de>,
>(
    bench: divan::Bencher,
    batch_size: usize,
) {
    bench
        .with_inputs(|| {
            let tensor = gen_image_tensor::<T>(batch_size);
            bincode::encode_to_vec(&tensor, bincode::config::standard()).unwrap()
        })
        .bench_values(|bytes| {
            let _: (DynTensor, usize) =
                bincode::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
        });
}

#[cfg(feature = "bincode")]
#[divan::bench(
    types = [f32, f64, i32, u8],
    args = [1, 2, 4, 8, 16, 32],
    max_time = 1
)]
fn bincode_roundtrip<
    T: Default + IntoTensorElementType + serde::Serialize + for<'de> serde::Deserialize<'de>,
>(
    bench: divan::Bencher,
    batch_size: usize,
) {
    bench
        .with_inputs(|| gen_image_tensor::<T>(batch_size))
        .bench_values(|tensor| {
            let bytes = bincode::encode_to_vec(&tensor, bincode::config::standard()).unwrap();
            let _: (DynTensor, usize) =
                bincode::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
        });
}
