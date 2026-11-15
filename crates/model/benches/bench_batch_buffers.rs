//! Bounded owned-buffer benchmarks. Inputs are created outside the timed region.
//! Include the production helpers so the benchmark does not expose private APIs.
use divan::AllocProfiler;
use ndarray::{ArrayD, Axis, Dimension, IxDyn, Slice};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

#[allow(dead_code)]
#[path = "../src/tensor/dyn_tensor/tensor_handler.rs"]
mod handler;
mod tensor {
    pub use crate::handler::TensorOperationError;
    pub mod dyn_tensor {
        pub type EraserData = dyn std::any::Any + Send + Sync;
    }
}

fn main() {
    validate_split_matrix();
    divan::main();
}

// (batch, image versus small tensor, cropped backing buffer)
type Case = (usize, bool, bool);
fn input((batch, image, cropped): Case) -> ArrayD<f32> {
    let mut shape = if image {
        vec![batch, 3, 224, 224]
    } else {
        vec![batch, 8]
    };
    if cropped {
        shape[0] += 2;
    }
    let mut arr = ArrayD::from_elem(IxDyn(&shape), 1.0);
    if cropped {
        arr.slice_axis_inplace(Axis(0), Slice::from(1..batch + 1));
    }
    arr
}
fn concat_input(case: Case) -> Vec<ArrayD<f32>> {
    // Two equally sized input batches, each with the reported batch size.
    vec![input(case), input(case)]
}
fn cuts(batch: usize) -> Vec<usize> {
    vec![batch / 4, batch / 2, batch * 3 / 4]
}

// Legacy Vec path, restricted to unsliced contiguous inputs. The old cropped
// buffer handling was incorrect, so it is deliberately not benchmarked there.
fn legacy_concat(arrs: Vec<ArrayD<f32>>) -> ArrayD<f32> {
    let mut shape = arrs[0].raw_dim();
    shape[0] *= arrs.len();
    let mut arrs = arrs.into_iter();
    let mut data = arrs.next().unwrap().into_raw_vec_and_offset().0;
    data.reserve_exact(shape.slice().iter().product::<usize>() - data.len());
    for arr in arrs {
        data.append(&mut arr.into_raw_vec_and_offset().0);
    }
    ArrayD::from_shape_vec(shape, data).unwrap()
}
fn legacy_split(arr: ArrayD<f32>, indexes: Vec<usize>) -> Vec<ArrayD<f32>> {
    // The public legacy handler also bypassed the helper for no split points.
    if indexes.is_empty() {
        return vec![arr];
    }
    // Preserve the pre-review Vec algorithm's allocation pattern, rather than
    // comparing against a simplified loop that already resembles the new one.
    let shape = arr.raw_dim();
    let batch = shape[0];
    assert!(indexes.iter().all(|&index| index <= batch));
    let stride = arr.stride_of(Axis(0)) as usize;
    let lengths = indexes
        .iter()
        .chain(std::iter::once(&batch))
        .scan(0, |last, &index| {
            let length = index - *last;
            *last = index;
            Some(length)
        })
        .collect::<Vec<_>>();
    let mut data = arr.into_raw_vec_and_offset().0;
    let mut parts = indexes
        .iter()
        .rev()
        .map(|&index| data.split_off(index * stride))
        .collect::<Vec<_>>()
        .into_iter()
        .chain(std::iter::once(data))
        .collect::<Vec<_>>();
    parts.reverse();
    parts
        .into_iter()
        .zip(lengths)
        .map(|(data, length)| {
            let mut shape = shape.clone();
            shape[0] = length;
            // SAFETY: this historical benchmark only accepts compact arrays and
            // validated sorted cuts. Each Vec has exactly the resulting shape.
            unsafe { ArrayD::from_shape_vec_unchecked(shape, data) }
        })
        .collect()
}

fn ndarray_split_values(arr: ArrayD<f32>, indexes: Vec<usize>) -> Vec<ArrayD<f32>> {
    if indexes.is_empty() {
        return vec![arr];
    }
    let batch = arr.shape()[0];
    let first_end = indexes[0];
    let mut start = first_end;
    let mut parts = Vec::with_capacity(indexes.len() + 1);
    for end in indexes.into_iter().skip(1).chain(std::iter::once(batch)) {
        parts.push(arr.slice_axis(Axis(0), Slice::from(start..end)).to_owned());
        start = end;
    }
    let mut first = arr;
    first.slice_axis_inplace(Axis(0), Slice::from(..first_end));
    parts.insert(0, first);
    parts
}

#[derive(Clone, Copy, Debug)]
enum Split {
    None,
    At(usize),
    Quarters,
    Samples,
    Repeated,
}
type SplitCase = (usize, Split);

fn split_cases() -> Vec<SplitCase> {
    (1..=32)
        .flat_map(|batch| {
            std::iter::once((batch, Split::None))
                .chain((0..=batch).map(move |cut| (batch, Split::At(cut))))
                .chain(
                    [Split::Quarters, Split::Samples, Split::Repeated].map(|split| (batch, split)),
                )
        })
        .collect()
}
fn split_indexes((batch, split): SplitCase) -> Vec<usize> {
    match split {
        Split::None => vec![],
        Split::At(cut) => vec![cut],
        Split::Quarters => cuts(batch),
        Split::Samples => (1..batch).collect(),
        Split::Repeated => vec![0, batch / 2, batch / 2, batch],
    }
}

// Check shapes, ordering, values and empty parts before timing. Small spatial
// axes make this exhaustive check cheap; the measured inputs are 224 x 224.
fn validate_split_matrix() {
    for case in split_cases() {
        let batch = case.0;
        let arr = ArrayD::from_shape_vec(
            IxDyn(&[batch, 3, 2, 2]),
            (0..batch * 12).map(|value| value as f32).collect(),
        )
        .unwrap();
        let indexes = split_indexes(case);
        let ends = indexes
            .iter()
            .copied()
            .chain(std::iter::once(batch))
            .collect::<Vec<_>>();
        for parts in [
            handler::split_batch(arr.clone(), indexes.clone()).unwrap(),
            ndarray_split_values(arr.clone(), indexes.clone()),
            legacy_split(arr.clone(), indexes.clone()),
        ] {
            assert_eq!(parts.len(), ends.len(), "{case:?}");
            let mut start = 0;
            for (part, &end) in parts.iter().zip(&ends) {
                assert_eq!(part.shape(), &[end - start, 3, 2, 2], "{case:?}");
                assert_eq!(
                    part.view(),
                    arr.slice_axis(Axis(0), Slice::from(start..end)),
                    "{case:?}"
                );
                start = end;
            }
        }
    }
}

mod split_image {
    use super::*;

    #[divan::bench(args = split_cases(), max_time = 0.1)]
    fn owned(bench: divan::Bencher, case: SplitCase) {
        bench
            .with_inputs(|| (input((case.0, true, false)), split_indexes(case)))
            .bench_values(|(arr, indexes)| handler::split_batch(arr, indexes).unwrap());
    }
    #[divan::bench(args = split_cases(), max_time = 0.1)]
    fn ndarray(bench: divan::Bencher, case: SplitCase) {
        bench
            .with_inputs(|| (input((case.0, true, false)), split_indexes(case)))
            .bench_values(|(arr, indexes)| ndarray_split_values(arr, indexes));
    }
    #[divan::bench(args = split_cases(), max_time = 0.1)]
    fn legacy_vec(bench: divan::Bencher, case: SplitCase) {
        bench
            .with_inputs(|| (input((case.0, true, false)), split_indexes(case)))
            .bench_values(|(arr, indexes)| legacy_split(arr, indexes));
    }
}

#[divan::bench(args = [(1,true,false),(2,true,false),(4,true,false),(8,true,false),(16,true,false),(32,true,false),(32,false,false),(8,true,true)], max_time = 0.1)]
fn owned_concat(bench: divan::Bencher, case: Case) {
    bench
        .with_inputs(|| concat_input(case))
        .bench_values(|arrs| handler::concat_batches(arrs).unwrap());
}
#[divan::bench(args = [(1,true,false),(2,true,false),(4,true,false),(8,true,false),(16,true,false),(32,true,false),(32,false,false),(8,true,true)], max_time = 0.1)]
fn ndarray_concat(bench: divan::Bencher, case: Case) {
    bench
        .with_inputs(|| concat_input(case))
        .bench_values(|arrs| {
            ndarray::concatenate(Axis(0), &arrs.iter().map(ArrayD::view).collect::<Vec<_>>())
                .unwrap()
        });
}
#[divan::bench(args = [(1,true,false),(2,true,false),(4,true,false),(8,true,false),(16,true,false),(32,true,false),(32,false,false)], max_time = 0.1)]
fn vec_concat(bench: divan::Bencher, case: Case) {
    bench
        .with_inputs(|| concat_input(case))
        .bench_values(legacy_concat);
}
#[divan::bench(args = [(1,true,false),(2,true,false),(4,true,false),(8,true,false),(16,true,false),(32,true,false),(32,false,false),(8,true,true)], max_time = 0.1)]
fn owned_split(bench: divan::Bencher, case: Case) {
    bench
        .with_inputs(|| (input(case), cuts(case.0)))
        .bench_values(|(arr, indexes)| handler::split_batch(arr, indexes).unwrap());
}
#[divan::bench(args = [(1,true,false),(2,true,false),(4,true,false),(8,true,false),(16,true,false),(32,true,false),(32,false,false),(8,true,true)], max_time = 0.1)]
fn ndarray_split(bench: divan::Bencher, case: Case) {
    bench
        .with_inputs(|| (input(case), cuts(case.0)))
        .bench_values(|(arr, indexes)| ndarray_split_values(arr, indexes));
}
#[divan::bench(args = [(1,true,false),(2,true,false),(4,true,false),(8,true,false),(16,true,false),(32,true,false),(32,false,false)], max_time = 0.1)]
fn vec_split(bench: divan::Bencher, case: Case) {
    bench
        .with_inputs(|| (input(case), cuts(case.0)))
        .bench_values(|(arr, indexes)| legacy_split(arr, indexes));
}
