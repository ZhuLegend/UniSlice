//! Virtual Compute Node's Scheduler implementation

/// Scheduling algorithms.
pub mod algorithms;

/// GPU resource adapters for scheduling.
pub mod resources;

/// System models and components.
pub mod system;

use std::{collections::HashMap, hash::Hash};

/// [WellSortedNodes] is a utility struct that holds a vector of vectors of share parameters,
/// which guarantees that all the inner vectors are sorted.
///
/// # Example
/// ```no_compile
/// let nodes = vec![vec![1, 2], vec![4, 3], vec![2, 1]];
/// let well_sorted_nodes = WellSortedNodes::new(nodes);
/// assert_eq!(well_sorted_nodes.0, vec![vec![1, 2], vec![3, 4], vec![1, 2]]);
/// ```
struct WellSortedNodes<ShareParam: Ord>(Vec<Vec<ShareParam>>);

impl<ShareParam: Ord> WellSortedNodes<ShareParam> {
    fn new(nodes: impl IntoIterator<Item = Vec<ShareParam>>) -> Self {
        let sorted_nodes = nodes
            .into_iter()
            .map(|mut group| {
                group.sort_unstable();
                group
            })
            .collect::<Vec<_>>();
        WellSortedNodes(sorted_nodes)
    }

    fn as_group_map(&self) -> HashMap<Vec<ShareParam>, usize>
    where
        ShareParam: Hash + Clone,
    {
        self.0.iter().fold(HashMap::new(), |mut acc, group| {
            *acc.entry(group.clone()).or_insert(0) += 1;
            acc
        })
    }

    fn get_same_group_count(
        map1: &mut HashMap<Vec<ShareParam>, usize>,
        map2: &mut HashMap<Vec<ShareParam>, usize>,
    ) -> HashMap<Vec<ShareParam>, usize>
    where
        ShareParam: Hash + Clone,
    {
        let mut preserved = HashMap::new();
        for (item, count) in map1.iter_mut() {
            if let Some(other_count) = map2.get_mut(item) {
                let min_count = *count.min(other_count);
                if min_count > 0 {
                    preserved.insert(item.clone(), min_count);
                }
            }
        }
        for (item, count) in &preserved {
            map1.entry(item.clone()).and_modify(|c| *c -= count);
            map2.entry(item.clone()).and_modify(|c| *c -= count);
        }
        preserved
    }

    /// Extracts the same nodes from `self` and `other`, returning a new instance which contains these same nodes.
    /// This method modifies `self` and `other` in place, removing the same nodes from both.
    fn extract_same_nodes(&mut self, other: &mut Self) -> Self
    where
        ShareParam: Hash + Clone,
    {
        let mut map1 = self.as_group_map();
        let mut map2 = other.as_group_map();

        let preserved = Self::get_same_group_count(&mut map1, &mut map2);

        self.0.retain(|group| {
            if let Some(count) = map1.get_mut(group)
                && *count > 0
            {
                *count -= 1;
                return true;
            }
            false
        });

        other.0.retain(|group| {
            if let Some(count) = map2.get_mut(group)
                && *count > 0
            {
                *count -= 1;
                return true;
            }
            false
        });

        Self::new(
            preserved
                .into_iter()
                .fold(Vec::new(), |mut acc, (item, count)| {
                    for _ in 0..count {
                        acc.push(item.clone());
                    }
                    acc
                }),
        )
    }
}

trait IntoIteratorExt: IntoIterator {
    fn into_map_count(self) -> HashMap<Self::Item, usize>
    where
        Self::Item: Hash + Eq,
        Self: Sized,
    {
        self.into_iter().fold(HashMap::new(), |mut acc, item| {
            *acc.entry(item).or_insert(0) += 1;
            acc
        })
    }

    fn into_map_count_fn<F, K: Hash + Eq>(self, key_fn: F) -> HashMap<K, usize>
    where
        F: Fn(Self::Item) -> K,
        Self: Sized,
    {
        self.into_iter().fold(HashMap::new(), |mut acc, item| {
            *acc.entry(key_fn(item)).or_insert(0) += 1;
            acc
        })
    }

    fn into_map_vec<K: Hash + Eq, V>(self) -> HashMap<K, Vec<V>>
    where
        Self::Item: Into<(K, V)>,
        Self: Sized,
    {
        self.into_iter()
            .map(Into::into)
            .fold(HashMap::new(), |mut acc, (key, value)| {
                acc.entry(key).or_default().push(value);
                acc
            })
    }

    fn into_map_vec_fn<K: Hash + Eq, V, F>(self, key_fn: F) -> HashMap<K, Vec<V>>
    where
        F: Fn(Self::Item) -> (K, V),
        Self: Sized,
    {
        self.into_iter().map(key_fn).into_map_vec()
    }
}

impl<T: IntoIterator> IntoIteratorExt for T {}
