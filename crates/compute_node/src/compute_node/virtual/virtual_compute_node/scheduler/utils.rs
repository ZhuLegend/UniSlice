use std::{collections::HashMap, hash::Hash};

pub trait IntoIteratorExt: IntoIterator {
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
}

impl<T: IntoIterator> IntoIteratorExt for T {}
