//! GPU 资源最优组合搜索算法
//!
//! 提供用于 GPU 资源分配的递归搜索算法，可以被不同的 GPU 资源类型复用

use std::collections::HashMap;

/// GPU 资源组合搜索器
pub struct GpuResourceCombinationSearcher;

impl GpuResourceCombinationSearcher {
    /// 搜索一维资源的最优组合（如 MPS 百分比）
    ///
    /// # 参数
    /// - `need`: 需要的资源量
    /// - `candidates`: 候选资源映射 (资源大小 -> 可用数量)
    /// - `max_depth`: 最大搜索深度
    ///
    /// # 返回值
    /// 返回最优的资源组合
    pub fn search_1d_combination(
        need: u32,
        candidates: &HashMap<u32, usize>,
        max_depth: usize,
    ) -> HashMap<u32, usize> {
        if need == 0 {
            return HashMap::new();
        }

        // 按资源大小从大到小排序
        let mut targets: Vec<(u32, usize)> = candidates
            .iter()
            .map(|(&size, &count)| (size, count))
            .collect();
        targets.sort_unstable_by(|a, b| b.0.cmp(&a.0));

        let mut used = HashMap::new();
        if let Some(result) = Self::search_1d_recursive(&targets, need, max_depth, &mut used) {
            return result;
        }

        HashMap::new()
    }

    /// 搜索二维资源的最优组合（如 MIG 的 CE + Memory）
    ///
    /// # 参数
    /// - `need_dim1`: 第一维度需要的资源量
    /// - `need_dim2`: 第二维度需要的资源量
    /// - `candidates`: 候选资源映射 (资源类型 -> (第一维度大小, 第二维度大小, 可用数量))
    /// - `max_depth`: 最大搜索深度
    ///
    /// # 返回值
    /// 返回最优的资源组合
    pub fn search_2d_combination<T>(
        need_dim1: u32,
        need_dim2: u32,
        candidates: &HashMap<T, (u32, u32, usize)>,
        max_depth: usize,
    ) -> HashMap<T, usize>
    where
        T: Clone + Copy + PartialEq + Eq + std::hash::Hash,
    {
        if need_dim1 == 0 && need_dim2 == 0 {
            return HashMap::new();
        }

        // 按资源大小从大到小排序
        let mut targets: Vec<(T, u32, u32, usize)> = candidates
            .iter()
            .map(|(&resource_type, &(dim1, dim2, count))| (resource_type, dim1, dim2, count))
            .collect();
        targets.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(b.2.cmp(&a.2)));

        let mut used = HashMap::new();
        if let Some(result) =
            Self::search_2d_recursive(&targets, need_dim1, need_dim2, max_depth, &mut used)
        {
            return result;
        }

        HashMap::new()
    }

    /// 搜索三维资源的最优组合（如华为 MIG 的 AI Core + AI CPU + Memory）
    ///
    /// # 参数
    /// - `need_dim1`: 第一维度需要的资源量
    /// - `need_dim2`: 第二维度需要的资源量
    /// - `need_dim3`: 第三维度需要的资源量
    /// - `candidates`: 候选资源映射 (资源类型 -> (第一维度大小, 第二维度大小, 第三维度大小, 可用数量))
    /// - `max_depth`: 最大搜索深度
    ///
    /// # 返回值
    /// 返回最优的资源组合
    pub fn search_3d_combination<T>(
        need_dim1: u32,
        need_dim2: u32,
        need_dim3: u32,
        candidates: &HashMap<T, (u32, u32, u32, usize)>,
        max_depth: usize,
    ) -> HashMap<T, usize>
    where
        T: Clone + Copy + PartialEq + Eq + std::hash::Hash,
    {
        if need_dim1 == 0 && need_dim2 == 0 && need_dim3 == 0 {
            return HashMap::new();
        }

        // 按资源大小从大到小排序
        let mut targets: Vec<(T, u32, u32, u32, usize)> = candidates
            .iter()
            .map(|(&resource_type, &(dim1, dim2, dim3, count))| {
                (resource_type, dim1, dim2, dim3, count)
            })
            .collect();
        targets.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(b.2.cmp(&a.2)).then(b.3.cmp(&a.3)));

        let mut used = HashMap::new();
        if let Some(result) = Self::search_3d_recursive(
            &targets, need_dim1, need_dim2, need_dim3, max_depth, &mut used,
        ) {
            return result;
        }

        HashMap::new()
    }

    /// 一维资源的递归搜索实现
    fn search_1d_recursive(
        targets: &[(u32, usize)],
        need: u32,
        depth: usize,
        used: &mut HashMap<u32, usize>,
    ) -> Option<HashMap<u32, usize>> {
        if need == 0 {
            return Some(used.clone());
        }
        if depth == 0 || targets.is_empty() {
            return None;
        }

        for (i, &(size, count)) in targets.iter().enumerate() {
            if size > need || count == 0 {
                continue;
            }

            let max_take = (need / size).min(count as u32).min(depth as u32) as usize;

            for take in 1..=max_take {
                *used.entry(size).or_insert(0) += take;
                let left_need = need - size * take as u32;

                let mut next_targets = targets.to_vec();
                next_targets[i].1 -= take;

                let found = Self::search_1d_recursive(&next_targets, left_need, depth - take, used);
                if found.is_some() {
                    return found;
                }

                *used.get_mut(&size).unwrap() -= take;
                if used[&size] == 0 {
                    used.remove(&size);
                }
            }
        }
        None
    }

    /// 二维资源的递归搜索实现
    fn search_2d_recursive<T>(
        targets: &[(T, u32, u32, usize)],
        need_dim1: u32,
        need_dim2: u32,
        depth: usize,
        used: &mut HashMap<T, usize>,
    ) -> Option<HashMap<T, usize>>
    where
        T: Clone + Copy + PartialEq + Eq + std::hash::Hash,
    {
        if need_dim1 == 0 && need_dim2 == 0 {
            return Some(used.clone());
        }
        if depth == 0 || targets.is_empty() {
            return None;
        }

        for (i, &(resource_type, dim1, dim2, count)) in targets.iter().enumerate() {
            if dim1 > need_dim1 || dim2 > need_dim2 || count == 0 {
                continue;
            }

            let max_by_dim1 = if dim1 > 0 { need_dim1 / dim1 } else { u32::MAX };
            let max_by_dim2 = if dim2 > 0 { need_dim2 / dim2 } else { u32::MAX };
            let max_take = max_by_dim1
                .min(max_by_dim2)
                .min(count as u32)
                .min(depth as u32) as usize;

            for take in 1..=max_take {
                *used.entry(resource_type).or_insert(0) += take;
                let left_dim1 = need_dim1 - dim1 * take as u32;
                let left_dim2 = need_dim2 - dim2 * take as u32;

                let mut next_targets = targets.to_vec();
                next_targets[i].3 -= take;

                let found = Self::search_2d_recursive(
                    &next_targets,
                    left_dim1,
                    left_dim2,
                    depth - take,
                    used,
                );
                if found.is_some() {
                    return found;
                }

                *used.get_mut(&resource_type).unwrap() -= take;
                if used[&resource_type] == 0 {
                    used.remove(&resource_type);
                }
            }
        }
        None
    }

    /// 三维资源的递归搜索实现
    fn search_3d_recursive<T>(
        targets: &[(T, u32, u32, u32, usize)],
        need_dim1: u32,
        need_dim2: u32,
        need_dim3: u32,
        depth: usize,
        used: &mut HashMap<T, usize>,
    ) -> Option<HashMap<T, usize>>
    where
        T: Clone + Copy + PartialEq + Eq + std::hash::Hash,
    {
        if need_dim1 == 0 && need_dim2 == 0 && need_dim3 == 0 {
            return Some(used.clone());
        }
        if depth == 0 || targets.is_empty() {
            return None;
        }

        for (i, &(resource_type, dim1, dim2, dim3, count)) in targets.iter().enumerate() {
            if dim1 > need_dim1 || dim2 > need_dim2 || dim3 > need_dim3 || count == 0 {
                continue;
            }

            let max_by_dim1 = if dim1 > 0 { need_dim1 / dim1 } else { u32::MAX };
            let max_by_dim2 = if dim2 > 0 { need_dim2 / dim2 } else { u32::MAX };
            let max_by_dim3 = if dim3 > 0 { need_dim3 / dim3 } else { u32::MAX };
            let max_take = max_by_dim1
                .min(max_by_dim2)
                .min(max_by_dim3)
                .min(count as u32)
                .min(depth as u32) as usize;

            for take in 1..=max_take {
                *used.entry(resource_type).or_insert(0) += take;
                let left_dim1 = need_dim1 - dim1 * take as u32;
                let left_dim2 = need_dim2 - dim2 * take as u32;
                let left_dim3 = need_dim3 - dim3 * take as u32;

                let mut next_targets = targets.to_vec();
                next_targets[i].4 -= take;

                let found = Self::search_3d_recursive(
                    &next_targets,
                    left_dim1,
                    left_dim2,
                    left_dim3,
                    depth - take,
                    used,
                );
                if found.is_some() {
                    return found;
                }

                *used.get_mut(&resource_type).unwrap() -= take;
                if used[&resource_type] == 0 {
                    used.remove(&resource_type);
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1d_combination() {
        let mut candidates = HashMap::new();
        candidates.insert(50u32, 1);
        candidates.insert(25u32, 2);
        candidates.insert(10u32, 5);

        let result = GpuResourceCombinationSearcher::search_1d_combination(100, &candidates, 10);

        let total: u32 = result
            .iter()
            .map(|(&size, &count)| size * count as u32)
            .sum();

        assert_eq!(total, 100);
        assert!(result.contains_key(&50u32));
    }

    #[test]
    fn test_2d_combination() {
        let mut candidates = HashMap::new();
        candidates.insert("large", (7u32, 80u32, 1usize));
        candidates.insert("medium", (4u32, 40u32, 2usize));
        candidates.insert("small", (1u32, 10u32, 5usize));

        let result = GpuResourceCombinationSearcher::search_2d_combination(7, 80, &candidates, 10);

        assert!(result.contains_key(&"large"));
        assert_eq!(result[&"large"], 1);
    }

    #[test]
    fn test_3d_combination() {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        enum HuaweiTarget {
            Large,
            Medium,
            Small,
        }

        let mut candidates = HashMap::new();
        candidates.insert(HuaweiTarget::Large, (24u32, 7u32, 64u32, 1usize));
        candidates.insert(HuaweiTarget::Medium, (12u32, 3u32, 32u32, 2usize));
        candidates.insert(HuaweiTarget::Small, (6u32, 1u32, 16u32, 5usize));

        let result =
            GpuResourceCombinationSearcher::search_3d_combination(24, 7, 64, &candidates, 10);

        assert!(result.contains_key(&HuaweiTarget::Large));
        assert_eq!(result[&HuaweiTarget::Large], 1);
    }
}
