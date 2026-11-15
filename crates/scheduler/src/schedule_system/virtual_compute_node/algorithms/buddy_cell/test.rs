use compute_node::compute_device::vgpu::shareable::test::{TestResource, TestShareTarget};

use crate::schedule_system::virtual_compute_node::algorithms::buddy_cell::{
    constrained::ConstrainedResource, *,
};

#[test]
fn test_constrained_resource_creation() {
    let resource = TestResource::new(3, 40);
    let limit = TestResource::MAX_SELF;
    let constrained = ConstrainedResource::new(resource, limit);

    assert_eq!(constrained.resource(), &resource);
    assert_eq!(constrained.max_limit(), &limit);
}

#[test]
fn test_can_compose_within_limit() {
    let resource1 = ConstrainedResource::new(TestResource::new(3, 40), TestResource::MAX_SELF);
    let resource2 = TestResource::new(4, 40);

    assert!(resource1.can_compose(&resource2));
}

#[test]
fn test_can_compose_exceeds_limit() {
    let resource1 = ConstrainedResource::new(TestResource::new(4, 40), TestResource::MAX_SELF);
    let resource2 = TestResource::new(4, 40);

    // 4 + 4 = 8 > 7，应该拒绝合并
    assert!(!resource1.can_compose(&resource2));
}

#[test]
fn test_compose_within_limit() {
    let mut resource1 = ConstrainedResource::new(TestResource::new(3, 40), TestResource::MAX_SELF);
    let resource2 = TestResource::new(4, 40);

    let result = resource1.compose(resource2);
    assert!(result.is_none());
    assert_eq!(resource1.resource().g, 7);
    assert_eq!(resource1.resource().gb, 80);
}

#[test]
fn test_compose_exceeds_limit() {
    let mut resource1 = ConstrainedResource::new(TestResource::new(4, 40), TestResource::MAX_SELF);
    let resource2 = TestResource::new(4, 60);

    let result = resource1.compose(resource2);
    assert!(result.is_some());
    assert_eq!(resource1.resource(), &TestResource::new(4, 40)); // 没有改变
    assert_eq!(result.unwrap(), TestResource::new(4, 60)); // 返回原来的resource2
}

#[test]
fn test_find_optimal_combination() {
    let resource = ConstrainedResource::new(TestResource::new(0, 0), TestResource::MAX_SELF);

    let mut candidates = HashMap::new();
    candidates.insert(TestShareTarget::G1Gb10, 2);
    candidates.insert(TestShareTarget::G2Gb20, 3);
    candidates.insert(TestShareTarget::G3Gb40, 2);

    let result = resource.find_optimal_combination(&candidates);

    assert_eq!(result.len(), 2);
    assert_eq!(*result.get(&TestShareTarget::G3Gb40).unwrap(), 1);
    assert_eq!(*result.get(&TestShareTarget::G2Gb20).unwrap(), 2);
}

#[test]
fn test_is_fully_utilized() {
    let full_resource = ConstrainedResource::new(TestResource::MAX_SELF, TestResource::MAX_SELF);

    let partial_resource =
        ConstrainedResource::new(TestResource::new(1, 10), TestResource::MAX_SELF);

    assert!(full_resource.is_fully_utilized());
    assert!(!partial_resource.is_fully_utilized());
}

#[test]
fn test_split_resource() {
    let mut res = ConstrainedResource::new(TestResource::MAX_SELF, TestResource::MAX_SELF);
    let target = TestResource { g: 4, gb: 40 };
    let borrowed = res.share(&target);
    assert_eq!(borrowed, Some(TestResource { g: 4, gb: 40 }));
    assert_eq!(res.resource, TestResource { g: 3, gb: 40 });

    let target = TestResource { g: 1, gb: 10 };
    let borrowed = res.share(&target);
    assert_eq!(borrowed, Some(TestResource { g: 1, gb: 10 }));
    assert_eq!(res.resource, TestResource { g: 2, gb: 30 });
}

#[test]
fn test_compose_resource() {
    let mut res1 = ConstrainedResource::new(TestResource::new(3, 40), TestResource::MAX_SELF);
    let res2 = TestResource { g: 4, gb: 40 };
    let compose_result = res1.compose(res2);
    assert_eq!(compose_result, None);
    assert_eq!(res1.resource, TestResource { g: 7, gb: 80 });
}

#[test]
fn test_compose_invalid_resource() {
    let mut res1 = ConstrainedResource::new(TestResource::MAX_SELF, TestResource::MAX_SELF);
    let res2 = TestResource { g: 1, gb: 10 };
    let compose_result = res1.compose(res2);
    // 合并后超出最大资源，应返回Some(res2)且res1不变
    assert_eq!(compose_result, Some(TestResource { g: 1, gb: 10 }));
    assert_eq!(res1.resource, TestResource::MAX_SELF)
}

#[test]
fn test_remain_resource_share_and_refund() {
    // 初始资源：7G80Gb 和 4G40Gb
    let mut remain = RemainResource::new(vec![
        ConstrainedResource::new(TestResource::MAX_SELF, TestResource::MAX_SELF),
        ConstrainedResource::new(TestResource::new(4, 40), TestResource::MAX_SELF),
    ]);

    // 借用4G40Gb，应该优先从4G40Gb借出，剩余7G80Gb
    let borrowed = remain.share(TestShareTarget::G4Gb40);
    assert_eq!(borrowed, Some(TestResource { g: 4, gb: 40 }));
    assert_eq!(
        (&remain).into_iter().cloned().collect::<Vec<_>>(),
        vec![
            ConstrainedResource::new(TestResource::new(0, 0), TestResource::MAX_SELF),
            ConstrainedResource::new(TestResource::MAX_SELF, TestResource::MAX_SELF)
        ]
    );

    // 借用3G40Gb，应该从7G80Gb拆分，剩余4G40Gb
    let borrowed = remain.share(TestShareTarget::G3Gb40);
    assert_eq!(borrowed, Some(TestResource { g: 3, gb: 40 }));
    assert_eq!(
        (&remain).into_iter().cloned().collect::<Vec<_>>(),
        vec![
            ConstrainedResource::new(TestResource::new(0, 0), TestResource::MAX_SELF),
            ConstrainedResource::new(TestResource::new(4, 40), TestResource::MAX_SELF)
        ]
    );

    // 借用2G20Gb，应该从4G40Gb拆分，剩余2G20Gb
    let borrowed = remain.share(TestShareTarget::G2Gb20);
    assert_eq!(borrowed, Some(TestResource { g: 2, gb: 20 }));
    assert_eq!(
        (&remain).into_iter().cloned().collect::<Vec<_>>(),
        vec![
            ConstrainedResource::new(TestResource::new(0, 0), TestResource::MAX_SELF),
            ConstrainedResource::new(TestResource { g: 2, gb: 20 }, TestResource::MAX_SELF)
        ]
    );

    // 借用1G10Gb，应该从2G20Gb拆分，剩余1G10Gb
    let borrowed = remain.share(TestShareTarget::G1Gb10);
    assert_eq!(borrowed, Some(TestResource { g: 1, gb: 10 }));
    assert_eq!(
        (&remain).into_iter().cloned().collect::<Vec<_>>(),
        vec![
            ConstrainedResource::new(TestResource::new(0, 0), TestResource::MAX_SELF),
            ConstrainedResource::new(TestResource { g: 1, gb: 10 }, TestResource::MAX_SELF)
        ]
    );

    // 借用1G10Gb，正好用完
    let borrowed = remain.share(TestShareTarget::G1Gb10);
    assert_eq!(borrowed, Some(TestResource { g: 1, gb: 10 }));
    assert_eq!(
        (&remain).into_iter().cloned().collect::<Vec<_>>(),
        vec![
            ConstrainedResource::new(TestResource::new(0, 0), TestResource::MAX_SELF),
            ConstrainedResource::new(TestResource { g: 0, gb: 0 }, TestResource::MAX_SELF)
        ]
    );

    // 归还1G10Gb
    remain.refund(TestShareTarget::G1Gb10);
    assert_eq!(
        (&remain).into_iter().cloned().collect::<Vec<_>>(),
        vec![
            ConstrainedResource::new(TestResource::new(0, 0), TestResource::MAX_SELF),
            ConstrainedResource::new(TestResource { g: 1, gb: 10 }, TestResource::MAX_SELF)
        ]
    );

    // 归还2G20Gb
    remain.refund(TestShareTarget::G2Gb20);
    // 归还后应该自动合并为3G30Gb
    assert_eq!(
        (&remain).into_iter().cloned().collect::<Vec<_>>(),
        vec![
            ConstrainedResource::new(TestResource::new(0, 0), TestResource::MAX_SELF),
            ConstrainedResource::new(TestResource { g: 3, gb: 30 }, TestResource::MAX_SELF)
        ]
    );

    // 归还4G40Gb
    remain.refund(TestShareTarget::G4Gb40);
    assert_eq!(
        (&remain).into_iter().cloned().collect::<Vec<_>>(),
        vec![
            ConstrainedResource::new(TestResource::new(3, 30), TestResource::MAX_SELF),
            ConstrainedResource::new(TestResource { g: 4, gb: 40 }, TestResource::MAX_SELF)
        ]
    );
}

#[test]
fn test_refund_over_limit() {
    // 初始资源：7G80Gb
    let mut remain = RemainResource::new(vec![ConstrainedResource::new(
        TestResource::MAX_SELF,
        TestResource::MAX_SELF,
    )]);

    assert!(!remain.refund(TestShareTarget::G1Gb10)); // 额外归还1G10Gb，应该无法执行
}

#[test]
fn test_remain_resource_multi_full_resource() {
    // 初始资源：4个7G80Gb
    let mut remain = RemainResource::new(vec![
        ConstrainedResource::new(
            TestResource::MAX_SELF,
            TestResource::MAX_SELF
        );
        4
    ]);

    // 连续借用4次7G80Gb，全部借完
    for _ in 0..4 {
        let borrowed = remain.share(TestShareTarget::G7Gb80);
        assert_eq!(borrowed, Some(TestResource { g: 7, gb: 80 }));
    }
    assert!(remain.0.iter().all(|item| &item.resource
        == &ConstrainedResource::new(TestResource::new(0, 0), TestResource::MAX_SELF)));

    // 归还3个7G80Gb
    for _ in 0..3 {
        remain.refund(TestShareTarget::G7Gb80);
    }
    assert_eq!(
        (&remain).into_iter().cloned().collect::<Vec<_>>(),
        vec![
            ConstrainedResource::new(TestResource::new(0, 0), TestResource::MAX_SELF),
            ConstrainedResource::new(TestResource::MAX_SELF, TestResource::MAX_SELF),
            ConstrainedResource::new(TestResource::MAX_SELF, TestResource::MAX_SELF),
            ConstrainedResource::new(TestResource::MAX_SELF, TestResource::MAX_SELF),
        ]
    );

    // 借用1个4G40Gb，应该从一个7G80Gb拆分，剩余2个7G80Gb和1个3G40Gb
    let borrowed = remain.share(TestShareTarget::G4Gb40);
    assert_eq!(borrowed, Some(TestResource { g: 4, gb: 40 }));
    assert!(
        (&remain)
            .into_iter()
            .collect::<Vec<_>>()
            .contains(&&ConstrainedResource::new(
                TestResource { g: 3, gb: 40 },
                TestResource::MAX_SELF
            ))
    );
    assert_eq!(
        remain
            .0
            .iter()
            .filter(|r| r.resource.resource == TestResource { g: 7, gb: 80 })
            .count(),
        2
    );

    // 归还4G40Gb，应该和3G40Gb合并为7G80Gb
    remain.refund(TestShareTarget::G4Gb40);
    assert_eq!(
        remain
            .0
            .iter()
            .filter(|r| r.resource.resource == TestResource { g: 7, gb: 80 })
            .count(),
        3
    );
    assert_eq!(
        remain
            .0
            .iter()
            .filter(|item| item.resource.resource != TestResource::new(0, 0))
            .count(),
        3
    );
}

#[test]
fn test_regroup_buddy_cell_empty() {
    // 测试空列表情况
    let empty_targets: Vec<TestShareTarget> = vec![];
    let result = regroup_buddy_cell::<ConstrainedResource<TestResource>>(
        empty_targets,
        TestResource::MAX_SELF,
    );
    assert!(result.is_empty());
}

#[test]
fn test_regroup_buddy_cell_single_group() {
    // 测试可以合并的简单情况，所有资源可合并为单个组
    let targets = vec![TestShareTarget::G3Gb40, TestShareTarget::G4Gb40];

    let result =
        regroup_buddy_cell::<ConstrainedResource<TestResource>>(targets, TestResource::MAX_SELF);

    // 应该合并成一组，因为3G40GB + 4G40GB = 7G80GB是有效资源
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), 2);
    assert!(result[0].contains(&TestShareTarget::G3Gb40));
    assert!(result[0].contains(&TestShareTarget::G4Gb40));
}

#[test]
fn test_regroup_buddy_cell_two_groups() {
    // 测试多个资源分成两个组的情况
    let targets = vec![
        TestShareTarget::G1Gb10,
        TestShareTarget::G2Gb20,
        TestShareTarget::G4Gb40,
        TestShareTarget::G1Gb10,
    ];

    let result =
        regroup_buddy_cell::<ConstrainedResource<TestResource>>(targets, TestResource::MAX_SELF);

    // 4G40GB + 2G20GB + 1G10GB = 7G70GB 应合并为一组
    // 剩余 1G10GB 为另一组
    assert_eq!(result.len(), 2);

    // 找出包含4G40GB的组和其他组
    let mut g1_group = None;
    let mut other_group = None;

    for group in &result {
        if group.contains(&TestShareTarget::G1Gb10) && group.len() == 1 {
            g1_group = Some(group);
        } else {
            other_group = Some(group);
        }
    }

    // 1G10GB应该自成一组
    assert!(g1_group.is_some());
    assert_eq!(g1_group.unwrap().len(), 1);
    assert!(g1_group.unwrap().contains(&TestShareTarget::G1Gb10));

    // 其他组应包含一个1G10GB，一个2G20GB和一个4G40GB
    assert!(other_group.is_some());
    assert_eq!(other_group.unwrap().len(), 3);
    assert_eq!(
        other_group
            .unwrap()
            .iter()
            .filter(|&t| *t == TestShareTarget::G1Gb10)
            .count(),
        1
    );
    assert_eq!(
        other_group
            .unwrap()
            .iter()
            .filter(|&t| *t == TestShareTarget::G2Gb20)
            .count(),
        1
    );
    assert_eq!(
        other_group
            .unwrap()
            .iter()
            .filter(|&t| *t == TestShareTarget::G4Gb40)
            .count(),
        1
    );
}

#[test]
fn test_regroup_buddy_cell_no_grouping() {
    // 测试无法组合的情况，全部独立分组
    let targets = vec![
        TestShareTarget::G7Gb80,
        TestShareTarget::G7Gb80,
        TestShareTarget::G4Gb40,
    ];

    let result =
        regroup_buddy_cell::<ConstrainedResource<TestResource>>(targets, TestResource::MAX_SELF);

    // 无法组合，应该有3个独立的组
    assert_eq!(result.len(), 3);

    // 每组应该只包含一个元素
    for group in &result {
        assert_eq!(group.len(), 1);
    }

    // 应该有2个7G80GB组和1个4G40GB组
    assert_eq!(
        result
            .iter()
            .filter(|g| g.contains(&TestShareTarget::G7Gb80))
            .count(),
        2
    );
    assert_eq!(
        result
            .iter()
            .filter(|g| g.contains(&TestShareTarget::G4Gb40))
            .count(),
        1
    );
}

#[test]
fn test_regroup_buddy_cell_three_groups() {
    // 测试多个资源最终组合为3个组的复杂情况
    let targets = vec![
        TestShareTarget::G1Gb10, // 组1的一部分
        TestShareTarget::G1Gb10, // 组1的一部分
        TestShareTarget::G3Gb40, // 组2的一部分
        TestShareTarget::G2Gb20, // 组1的一部分
        TestShareTarget::G7Gb80, // 组3 (独立)
        TestShareTarget::G1Gb10, // 组1的一部分
        TestShareTarget::G4Gb40, // 组2的一部分
    ];

    let result =
        regroup_buddy_cell::<ConstrainedResource<TestResource>>(targets, TestResource::MAX_SELF);

    // 预期分组:
    // 组1: G1Gb10 + G1Gb10 + G1Gb10 + G2Gb20  = 5G50GB
    // 组2: G3Gb40 + G4Gb40 = 7G80GB
    // 组3: G7Gb80 (独立)
    assert_eq!(result.len(), 3);

    // 找出不同类型的组
    let mut group_5g50 = false; // 由三个G1Gb10和一个G2Gb20组成的5G50GB
    let mut group_7g80_composed = false; // 由G3Gb40和G4Gb40组成的7G80GB
    let mut group_7g80_single = false; // 单个G7Gb80

    for group in &result {
        if group.len() == 4
            && group
                .iter()
                .filter(|&t| *t == TestShareTarget::G1Gb10)
                .count()
                == 3
            && group
                .iter()
                .filter(|&t| *t == TestShareTarget::G2Gb20)
                .count()
                == 1
        {
            // 组1: G1Gb10 + G2Gb20 + G1Gb10
            group_5g50 = true;
        } else if group.len() == 2
            && group.contains(&TestShareTarget::G3Gb40)
            && group.contains(&TestShareTarget::G4Gb40)
        {
            // 组2: G3Gb40 + G4Gb40
            group_7g80_composed = true;
        } else if group.len() == 1 && group[0] == TestShareTarget::G7Gb80 {
            // 组3: 单个G7Gb80
            group_7g80_single = true;
        }
    }

    // 验证所有预期的组都存在
    assert!(group_5g50, "应存在由三个G1Gb10和一个G2Gb20组成的组");
    assert!(group_7g80_composed, "应存在由G3Gb40和G4Gb40组成的组");
    assert!(group_7g80_single, "应存在单个G7Gb80的组");
}

#[test]
#[ignore]
fn test_large_scale_regroup_performance() {
    use std::{
        sync::atomic::{AtomicUsize, Ordering},
        time::Instant,
    };

    use rand::prelude::*;
    use rayon::prelude::*;

    // 记录原始集群大小和重组后集群大小
    let original_cluster_size = 64;
    let total_iterations = 1000;
    let worse_cases = AtomicUsize::new(0);

    let start_time = Instant::now();

    // 并行执行1000次随机切分和重组测试
    (0..total_iterations).into_par_iter().for_each(|iteration| {
        let mut rng = rand::rng();

        // 初始创建1000个7G80GB资源
        let resources =
            vec![
                ConstrainedResource::new(TestResource::MAX_SELF, TestResource::MAX_SELF);
                original_cluster_size
            ];
        let mut remaining_resources = RemainResource::new(resources.clone());

        // 记录被切分出来的所有资源
        let mut split_resources: Vec<TestShareTarget> = Vec::new();

        // 随机切分资源，直到无法继续切分
        let available_targets = vec![
            TestShareTarget::G4Gb40,
            TestShareTarget::G3Gb40,
            TestShareTarget::G2Gb20,
            TestShareTarget::G1Gb10,
        ];

        let mut can_continue = true;
        while can_continue {
            // 随机选择一个目标类型进行切分
            let target = available_targets.choose(&mut rng).unwrap();

            if remaining_resources.can_share(*target) {
                if remaining_resources.share(*target).is_some() {
                    split_resources.push(*target);
                }
            } else {
                // 尝试所有可能的目标，如果都不能切分，则停止
                can_continue = false;
                for target in &available_targets {
                    if remaining_resources.can_share(*target) {
                        can_continue = true;
                        break;
                    }
                }
            }

            // 添加随机终止条件，避免无限循环
            if split_resources.len() > original_cluster_size * 7 {
                break;
            }
        }

        // 对切分出来的资源进行重组
        let regrouped = regroup_buddy_cell::<ConstrainedResource<TestResource>>(
            split_resources.clone(),
            TestResource::MAX_SELF,
        );
        let regrouped_size = regrouped.len();

        // 如果重组后的资源数量大于原始数量，记录此情况
        if regrouped_size > original_cluster_size {
            worse_cases.fetch_add(1, Ordering::SeqCst);

            // 输出第一个发现的样例的详细信息
            if worse_cases.load(Ordering::SeqCst) == 1 {
                println!("发现重组后资源增加的情况，迭代 #{iteration}");
                println!("原始资源数量: {original_cluster_size}");
                println!("切分后资源数量: {}", split_resources.len());
                println!("重组后资源数量: {regrouped_size}");
                println!("资源增加数量: {}", regrouped_size - original_cluster_size);
                println!("切分资源: {split_resources:?}");
                println!("重组资源: {regrouped:?}");

                // 计算各类资源的数量
                let mut g1_count = 0;
                let mut g2_count = 0;
                let mut g3_count = 0;
                let mut g4_count = 0;

                for target in &split_resources {
                    match target {
                        TestShareTarget::G1Gb10 => g1_count += 1,
                        TestShareTarget::G2Gb20 => g2_count += 1,
                        TestShareTarget::G3Gb40 => g3_count += 1,
                        TestShareTarget::G4Gb40 => g4_count += 1,
                        _ => (),
                    }
                }

                println!("切分资源统计：");
                println!("G1Gb10: {g1_count}");
                println!("G2Gb20: {g2_count}");
                println!("G3Gb40: {g3_count}");
                println!("G4Gb40: {g4_count}");
            }
        }
    });

    let elapsed = start_time.elapsed();
    println!("测试完成，用时: {elapsed:?}");
    println!("总迭代次数: {total_iterations}");
    println!(
        "重组后资源增加的样例数: {}",
        worse_cases.load(Ordering::SeqCst)
    );
    println!(
        "重组资源增加的比例: {:.2}%",
        (worse_cases.load(Ordering::SeqCst) as f64 / total_iterations as f64) * 100.0
    );
    assert_eq!(
        worse_cases.load(Ordering::SeqCst),
        0,
        "存在重组后资源数量增加的情况"
    );
}

#[test]
fn test_regroup_large_random_sample() {
    use std::collections::HashMap;

    use rand::Rng;

    // 创建一组随机的资源切分样本
    let mut rng = rand::rng();
    let mut targets = Vec::new();

    // 生成随机切分结果，确保包含各种资源类型
    for _ in 0..100 {
        let choice = rng.random_range(0..5);
        let target = match choice {
            0 => TestShareTarget::G1Gb10,
            1 => TestShareTarget::G2Gb20,
            2 => TestShareTarget::G3Gb40,
            3 => TestShareTarget::G4Gb40,
            _ => TestShareTarget::G7Gb80,
        };
        targets.push(target);
    }

    // 记录原始资源数量
    let counts = targets.iter().fold(HashMap::new(), |mut acc, t| {
        *acc.entry(t).or_insert(0) += 1;
        acc
    });
    println!("原始资源统计: {counts:?}");

    // 执行重组
    let regrouped =
        regroup_buddy_cell::<ConstrainedResource<TestResource>>(targets, TestResource::MAX_SELF);
    println!("重组后组数: {}", regrouped.len());

    // 验证重组结果是否有效
    for group in &regrouped {
        // 计算该组资源的总量
        let mut total_g = 0;
        let mut total_gb = 0;

        for target in group {
            let resource: TestResource = (*target).into();
            total_g += resource.g;
            total_gb += resource.gb;
        }

        // 确保资源总量不超过最大值
        assert!(
            total_g <= TestResource::MAX_G,
            "重组后的组超出最大G限制: {total_g}",
        );
        assert!(
            total_gb <= TestResource::MAX_GB,
            "重组后的组超出最大GB限制: {total_gb}",
        );
    }
}

#[cfg(feature = "nvidia")]
#[test]
fn multidimensional_availability_checks_every_remaining_device() {
    use compute_node::compute_device::vgpu::resource_spec::nv::mig::{
        A100Template as T, NvidiaMigResource as R, VDeviceScale as S,
    };

    use crate::schedule_system::virtual_compute_node::resources::nvidia_mig::ConstrainedNvidiaMigResource as C;
    let capacity = R { ce: 7, memory: 80 };
    let profiles = [T::Tiny, T::Small, T::Medium, T::Big, T::Large]
        .map(S::A100)
        .to_vec();
    let mut allocator = BuddyCellAllocator::new(
        vec![C::new_with_device_capacity(capacity, capacity); 2],
        profiles,
        vec![S::A100(T::Tiny)],
        capacity,
    );
    for t in [T::Tiny, T::Small, T::Small, T::Small, T::Big, T::Medium] {
        assert!(allocator.share(S::A100(t)).is_some());
    }
    let small = S::A100(T::Small);
    assert!(allocator.can_share(small));
    assert!(allocator.choosable_share_targets().contains(&small));
}
