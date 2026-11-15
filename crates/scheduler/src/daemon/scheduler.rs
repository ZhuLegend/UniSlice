//! 调度器模块，包含调度循环

use std::{collections::HashMap, time::Duration};

use tokio::time::{MissedTickBehavior, interval};
use tracing::{error, info};

#[cfg(all(feature = "mock", feature = "nvidia"))]
use crate::daemon::experiment_allocation::{FixedAllocationScheduler, export_allocation};
use crate::{
    SchedulableSystem, Scheduler,
    daemon::types::AppState,
    schedule_system::{
        schedulers::greedy::GreedyScheduler,
        virtual_compute_node::system::heterogeneous::system::HeterogeneousSystem,
    },
};
type Allocation = HashMap<
    uuid::Uuid,
    HashMap<
        String,
        Vec<crate::schedule_system::virtual_compute_node::system::heterogeneous::bid::Bid>,
    >,
>;

/// Experiment policies replace only global allocation; local batching stays live.
pub(crate) enum SchedulingPolicy {
    Dynamic,
    #[cfg(all(feature = "mock", feature = "nvidia"))]
    Fixed(FixedAllocationScheduler),
    #[cfg(all(feature = "mock", feature = "nvidia"))]
    FrozenGreedy,
}

struct AllocationPolicy {
    strategy: SchedulingPolicy,
    cached: Option<Allocation>,
    applied: bool,
    pending_record: Option<serde_json::Value>,
    #[cfg(all(feature = "mock", feature = "nvidia"))]
    pool_names: HashMap<uuid::Uuid, String>,
}

impl AllocationPolicy {
    fn is_static(&self) -> bool {
        !matches!(self.strategy, SchedulingPolicy::Dynamic)
    }

    fn mode(&self) -> &'static str {
        match self.strategy {
            SchedulingPolicy::Dynamic => "dynamic_greedy",
            #[cfg(all(feature = "mock", feature = "nvidia"))]
            SchedulingPolicy::Fixed(_) => "fixed",
            #[cfg(all(feature = "mock", feature = "nvidia"))]
            SchedulingPolicy::FrozenGreedy => "frozen_greedy",
        }
    }

    fn held(&self) -> bool {
        self.is_static() && self.applied
    }
}

/// 调度循环
pub async fn scheduler_loop(state: AppState, interval_secs: u64) {
    scheduler_loop_with_policy(
        state,
        interval_secs,
        SchedulingPolicy::Dynamic,
        HashMap::new(),
    )
    .await;
}

pub(crate) async fn scheduler_loop_with_policy(
    state: AppState,
    interval_secs: u64,
    strategy: SchedulingPolicy,
    _pool_names: HashMap<uuid::Uuid, String>,
) {
    let mut interval = interval(Duration::from_secs(interval_secs));
    interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let mut policy = AllocationPolicy {
        strategy,
        cached: None,
        applied: false,
        pending_record: None,
        #[cfg(all(feature = "mock", feature = "nvidia"))]
        pool_names: _pool_names,
    };

    info!("调度循环已启动，调度间隔: {} 秒", interval_secs);

    loop {
        tokio::select! {
            _ = interval.tick() => {
                // Returning the same target still reapplies the full plan in this
                // system. A held experiment must skip the complete apply path.
                if policy.held() {
                    tracing::debug!(action = "held", mode = policy.mode(), "实验资源分配保持不变");
                    continue;
                }
                tracing::debug!("开始调度周期");

                // 记录调度开始时间
                let start_time = state.scheduler_stats.record_schedule_start();

                let result = tokio::select! {
                    res = execute_scheduling_cycle(&state, &mut policy) => {res}
                    _ = tokio::signal::ctrl_c() => {break;}
                };

                // 记录调度结束时间和结果
                match result {
                    Ok(_) => {
                        // apply() has awaited every slice creation and model load.
                        if let Some(gate) = &state.simulation_startup {
                            if !gate.ready.load(std::sync::atomic::Ordering::Acquire) {
                                state.rate_predictor.arrivals.write().unwrap().clear();
                                gate.ready.store(true, std::sync::atomic::Ordering::Release);
                                state.scheduler_stats.record_schedule_end(start_time, true).await;
                                let mut started = gate.started.subscribe();
                                tokio::select! {
                                    _ = started.wait_for(|value| *value) => {},
                                    _ = tokio::signal::ctrl_c() => break,
                                }
                                // Construction time must not cause an immediate overdue tick.
                                interval.reset();
                                continue;
                            }
                        }
                        state.scheduler_stats.record_schedule_end(start_time, true).await;
                    }
                    Err(e) => {
                        state.scheduler_stats.record_schedule_end(start_time, false).await;
                        error!("调度周期执行失败: {}", e);
                    }
                }
                // The period is idle time after apply, on both success and failure.
                // An overdue interval tick must not immediately retire fresh workers.
                interval.reset();
            }
            _ = tokio::signal::ctrl_c() => {
                info!("收到中断信号，停止调度循环");
                break;
            }
        }
    }

    info!("调度循环已停止");
}

/// 执行单次调度周期
async fn execute_scheduling_cycle(
    state: &AppState,
    policy: &mut AllocationPolicy,
) -> Result<(), String> {
    let started_at_ms = crate::daemon::utils::current_timestamp();
    let start = std::time::Instant::now();
    let (mut query_ms, mut decision_ms, mut plan_ms, mut apply_ms) = (0.0, 0.0, 0.0, 0.0);
    let initial = state
        .simulation_startup
        .as_ref()
        .is_some_and(|s| !s.ready.load(std::sync::atomic::Ordering::Acquire));
    let rates = if initial {
        state
            .simulation_startup
            .as_ref()
            .unwrap()
            .initial_rates
            .clone()
    } else {
        state.rate_predictor.predict().await
    };
    let action = if policy.pending_record.is_some() {
        "persist"
    } else {
        "apply"
    };
    let result: Result<(), String> = async {
        // A successful deployment must not be repeated merely because writing
        // its evidence failed. Retry only the pending record in that case.
        if policy.pending_record.is_some() {
            return persist_experiment_record(state, policy).await;
        }
        let stage = std::time::Instant::now();
        let context = state.schedule_system.query_context().await;
        query_ms = stage.elapsed().as_secs_f64() * 1000.0;
        let stage = std::time::Instant::now();
        let scheduled = if let Some(cached) = &policy.cached {
            Ok(cached.clone())
        } else {
            match &policy.strategy {
                #[cfg(all(feature = "mock", feature = "nvidia"))]
                SchedulingPolicy::Fixed(scheduler) => scheduler.schedule(&context).await,
                SchedulingPolicy::Dynamic => GreedyScheduler.schedule(&context).await,
                #[cfg(all(feature = "mock", feature = "nvidia"))]
                SchedulingPolicy::FrozenGreedy => GreedyScheduler.schedule(&context).await,
            }
        };
        decision_ms = stage.elapsed().as_secs_f64() * 1000.0;
        let new_state = scheduled?;
        #[cfg(all(feature = "mock", feature = "nvidia"))]
        let resolved: Option<serde_json::Value> = if policy.is_static() {
            let plan = serde_json::to_value(export_allocation(&new_state, &policy.pool_names)?)
                .map_err(|e| e.to_string())?;
            // Retry this exact selection following a partial apply failure.
            policy.cached = Some(new_state.clone());
            Some(plan)
        } else {
            None
        };
        #[cfg(not(all(feature = "mock", feature = "nvidia")))]
        let resolved: Option<serde_json::Value> = None;
        let stage = std::time::Instant::now();
        let commands = HeterogeneousSystem::generate_command(&context, new_state);
        plan_ms = stage.elapsed().as_secs_f64() * 1000.0;
        let stage = std::time::Instant::now();
        let applied = state.schedule_system.execute_commands(commands).await;
        apply_ms = stage.elapsed().as_secs_f64() * 1000.0;
        applied?;
        if let Some(plan) = resolved {
            let record = serde_json::json!({
                "mode": policy.mode(), "applied": true, "plan": plan,
                "applied_at_ms": crate::daemon::utils::current_timestamp(),
            });
            policy.pending_record = Some(record);
            persist_experiment_record(state, policy).await?;
        }
        Ok(())
    }
    .await;
    let row = serde_json::json!({
        "started_at_ms": started_at_ms,
        "finished_at_ms": crate::daemon::utils::current_timestamp(),
        "initial": initial, "success": result.is_ok(), "error": result.as_ref().err(),
        "allocation_mode": policy.mode(), "action": action,
        "query_ms": query_ms, "decision_ms": decision_ms,
        "plan_ms": plan_ms, "apply_ms": apply_ms,
        "total_ms": start.elapsed().as_secs_f64() * 1000.0, "rates_at_start": rates
    });
    // Persist before publishing the successful-schedule readiness signal.
    use tokio::io::AsyncWriteExt;
    let path = state
        .scheduler_stats
        .csv_path()
        .with_file_name("schedule_events.jsonl");
    match tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .await
    {
        Ok(mut file) => {
            if let Err(err) = file.write_all(format!("{row}\n").as_bytes()).await {
                error!("Cannot write schedule timing: {err}");
            }
        }
        Err(err) => error!("Cannot open schedule timing: {err}"),
    }
    result
}

async fn persist_experiment_record(
    state: &AppState,
    policy: &mut AllocationPolicy,
) -> Result<(), String> {
    let record = policy
        .pending_record
        .as_ref()
        .ok_or("no pending allocation record")?;
    let path = state
        .scheduler_stats
        .csv_path()
        .with_file_name("experiment-allocation-resolved.json");
    let temporary = path.with_extension("json.tmp");
    tokio::fs::write(&temporary, format!("{record}\n"))
        .await
        .map_err(|e| e.to_string())?;
    tokio::fs::rename(temporary, path)
        .await
        .map_err(|e| e.to_string())?;
    policy.pending_record = None;
    policy.applied = true;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, future::Future, pin::Pin, sync::Arc};

    use super::*;
    use crate::schedule_system::virtual_compute_node::system::heterogeneous::{
        bid::Bid,
        device::{BoxedShareableDevice, ShareableResourceAllocator},
    };
    #[derive(Debug)]
    struct TestAllocator {
        bid: Bid,
        available: bool,
    }
    impl ShareableResourceAllocator for TestAllocator {
        type Target = Bid;
        fn available_targets(&self) -> Vec<Bid> {
            if self.available {
                vec![self.bid.clone()]
            } else {
                vec![]
            }
        }
        fn share(&mut self, _: &Bid) -> bool {
            if self.available {
                self.available = false;
                true
            } else {
                false
            }
        }
        fn refund(&mut self, _: &Bid) -> bool {
            self.available = true;
            true
        }
        fn min_targets(&self) -> Vec<Bid> {
            vec![self.bid.clone()]
        }
    }
    struct SlowDevice {
        id: uuid::Uuid,
        starts: tokio::sync::mpsc::UnboundedSender<tokio::time::Instant>,
    }
    impl BoxedShareableDevice for SlowDevice {
        type Target = Bid;
        fn id(&self) -> uuid::Uuid {
            self.id
        }
        fn boxed_allocator(&self) -> Box<dyn ShareableResourceAllocator<Target = Bid>> {
            Box::new(TestAllocator {
                bid: Bid::new(
                    self.id,
                    "tiny".into(),
                    Arc::new(()),
                    Arc::new(|_, batch| batch as f32 / 100.),
                    1.,
                ),
                available: true,
            })
        }
        fn apply(
            &self,
            _: HashMap<String, Vec<Bid>>,
        ) -> Pin<Box<dyn Future<Output = Result<(), String>> + Send + '_>> {
            Box::pin(async {
                self.starts.send(tokio::time::Instant::now()).unwrap();
                tokio::time::sleep(Duration::from_secs(60)).await;
                Ok(())
            })
        }
        fn support_models(&self) -> Pin<Box<dyn Future<Output = Vec<String>> + Send + '_>> {
            Box::pin(async { vec!["m".into()] })
        }
        fn compute(
            &self,
            _: compute_node::compute_node::r#virtual::virtual_compute_node::WorkTask<
                compute_node::util::Input,
            >,
        ) -> Pin<
            Box<
                dyn Future<
                        Output = Result<
                            compute_node::compute_node::r#virtual::virtual_compute_node::WorkTask<
                                compute_node::util::Output,
                            >,
                            String,
                        >,
                    > + Send
                    + '_,
            >,
        > {
            unimplemented!()
        }
        fn perform_latency_update(&self) {}
    }
    #[tokio::test(start_paused = true)]
    async fn long_apply_does_not_consume_the_next_idle_period() {
        let directory = tempfile::tempdir().unwrap();
        let (starts, mut started) = tokio::sync::mpsc::unbounded_channel();
        let id = uuid::Uuid::new_v4();
        let devices = HashMap::from([(
            id,
            Arc::new(SlowDevice { id, starts }) as Arc<dyn BoxedShareableDevice<Target = Bid>>,
        )]);
        let system = HeterogeneousSystem::new(
            devices,
            Arc::new(|| Box::pin(async { HashMap::from([("m".into(), 1.)]) })),
            1..=32,
        );
        let state = AppState {
            request_slots: Arc::new(tokio::sync::Semaphore::new(65536)),
            request_bytes: Arc::new(tokio::sync::Semaphore::new(256 * 1024 * 1024)),
            task_totals: Arc::default(),
            terminal_history: Arc::default(),
            supported_models: Arc::new(["model".to_string()].into_iter().collect()),
            simulation_startup: None,
            task_stats: Default::default(),
            schedule_system: Arc::new(system),
            rate_predictor: Default::default(),
            csv_logger: crate::daemon::csv_logger::CsvTaskLogger::new(directory.path())
                .await
                .unwrap(),
            scheduler_stats: crate::daemon::stats::SchedulerStatsManager::new(directory.path())
                .await
                .unwrap(),
        };
        let running = tokio::spawn(scheduler_loop(state, 15));
        let first = started.recv().await.unwrap();
        let second = started.recv().await.unwrap();
        running.abort();
        let _ = running.await;
        assert!(second.duration_since(first) >= Duration::from_secs(75));
    }

    #[cfg(all(feature = "mock", feature = "nvidia"))]
    struct ExperimentalDevice {
        id: uuid::Uuid,
        fail_first: bool,
        calls: Arc<std::sync::atomic::AtomicUsize>,
    }

    #[cfg(all(feature = "mock", feature = "nvidia"))]
    impl BoxedShareableDevice for ExperimentalDevice {
        type Target = Bid;
        fn id(&self) -> uuid::Uuid {
            self.id
        }
        fn boxed_allocator(&self) -> Box<dyn ShareableResourceAllocator<Target = Bid>> {
            use compute_node::compute_device::vgpu::resource_spec::nv::mig::{
                A100Template, VDeviceScale,
            };
            Box::new(TestAllocator {
                bid: Bid::new(
                    self.id,
                    "tiny".into(),
                    Arc::new(VDeviceScale::A100(A100Template::Tiny)),
                    Arc::new(|_, batch| batch as f32 / 100.),
                    1.,
                ),
                available: true,
            })
        }
        fn apply(
            &self,
            _: HashMap<String, Vec<Bid>>,
        ) -> Pin<Box<dyn Future<Output = Result<(), String>> + Send + '_>> {
            Box::pin(async move {
                let previous = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if self.fail_first && previous == 0 {
                    Err("injected apply failure".into())
                } else {
                    Ok(())
                }
            })
        }
        fn support_models(&self) -> Pin<Box<dyn Future<Output = Vec<String>> + Send + '_>> {
            Box::pin(async { vec!["m".into()] })
        }
        fn compute(
            &self,
            _: compute_node::compute_node::r#virtual::virtual_compute_node::WorkTask<
                compute_node::util::Input,
            >,
        ) -> Pin<
            Box<
                dyn Future<
                        Output = Result<
                            compute_node::compute_node::r#virtual::virtual_compute_node::WorkTask<
                                compute_node::util::Output,
                            >,
                            String,
                        >,
                    > + Send
                    + '_,
            >,
        > {
            unimplemented!()
        }
        fn perform_latency_update(&self) {}
    }

    #[cfg(all(feature = "mock", feature = "nvidia"))]
    async fn experiment_state(
        directory: &std::path::Path,
        fail_first: bool,
    ) -> (
        AppState,
        Arc<std::sync::atomic::AtomicUsize>,
        HashMap<uuid::Uuid, String>,
    ) {
        let id = uuid::Uuid::new_v4();
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let device = Arc::new(ExperimentalDevice {
            id,
            fail_first,
            calls: calls.clone(),
        });
        let system = HeterogeneousSystem::new(
            HashMap::from([(id, device as Arc<dyn BoxedShareableDevice<Target = Bid>>)]),
            Arc::new(|| Box::pin(async { HashMap::from([("m".into(), 1.)]) })),
            1..=32,
        );
        let state = AppState {
            request_slots: Arc::new(tokio::sync::Semaphore::new(65536)),
            request_bytes: Arc::new(tokio::sync::Semaphore::new(256 * 1024 * 1024)),
            task_totals: Arc::default(),
            terminal_history: Arc::default(),
            supported_models: Arc::new(["m".to_string()].into_iter().collect()),
            simulation_startup: None,
            task_stats: Default::default(),
            schedule_system: Arc::new(system),
            rate_predictor: Default::default(),
            csv_logger: crate::daemon::csv_logger::CsvTaskLogger::new(directory)
                .await
                .unwrap(),
            scheduler_stats: crate::daemon::stats::SchedulerStatsManager::new(directory)
                .await
                .unwrap(),
        };
        (state, calls, HashMap::from([(id, "parent_full".into())]))
    }

    #[cfg(all(feature = "mock", feature = "nvidia"))]
    #[tokio::test]
    async fn experiment_retry_keeps_cached_plan_and_holds_only_after_success() {
        let directory = tempfile::tempdir().unwrap();
        let (state, calls, pool_names) = experiment_state(directory.path(), true).await;
        let mut policy = AllocationPolicy {
            strategy: SchedulingPolicy::FrozenGreedy,
            cached: None,
            applied: false,
            pending_record: None,
            pool_names,
        };
        assert!(execute_scheduling_cycle(&state, &mut policy).await.is_err());
        assert!(!policy.held());
        let original =
            export_allocation(policy.cached.as_ref().unwrap(), &policy.pool_names).unwrap();
        assert!(
            !directory
                .path()
                .join("experiment-allocation-resolved.json")
                .exists()
        );
        execute_scheduling_cycle(&state, &mut policy).await.unwrap();
        assert!(policy.held());
        let current =
            export_allocation(policy.cached.as_ref().unwrap(), &policy.pool_names).unwrap();
        assert_eq!(
            serde_json::to_value(original).unwrap(),
            serde_json::to_value(current).unwrap()
        );
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
        let resolved: serde_json::Value = serde_json::from_slice(
            &tokio::fs::read(directory.path().join("experiment-allocation-resolved.json"))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(resolved["applied"], true);
        assert_eq!(resolved["mode"], "frozen_greedy");
    }

    #[cfg(all(feature = "mock", feature = "nvidia"))]
    #[tokio::test]
    async fn experiment_record_failure_retries_persistence_without_reapplying() {
        let directory = tempfile::tempdir().unwrap();
        let (state, calls, pool_names) = experiment_state(directory.path(), false).await;
        let resolved_path = directory.path().join("experiment-allocation-resolved.json");
        tokio::fs::create_dir(&resolved_path).await.unwrap();
        let mut policy = AllocationPolicy {
            strategy: SchedulingPolicy::FrozenGreedy,
            cached: None,
            applied: false,
            pending_record: None,
            pool_names,
        };
        assert!(execute_scheduling_cycle(&state, &mut policy).await.is_err());
        assert!(!policy.held());
        assert!(policy.pending_record.is_some());
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        tokio::fs::remove_dir(&resolved_path).await.unwrap();
        execute_scheduling_cycle(&state, &mut policy).await.unwrap();
        assert!(policy.held());
        assert!(policy.pending_record.is_none());
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[cfg(all(feature = "mock", feature = "nvidia"))]
    #[tokio::test]
    async fn experiment_loop_does_not_reapply_or_increment_schedules_when_held() {
        let directory = tempfile::tempdir().unwrap();
        let (state, calls, pool_names) = experiment_state(directory.path(), false).await;
        let stats = state.scheduler_stats.clone();
        let running = tokio::spawn(scheduler_loop_with_policy(
            state,
            1,
            SchedulingPolicy::FrozenGreedy,
            pool_names,
        ));
        tokio::time::timeout(Duration::from_secs(5), async {
            while !directory
                .path()
                .join("experiment-allocation-resolved.json")
                .exists()
            {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(2200)).await;
        running.abort();
        let _ = running.await;
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(stats.get_stats().successful_schedules, 1);
    }
}
