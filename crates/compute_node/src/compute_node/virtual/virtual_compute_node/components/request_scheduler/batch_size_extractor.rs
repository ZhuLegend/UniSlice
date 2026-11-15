//! A [BatchExtractorStrategy] define how to satisfy the required batch size

use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
use tokio::{
    join,
    sync::{oneshot, oneshot::Sender},
};

use crate::{
    compute_node::r#virtual::{
        BaseWork,
        virtual_compute_node::components::{
            request_scheduler::model_queue::{RequestEntry, Requests},
            worker_manager::BatchSize,
        },
    },
    util::{Input, Output},
};

/// Trait defining how to extract a batch of tasks from a queue.
pub(super) trait BatchSizeExtractor: Send + Sync {
    /// Determines whether a batch should be extracted at this moment.
    ///
    /// # Arguments
    /// * `desired_batch_size` - The batch size requested by the worker.
    /// * `model_reqs` - The current queue of pending requests for the model.
    ///
    /// # Returns
    /// * `true` if it's an opportune time to extract a batch.
    fn should_extract(&self, desired_batch_size: BatchSize, model_reqs: &Requests) -> bool {
        model_reqs.total_batch_size() >= desired_batch_size
    }

    /// Extracts a batch of tasks to satisfy the desired batch size.
    fn extract_batch(
        &self,
        desired_batch_size: BatchSize,
        model_reqs: &Requests,
    ) -> Vec<(BaseWork<Input>, Sender<BaseWork<Output>>)>;
}

// --- Implementation 1: Split and Satisfy ---

/// An extractor that satisfies the exact batch size, splitting tasks if necessary.
#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct SplitAndSatisfyBatchSizeExtractor;

impl BatchSizeExtractor for SplitAndSatisfyBatchSizeExtractor {
    #[inline]
    fn extract_batch(
        &self,
        desired_batch_size: BatchSize,
        model_reqs: &Requests,
    ) -> Vec<(BaseWork<Input>, Sender<BaseWork<Output>>)> {
        let (entry_batch_size, mut entries) = model_reqs.fetch_target(desired_batch_size);

        if entry_batch_size > desired_batch_size {
            let last_entry = entries.pop().unwrap();
            // The task will exceed, so it needs to be split
            let remaining_batch_size = entry_batch_size - desired_batch_size;
            let needed_batch_size = last_entry.batch_size - remaining_batch_size;

            match self.split_task_by_batch_size(last_entry, needed_batch_size, remaining_batch_size)
            {
                Ok((split_entry, remaining_entry)) => {
                    entries.push(split_entry);

                    model_reqs.push_front(remaining_entry);
                }
                Err(error) => {
                    // Dropping the failed request closes its response channel.
                    tracing::warn!(
                        "Failed to split task: needed={}, remaining={}, error={}",
                        needed_batch_size,
                        remaining_batch_size,
                        error
                    );
                }
            }
        }

        entries
            .into_iter()
            .map(|e| (e.work, e.response_tx))
            .collect()
    }
}

impl SplitAndSatisfyBatchSizeExtractor {
    /// Splits a task by batch size.
    fn split_task_by_batch_size(
        &self,
        entry: RequestEntry,
        needed_batch_size: usize,
        remaining_batch_size: usize,
    ) -> Result<(RequestEntry, RequestEntry), String> {
        // Pre-check if the task can be split
        let work_dimensions = entry.work.data().dimensions();
        if work_dimensions.is_empty() {
            return Err("Cannot split a scalar work item".into());
        }

        // Validate if the split point is valid
        if needed_batch_size >= entry.batch_size {
            return Err("Split point must precede the batch end".into());
        }

        let (split_work, remaining_work) = entry.work.split_at_batch(needed_batch_size)?;

        // Create channels for merging the results
        let (result_tx, result_rx) = oneshot::channel::<BaseWork<Output>>();
        let (remaining_tx, remaining_rx) = oneshot::channel::<BaseWork<Output>>();

        // Save the original response_tx
        let mut original_response_tx = entry.response_tx;

        // Create a result merging task
        tokio::spawn(async move {
            let mut final_result = None;

            // Wait for both parts of the result
            let results = tokio::select! {
                results = async { join!(result_rx, remaining_rx) } => results,
                _ = original_response_tx.closed() => return,
            };
            match results {
                (Ok(split_result), Ok(remaining_result)) => {
                    // Merge the results
                    match split_result.merge_with(remaining_result) {
                        Ok(merged) => {
                            final_result = Some(merged);
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Failed to merge split task results: batch_size={}, needed={}, remaining={}, error={}",
                                needed_batch_size + remaining_batch_size,
                                needed_batch_size,
                                remaining_batch_size,
                                e
                            );
                        }
                    }
                }
                (Err(e), _) => {
                    tracing::warn!(
                        ?e,
                        "Failed to receive split task result: needed_batch_size={}",
                        needed_batch_size
                    );
                }
                (_, Err(e)) => {
                    tracing::warn!(
                        ?e,
                        "Failed to receive remaining task result: remaining_batch_size={}",
                        remaining_batch_size
                    );
                }
            }

            // Send the final result
            if let Some(result) = final_result {
                let _ = original_response_tx.send(result);
            } else {
                tracing::warn!(
                    "No merged result to send back to original requester: needed_batch_size={}, remaining_batch_size={}",
                    needed_batch_size,
                    remaining_batch_size
                );
            }
        });

        let split_entry = RequestEntry {
            work: split_work,
            response_tx: result_tx,
            batch_size: needed_batch_size,
        };

        let remaining_entry = RequestEntry {
            work: remaining_work,
            response_tx: remaining_tx,
            batch_size: remaining_batch_size,
        };

        Ok((split_entry, remaining_entry))
    }
}

// --- Implementation 1.5: Timeout Split and Satisfy ---

/// An extractor that extends `SplitAndSatisfy` with a timeout mechanism.
///
/// This strategy ensures that even if the desired batch size is not met, tasks
/// will not be indefinitely starved. If the time since the last extraction
/// exceeds a specified timeout and the queue is not empty, a batch will be formed.
#[derive(Debug)]
#[allow(private_bounds)]
pub struct TimeoutBatchSizeExtractor<Inner: BatchSizeExtractor> {
    inner: Inner,
    timeout: Duration,
    last_extraction_time: Arc<Mutex<Option<Instant>>>,
}

// A cloned extractor is a new runtime instance. Share an Arc explicitly when
// repeated scans must use the same model's clock.
impl<Inner: BatchSizeExtractor + Clone> Clone for TimeoutBatchSizeExtractor<Inner> {
    fn clone(&self) -> Self {
        Self::new(self.timeout, self.inner.clone())
    }
}

impl<Inner: BatchSizeExtractor + Serialize + Clone> Serialize for TimeoutBatchSizeExtractor<Inner> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[derive(Serialize)]
        struct Helper<Inner> {
            timeout: Duration,
            inner: Inner,
        }

        let helper = Helper {
            timeout: self.timeout,
            inner: self.inner.clone(),
        };
        helper.serialize(serializer)
    }
}

impl<'de, Inner: BatchSizeExtractor + Deserialize<'de>> Deserialize<'de>
    for TimeoutBatchSizeExtractor<Inner>
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Helper<Inner> {
            timeout: Duration,
            inner: Inner,
        }

        let helper = Helper::deserialize(deserializer)?;
        Ok(Self::new(helper.timeout, helper.inner))
    }
}

#[allow(private_bounds)]
impl<Inner: BatchSizeExtractor> TimeoutBatchSizeExtractor<Inner> {
    /// Creates a new `TimeoutBatchSizeExtractor`.
    pub fn new(timeout: Duration, inner: Inner) -> Self {
        Self {
            inner,
            timeout,
            last_extraction_time: Arc::new(Mutex::new(None)),
        }
    }
}

impl<Inner: BatchSizeExtractor> BatchSizeExtractor for TimeoutBatchSizeExtractor<Inner> {
    fn should_extract(&self, desired_batch_size: BatchSize, model_reqs: &Requests) -> bool {
        let mut last_time = self.last_extraction_time.lock().expect(
            "Mutex poisoned while locking last_extraction_time in should_extract (timeout strategy)",
        );
        // Condition 1: The inner strategy says we should extract.
        if self.inner.should_extract(desired_batch_size, model_reqs) {
            *last_time = Some(Instant::now());
            return true;
        }

        // Condition 2: The timeout has been exceeded and the queue is not empty.
        if let Some(last) = *last_time {
            if !model_reqs.is_empty() && last.elapsed() >= self.timeout {
                *last_time = Some(Instant::now());
                return true;
            }
        } else {
            // If time was never set, set it now.
            *last_time = Some(Instant::now());
        }

        false
    }

    #[inline]
    fn extract_batch(
        &self,
        desired_batch_size: BatchSize,
        model_reqs: &Requests,
    ) -> Vec<(BaseWork<Input>, Sender<BaseWork<Output>>)> {
        // Delegate to the inner extractor.
        self.inner.extract_batch(desired_batch_size, model_reqs)
    }
}

/// An extractor that wraps another extractor and adds a probability-based extraction.
///
/// This extractor will always extract if the inner extractor decides to do so.
/// If the inner extractor does not extract, this extractor will extract with a probability
/// proportional to the ratio of the current total batch size to the desired batch size.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(private_bounds)]
pub struct ProbabilityBatchSizeExtractor<Inner: BatchSizeExtractor> {
    threshold: f64,
    inner: Inner,
    #[serde(skip)]
    random_stream: Option<Arc<crate::experiment_random::SeededStream>>,
}

impl<Inner: BatchSizeExtractor> BatchSizeExtractor for ProbabilityBatchSizeExtractor<Inner> {
    fn should_extract(&self, desired_batch_size: BatchSize, model_reqs: &Requests) -> bool {
        if self.inner.should_extract(desired_batch_size, model_reqs) {
            true
        } else {
            let total_batch_size = model_reqs.total_batch_size();
            if (total_batch_size as f64 / desired_batch_size as f64) >= self.threshold {
                let draw = self
                    .random_stream
                    .as_ref()
                    .map_or_else(rand::random::<u64>, |stream| stream.next_u64());
                let random_value = draw % (desired_batch_size as u64) + 1;
                random_value <= total_batch_size as u64
            } else {
                false
            }
        }
    }

    fn extract_batch(
        &self,
        desired_batch_size: BatchSize,
        model_reqs: &Requests,
    ) -> Vec<(BaseWork<Input>, Sender<BaseWork<Output>>)> {
        self.inner.extract_batch(desired_batch_size, model_reqs)
    }
}

// --- Implementation 2: Greedy ---

/// An extractor that collects tasks until the batch size is met or exceeded, without splitting.
#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct BestEffortBatchSizeExtractor;

impl BatchSizeExtractor for BestEffortBatchSizeExtractor {
    fn should_extract(&self, _desired_batch_size: BatchSize, model_reqs: &Requests) -> bool {
        !model_reqs.is_empty()
    }

    #[inline]
    fn extract_batch(
        &self,
        desired_batch_size: BatchSize,
        model_reqs: &Requests,
    ) -> Vec<(BaseWork<Input>, Sender<BaseWork<Output>>)> {
        model_reqs
            .fetch_target(desired_batch_size)
            .1
            .into_iter()
            .map(|e| (e.work, e.response_tx))
            .collect()
    }
}

// --- Implementation 3: Ignored ---

/// An extractor that ignores the required batch and just returns the first work in the queue.
#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct IgnoredBatchSizeExtractor;

impl BatchSizeExtractor for IgnoredBatchSizeExtractor {
    #[inline]
    fn should_extract(&self, _desired_batch_size: BatchSize, model_reqs: &Requests) -> bool {
        !model_reqs.is_empty()
    }

    #[inline]
    fn extract_batch(
        &self,
        _desired_batch_size: BatchSize,
        model_reqs: &Requests,
    ) -> Vec<(BaseWork<Input>, Sender<BaseWork<Output>>)> {
        model_reqs
            .fetch_target(1)
            .1
            .into_iter()
            .map(|e| (e.work, e.response_tx))
            .collect()
    }
}

// --- Strategy Enum ---

/// Enum wrapper for different batch size extraction strategies
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum BatchExtractorStrategy {
    /// Split batches to exactly satisfy the desired size when necessary
    SplitAndSatisfy(SplitAndSatisfyBatchSizeExtractor),
    /// Extends `SplitAndSatisfy` with a probability-based extraction
    ProbabilitySplitAndSatisfy(ProbabilityBatchSizeExtractor<SplitAndSatisfyBatchSizeExtractor>),
    /// Extends `SplitAndSatisfy` with a timeout to prevent task starvation
    TimeoutSplitAndSatisfy(TimeoutBatchSizeExtractor<SplitAndSatisfyBatchSizeExtractor>),
    /// Combines probability-based extraction with timeout to prevent starvation
    TimeoutProbabilitySplitAndSatisfy(
        TimeoutBatchSizeExtractor<ProbabilityBatchSizeExtractor<SplitAndSatisfyBatchSizeExtractor>>,
    ),
    /// Greedily collect tasks until the desired size is met or exceeded
    BestEffort(BestEffortBatchSizeExtractor),
    /// Just return the first work in the queue, ignore the desired batch size
    Ignored(IgnoredBatchSizeExtractor),
}

impl BatchExtractorStrategy {
    /// Inject the external model/pool seed without changing extraction rules.
    pub fn with_random_seed(mut self, seed: Option<u64>) -> Self {
        let random_stream =
            seed.map(|seed| Arc::new(crate::experiment_random::SeededStream::new(seed)));
        match &mut self {
            Self::ProbabilitySplitAndSatisfy(extractor) => extractor.random_stream = random_stream,
            Self::TimeoutProbabilitySplitAndSatisfy(extractor) => {
                extractor.inner.random_stream = random_stream
            }
            _ => {}
        }
        self
    }
    /// Creates a SplitAndSatisfy strategy
    pub fn split_and_satisfy() -> Self {
        Self::SplitAndSatisfy(SplitAndSatisfyBatchSizeExtractor)
    }

    /// Creates a TimeoutSplitAndSatisfy strategy with a given timeout.
    pub fn timeout_split_and_satisfy(timeout: Duration) -> Self {
        Self::TimeoutSplitAndSatisfy(TimeoutBatchSizeExtractor::new(
            timeout,
            SplitAndSatisfyBatchSizeExtractor,
        ))
    }

    /// Creates a Greedy strategy
    pub fn greedy() -> Self {
        Self::BestEffort(BestEffortBatchSizeExtractor)
    }

    /// Creates an Ignored strategy
    ///
    /// This strategy just returns the first work in the queue, ignores the desired batch size.
    pub fn ignored() -> Self {
        Self::Ignored(IgnoredBatchSizeExtractor)
    }
}

impl BatchSizeExtractor for BatchExtractorStrategy {
    #[inline]
    fn should_extract(&self, desired_batch_size: BatchSize, model_reqs: &Requests) -> bool {
        match self {
            Self::SplitAndSatisfy(extractor) => {
                extractor.should_extract(desired_batch_size, model_reqs)
            }
            Self::ProbabilitySplitAndSatisfy(extractor) => {
                extractor.should_extract(desired_batch_size, model_reqs)
            }
            Self::TimeoutSplitAndSatisfy(extractor) => {
                extractor.should_extract(desired_batch_size, model_reqs)
            }
            Self::TimeoutProbabilitySplitAndSatisfy(extractor) => {
                extractor.should_extract(desired_batch_size, model_reqs)
            }
            Self::BestEffort(extractor) => extractor.should_extract(desired_batch_size, model_reqs),
            Self::Ignored(extractor) => extractor.should_extract(desired_batch_size, model_reqs),
        }
    }

    #[inline]
    fn extract_batch(
        &self,
        desired_batch_size: BatchSize,
        model_reqs: &Requests,
    ) -> Vec<(BaseWork<Input>, Sender<BaseWork<Output>>)> {
        match self {
            Self::SplitAndSatisfy(extractor) => {
                extractor.extract_batch(desired_batch_size, model_reqs)
            }
            Self::ProbabilitySplitAndSatisfy(extractor) => {
                extractor.extract_batch(desired_batch_size, model_reqs)
            }
            Self::TimeoutSplitAndSatisfy(extractor) => {
                extractor.extract_batch(desired_batch_size, model_reqs)
            }
            Self::TimeoutProbabilitySplitAndSatisfy(extractor) => {
                extractor.extract_batch(desired_batch_size, model_reqs)
            }
            Self::BestEffort(extractor) => extractor.extract_batch(desired_batch_size, model_reqs),
            Self::Ignored(extractor) => extractor.extract_batch(desired_batch_size, model_reqs),
        }
    }
}

#[cfg(test)]
mod ownership_tests {
    use super::*;

    #[test]
    fn injected_probability_stream_replays_without_changing_the_extraction_probability() {
        use worker::model::tensor::{DynTensor, TensorElementType};
        let (response_tx, _response_rx) = oneshot::channel();
        let requests = Requests::new(std::collections::VecDeque::from([RequestEntry {
            work: BaseWork::<Input>::new(
                uuid::Uuid::new_v4(),
                DynTensor::zeros(vec![2, 0], TensorElementType::Float),
            ),
            response_tx,
            batch_size: 2,
        }]));
        let build = |seed| ProbabilityBatchSizeExtractor {
            threshold: 0.0,
            inner: SplitAndSatisfyBatchSizeExtractor,
            random_stream: Some(Arc::new(crate::experiment_random::SeededStream::new(seed))),
        };
        let (first, replay, other) = (build(101), build(101), build(102));
        let decisions = |stream: &ProbabilityBatchSizeExtractor<_>| {
            (0..10_000)
                .map(|_| stream.should_extract(4, &requests))
                .collect::<Vec<_>>()
        };
        let first = decisions(&first);
        assert_eq!(first, decisions(&replay));
        assert_ne!(first, decisions(&other));
        assert!(
            (first.iter().filter(|&&value| value).count() as f64 / 10_000.0 - 0.5).abs() < 0.02
        );
    }

    #[tokio::test]
    async fn invalid_split_fails_the_request_instead_of_requeuing_it() {
        use worker::model::tensor::{DynTensor, TensorElementType};

        let (response_tx, response_rx) = oneshot::channel();
        let requests = Requests::new(std::collections::VecDeque::from([RequestEntry {
            work: BaseWork::<Input>::new(
                uuid::Uuid::new_v4(),
                DynTensor::zeros(vec![1, 2], TensorElementType::Float),
            ),
            response_tx,
            batch_size: 4,
        }]));
        assert!(
            SplitAndSatisfyBatchSizeExtractor
                .extract_batch(2, &requests)
                .is_empty()
        );
        assert!(requests.is_empty());
        assert!(response_rx.await.is_err());
    }

    fn clock(strategy: &BatchExtractorStrategy) -> &Arc<Mutex<Option<Instant>>> {
        match strategy {
            BatchExtractorStrategy::TimeoutSplitAndSatisfy(e) => &e.last_extraction_time,
            BatchExtractorStrategy::TimeoutProbabilitySplitAndSatisfy(e) => &e.last_extraction_time,
            _ => panic!("expected a timeout strategy"),
        }
    }

    #[test]
    fn each_model_owns_its_clock_but_scans_share_it() {
        let strategies = [
            BatchExtractorStrategy::timeout_split_and_satisfy(Duration::from_millis(8)),
            BatchExtractorStrategy::TimeoutProbabilitySplitAndSatisfy(
                TimeoutBatchSizeExtractor::new(
                    Duration::from_millis(8),
                    ProbabilityBatchSizeExtractor {
                        threshold: 0.0,
                        inner: SplitAndSatisfyBatchSizeExtractor,
                        random_stream: None,
                    },
                ),
            ),
        ];
        for template in strategies {
            let model_a = std::sync::Arc::new(template.clone());
            let model_b = template.clone();
            let next_scan = std::sync::Arc::clone(&model_a);
            *clock(&model_a).lock().unwrap() = Some(Instant::now());
            assert!(clock(&next_scan).lock().unwrap().is_some());
            assert!(clock(&model_b).lock().unwrap().is_none());
            assert!(clock(&template).lock().unwrap().is_none());
            assert!(clock(&model_a.as_ref().clone()).lock().unwrap().is_none());
        }
    }
}
