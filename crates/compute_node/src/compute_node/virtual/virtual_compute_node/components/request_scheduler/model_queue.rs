use std::{
    collections::VecDeque,
    sync::{RwLock, atomic::AtomicUsize},
};

use tokio::sync::oneshot;

use crate::{
    compute_node::r#virtual::BaseWork,
    util::{Input, Output},
};

/// 请求条目，包含工作任务、响应发送器和batch size
#[derive(Debug)]
pub struct RequestEntry {
    /// 工作任务数据
    pub work: BaseWork<Input>,
    /// 响应发送器，用于返回处理结果
    pub response_tx: oneshot::Sender<BaseWork<Output>>,
    /// 任务的batch size（从tensor第一维度提取）
    pub batch_size: usize,
}

impl RequestEntry {
    pub(super) fn new(
        worker: BaseWork<Input>,
        response_tx: oneshot::Sender<BaseWork<Output>>,
    ) -> Result<Self, &'static str> {
        let batch_size = worker
            .data()
            .dimensions()
            .first()
            .copied()
            .and_then(|dim| usize::try_from(dim).ok())
            .filter(|&dim| dim > 0 && u32::try_from(dim).is_ok())
            .ok_or("Work batch size must be positive and fit u32")?;
        Ok(RequestEntry {
            work: worker,
            response_tx,
            batch_size,
        })
    }
}

/// 请求列表
pub struct Requests {
    total_batch_size: AtomicUsize,
    requests: RwLock<VecDeque<RequestEntry>>,
}

impl Requests {
    pub(super) fn new(requests: VecDeque<RequestEntry>) -> Self {
        let total_batch_size = requests.iter().map(|r| r.batch_size).sum();
        Requests {
            total_batch_size: AtomicUsize::new(total_batch_size),
            requests: RwLock::new(requests),
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.total_batch_size
            .load(std::sync::atomic::Ordering::Relaxed)
            == 0
    }

    pub(super) fn total_batch_size(&self) -> usize {
        self.total_batch_size
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub(super) fn push_back(&self, request: RequestEntry) {
        let mut requests = self.requests.write().unwrap();
        self.total_batch_size
            .fetch_add(request.batch_size, std::sync::atomic::Ordering::Relaxed);
        requests.push_back(request);
    }

    pub(super) fn fetch_target(&self, target_batch_size: usize) -> (usize, Vec<RequestEntry>) {
        if self
            .total_batch_size
            .load(std::sync::atomic::Ordering::Relaxed)
            == 0
        {
            return (0, Vec::new());
        }

        let mut current = 0;
        let mut selected_requests = Vec::new();
        let mut requests = self.requests.write().unwrap();
        let mut removed = 0;
        while let Some(request) = requests.pop_front() {
            removed += request.batch_size;
            if request.response_tx.is_closed() {
                continue;
            }
            current += request.batch_size;
            selected_requests.push(request);
            if current >= target_batch_size {
                break;
            }
        }
        self.total_batch_size
            .fetch_sub(removed, std::sync::atomic::Ordering::Relaxed);

        (current, selected_requests)
    }

    pub(super) fn push_front(&self, request: RequestEntry) {
        let mut requests = self.requests.write().unwrap();
        self.total_batch_size
            .fetch_add(request.batch_size, std::sync::atomic::Ordering::Relaxed);
        requests.push_front(request);
    }

    pub(super) fn drain(&self) -> VecDeque<RequestEntry> {
        let mut requests = self.requests.write().unwrap();
        self.total_batch_size
            .store(0, std::sync::atomic::Ordering::Relaxed);
        std::mem::take(&mut *requests)
    }
}
