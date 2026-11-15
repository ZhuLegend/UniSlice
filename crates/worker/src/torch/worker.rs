//! Torch Worker

use std::sync::Arc;

use model::model::InferenceModel;

use crate::{Worker, WorkerError, WorkerResult, torch::TorchModel};

/// Torch Worker
#[derive(Debug, Default)]
pub struct TorchWorker {
    model: Option<Arc<TorchModel>>,
}

impl Worker for TorchWorker {
    type Model = TorchModel;

    type ModelRef = Arc<TorchModel>;

    async fn model(&self) -> Option<Self::ModelRef> {
        self.model.clone()
    }

    #[tracing::instrument(level = "trace", skip(self))]
    async fn load_model(
        &mut self,
        load_config: <Self::Model as InferenceModel>::LoadConfig,
    ) -> WorkerResult<Self::ModelRef, <Self::Model as InferenceModel>::Error> {
        let model = Arc::new(TorchModel::load(load_config)?);
        self.model = Some(model.clone());

        Ok(model)
    }

    #[tracing::instrument(level = "trace", skip(self))]
    async fn unload_model(
        &mut self,
    ) -> WorkerResult<Self::ModelRef, <Self::Model as InferenceModel>::Error> {
        self.model.take().ok_or(WorkerError::ModelNotLoaded)
    }
}
