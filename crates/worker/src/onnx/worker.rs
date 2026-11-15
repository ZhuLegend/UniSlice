//! ONNX Runtime Worker

use std::{future::Future, sync::Arc};

use model::model::InferenceModel;
use snafu::OptionExt;
use tracing::{Level, event, span};

use crate::{ModelNotLoadedSnafu, Worker, WorkerResult, onnx::model::DynOnnxModel};

/// ONNX Runtime Worker
#[derive(Debug, Default)]
pub struct OnnxWorker {
    model: Option<Arc<DynOnnxModel>>,
}

impl Worker for OnnxWorker {
    type Model = DynOnnxModel;

    type ModelRef = Arc<DynOnnxModel>;

    async fn model(&self) -> Option<Self::ModelRef> {
        self.model.clone()
    }

    fn load_model(
        &mut self,
        load_config: <Self::Model as InferenceModel>::LoadConfig,
    ) -> impl Future<Output = WorkerResult<Self::ModelRef, <Self::Model as InferenceModel>::Error>> + Send
    {
        let span = span!(Level::TRACE, "OnnxWorker::load_model", config = ?load_config);
        let _enter = span.enter();
        event!(Level::TRACE, "load model");

        // produce result out of future in order to avoid require [OnnxModelLoadConfig] to be `Send`
        let result = DynOnnxModel::load(load_config)
            .map(|model| {
                let model = Arc::new(model);
                self.model = Some(model.clone());

                model
            })
            .map_err(Into::into);

        async move { result }
    }

    #[tracing::instrument(level = "trace", skip(self))]
    async fn unload_model(
        &mut self,
    ) -> WorkerResult<Self::ModelRef, <Self::Model as InferenceModel>::Error> {
        self.model.take().context(ModelNotLoadedSnafu {})
    }
}
