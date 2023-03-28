use crate::{
    common::Delete,
    error::{BuilderError, FallibleResponse, Result},
    file::File,
    prelude::Error,
    Client, OpenAiStream, Str,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

pub mod data;

/// Manage fine-tuning jobs to tailor a model to your specific training data.
#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct FineTune {
    pub id: String,
    pub model: String,
    #[serde(with = "chrono::serde::ts_seconds")]
    pub created_at: DateTime<Utc>,
    #[serde(default)]
    pub events: Option<Vec<FineTuneEvent>>,
    pub fine_tuned_model: Option<String>,
    pub hyperparams: Option<Hyperparams>,
    pub organization_id: Option<String>,
    pub result_files: Vec<File>,
    pub status: String,
    pub validation_files: Vec<File>,
    pub training_files: Vec<File>,
    #[serde(with = "chrono::serde::ts_seconds")]
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct Hyperparams {
    #[serde(default)]
    pub batch_size: Option<u64>,
    #[serde(default)]
    pub learning_rate_multiplier: Option<f64>,
    pub n_epochs: u64,
    pub prompt_loss_weight: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct FineTuneEvent {
    #[serde(with = "chrono::serde::ts_seconds")]
    pub created_at: DateTime<Utc>,
    pub level: String,
    pub message: String,
}

pub type FineTuneEventStream = OpenAiStream<FineTuneEvent>;

#[derive(Debug, Clone, Serialize)]
pub struct Builder<'a> {
    training_file: Str<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    validation_file: Option<Str<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<Str<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    n_epochs: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    batch_size: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    learning_rate_multiplier: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_loss_weight: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    compute_classification_metrics: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    classification_n_classes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    classification_positive_class: Option<Str<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    classification_betas: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    suffix: Option<Str<'a>>,
}

impl FineTune {
    /// Creates a job that fine-tunes a specified model from a given dataset.
    #[inline]
    pub async fn new(training_file: impl AsRef<str>, client: impl AsRef<Client>) -> Result<Self> {
        return Self::builder(training_file.as_ref()).build(client).await;
    }

    /// Gets info about the fine-tune job.
    pub async fn retreive(id: impl AsRef<str>, client: impl AsRef<Client>) -> Result<Self> {
        let ft = client
            .as_ref()
            .get(format!(
                "https://api.openai.com/v1/fine-tunes/{}",
                id.as_ref()
            ))
            .send()
            .await?
            .json::<FallibleResponse<Self>>()
            .await?
            .into_result()?;

        return Ok(ft);
    }

    #[inline]
    pub fn builder<'a>(training_file: impl Into<Str<'a>>) -> Builder<'a> {
        return Builder::new(training_file);
    }
}

impl FineTune {
    #[inline]
    pub fn fine_tuned_model(&self) -> Result<&str> {
        return self
            .fine_tuned_model
            .as_deref()
            .ok_or_else(|| Error::msg("Fine tuned model not found"));
    }

    /// Immediately cancel a fine-tune job.
    #[inline]
    pub async fn cancel(self, client: impl AsRef<Client>) -> Result<Self> {
        return cancel_fine_tune(self.id, client).await;
    }

    /// Get fine-grained status updates for a fine-tune job.
    #[inline]
    pub async fn events(&self, client: impl AsRef<Client>) -> Result<Vec<FineTuneEvent>> {
        return fine_tune_events(&self.id, client).await;
    }

    /// Get fine-grained status updates for a fine-tune job.
    #[inline]
    pub async fn event_stream(&self, client: impl AsRef<Client>) -> Result<FineTuneEventStream> {
        return fine_tune_event_stream(&self.id, client).await;
    }

    /// Delete a fine-tuned model. You must have the Owner role in your organization.
    #[inline]
    pub async fn delete_model(self, client: impl AsRef<Client>) -> Option<Result<Delete>> {
        return match self.fine_tuned_model {
            Some(ftm) => Some(delete_fine_tune_model(ftm, client).await),
            None => None,
        };
    }
}

impl<'a> Builder<'a> {
    pub fn new(training_file: impl Into<Str<'a>>) -> Self {
        return Self {
            training_file: training_file.into(),
            validation_file: None,
            model: None,
            n_epochs: None,
            batch_size: None,
            learning_rate_multiplier: None,
            prompt_loss_weight: None,
            compute_classification_metrics: None,
            classification_n_classes: None,
            classification_positive_class: None,
            classification_betas: None,
            suffix: None,
        };
    }

    /// The ID of an uploaded file that contains validation data.
    ///
    /// If you provide this file, the data is used to generate validation metrics periodically during fine-tuning. These metrics can be viewed in the fine-tuning results file. Your train and validation data should be mutually exclusive.
    ///
    /// Your dataset must be formatted as a JSONL file, where each validation example is a JSON object with the keys "prompt" and "completion". Additionally, you must upload your file with the purpose `fine-tune`.
    pub fn validation_file(mut self, validation_file: impl Into<Str<'a>>) -> Self {
        self.validation_file = Some(validation_file.into());
        self
    }

    /// The name of the base model to fine-tune. You can select one of "ada", "babbage", "curie", "davinci", or a fine-tuned model created after 2022-04-21.
    pub fn model(mut self, model: impl Into<Str<'a>>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// The number of epochs to train the model for. An epoch refers to one full cycle through the training dataset.
    pub fn n_epochs(mut self, n_epochs: u64) -> Self {
        self.n_epochs = Some(n_epochs);
        self
    }

    /// The batch size to use for training. The batch size is the number of training examples used to train a single forward and backward pass.
    ///
    /// By default, the batch size will be dynamically configured to be ~0.2% of the number of examples in the training set, capped at 256 - in general, we've found that larger batch sizes tend to work better for larger datasets.
    pub fn batch_size(mut self, batch_size: u64) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// The learning rate multiplier to use for training. The fine-tuning learning rate is the original learning rate used for pretraining multiplied by this value.
    ///
    /// By default, the learning rate multiplier is the 0.05, 0.1, or 0.2 depending on final `batch_size` (larger learning rates tend to perform better with larger batch sizes). We recommend experimenting with values in the range 0.02 to 0.2 to see what produces the best results.
    pub fn learning_rate_multiplier(mut self, learning_rate_multiplier: f64) -> Self {
        self.learning_rate_multiplier = Some(learning_rate_multiplier);
        self
    }

    /// The weight to use for loss on the prompt tokens. This controls how much the model tries to learn to generate the prompt (as compared to the completion which always has a weight of 1.0), and can add a stabilizing effect to training when completions are short.
    ///
    /// If prompts are extremely long (relative to completions), it may make sense to reduce this weight so as to avoid over-prioritizing learning the prompt.
    pub fn prompt_loss_weight(mut self, prompt_loss_weight: f64) -> Self {
        self.prompt_loss_weight = Some(prompt_loss_weight);
        self
    }

    /// If set, we calculate classification-specific metrics such as accuracy and F-1 score using the validation set at the end of every epoch. These metrics can be viewed in the results file.
    ///
    /// In order to compute classification metrics, you must provide a `validation_file`. Additionally, you must specify `classification_n_classes` for multiclass classification or `classification_positive_class` for binary classification.
    pub fn compute_classification_metrics(mut self, compute_classification_metrics: bool) -> Self {
        self.compute_classification_metrics = Some(compute_classification_metrics);
        self
    }

    /// The number of classes in a classification task.
    ///
    /// This parameter is required for multiclass classification.
    pub fn classification_n_classes(mut self, classification_n_classes: u64) -> Self {
        self.classification_n_classes = Some(classification_n_classes);
        self
    }

    /// The positive class in binary classification.
    ///
    /// This parameter is needed to generate precision, recall, and F1 metrics when doing binary classification.
    pub fn classification_positive_class(
        mut self,
        classification_positive_class: impl Into<Str<'a>>,
    ) -> Self {
        self.classification_positive_class = Some(classification_positive_class.into());
        self
    }

    /// If this is provided, we calculate F-beta scores at the specified beta values. The F-beta score is a generalization of F-1 score. This is only used for binary classification.
    ///
    /// With a beta of 1 (i.e. the F-1 score), precision and recall are given the same weight. A larger beta score puts more weight on recall and less on precision. A smaller beta score puts more weight on precision and less on recall.
    pub fn classification_betas(
        mut self,
        classification_betas: impl IntoIterator<Item = f64>,
    ) -> Self {
        self.classification_betas = Some(classification_betas.into_iter().collect());
        self
    }

    /// A string of up to 40 characters that will be added to your fine-tuned model name.
    ///
    /// For example, a suffix of "custom-model-name" would produce a model name like ada:ft-your-org:custom-model-name-2022-02-15-04-21-04.
    pub fn suffix(mut self, suffix: impl Into<Str<'a>>) -> Result<Self, BuilderError<Self>> {
        const MAX_LEN: usize = 40;
        let suffix: Str<'a> = suffix.into();

        return match suffix.len() > MAX_LEN {
            false => {
                self.suffix = Some(suffix);
                Ok(self)
            }
            true => Err(BuilderError::msg(
                self,
                format!("Esceeded maximum length of '{MAX_LEN}'"),
            )),
        };
    }

    /// Sends the request.
    ///
    /// Response includes details of the enqueued job including job status and the name of the fine-tuned models once complete.
    pub async fn build(self, client: impl AsRef<Client>) -> Result<FineTune> {
        let finetune = client
            .as_ref()
            .post("https://api.openai.com/v1/fine-tunes")
            .json(&self)
            .send()
            .await?
            .json::<FallibleResponse<FineTune>>()
            .await?
            .into_result()?;

        return Ok(finetune);
    }
}

async fn fine_tune_events_inner(
    id: impl AsRef<str>,
    stream: bool,
    client: impl AsRef<Client>,
) -> Result<reqwest::Response> {
    let resp = client
        .as_ref()
        .get(format!(
            "https://api.openai.com/v1/fine-tunes/{}/events",
            id.as_ref()
        ))
        .query(&serde_json::json!({ "stream": stream }))
        .send()
        .await?;
    return Ok(resp);
}

/// Get fine-grained status updates for a fine-tune job.
pub async fn fine_tune_events(
    id: impl AsRef<str>,
    client: impl AsRef<Client>,
) -> Result<Vec<FineTuneEvent>> {
    #[derive(Debug, Deserialize)]
    struct Response {
        data: Vec<FineTuneEvent>,
    }

    let resp = fine_tune_events_inner(id, false, client)
        .await?
        .json::<FallibleResponse<Response>>()
        .await?
        .into_result()?
        .data;

    return Ok(resp);
}

/// Get fine-grained status updates for a fine-tune job.
pub async fn fine_tune_event_stream(
    id: impl AsRef<str>,
    client: impl AsRef<Client>,
) -> Result<FineTuneEventStream> {
    let stream = fine_tune_events_inner(id, true, client)
        .await?
        .bytes_stream();

    return Ok(FineTuneEventStream {
        inner: Box::pin(stream),
        current_chunk: None,
        _phtm: PhantomData,
    });
}

/// Immediately cancel a fine-tune job.
pub async fn cancel_fine_tune(id: impl AsRef<str>, client: impl AsRef<Client>) -> Result<FineTune> {
    let ft = client
        .as_ref()
        .post(format!(
            "https://api.openai.com/v1/fine-tunes/{}/cancel",
            id.as_ref()
        ))
        .send()
        .await?
        .json::<FallibleResponse<FineTune>>()
        .await?
        .into_result()?;

    return Ok(ft);
}

/// Delete a fine-tuned model. You must have the Owner role in your organization.
pub async fn delete_fine_tune_model(
    model_id: impl AsRef<str>,
    client: impl AsRef<Client>,
) -> Result<Delete> {
    let del = client
        .as_ref()
        .delete(format!(
            "https://api.openai.com/v1/models/{}",
            model_id.as_ref()
        ))
        .send()
        .await?
        .json::<FallibleResponse<Delete>>()
        .await?
        .into_result()?;

    return Ok(del);
}

/// List your organization's fine-tuning jobs
pub async fn fine_tunes(client: impl AsRef<Client>) -> Result<Vec<FineTune>> {
    #[derive(Debug, Deserialize)]
    struct Response {
        data: Vec<FineTune>,
    }

    let files = client
        .as_ref()
        .get("https://api.openai.com/v1/fine-tunes")
        .send()
        .await?
        .json::<FallibleResponse<Response>>()
        .await?
        .into_result()?;

    return Ok(files.data);
}
