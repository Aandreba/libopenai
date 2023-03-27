use crate::{
    error::{FallibleResponse, Result},
    file::File,
    Client, Str,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct FineTune {
    pub id: String,
    pub object: String,
    pub model: String,
    #[serde(with = "chrono::serde::ts_seconds")]
    pub created_at: DateTime<Utc>,
    #[serde(default)]
    pub events: Option<Vec<Event>>,
    pub fine_tuned_model: Option<String>,
    pub hyperparams: Hyperparams,
    pub organization_id: String,
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
    pub batch_size: u64,
    pub learning_rate_multiplier: f64,
    pub n_epochs: u64,
    pub prompt_loss_weight: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct Event {
    pub object: String,
    #[serde(with = "chrono::serde::ts_seconds")]
    pub created_at: DateTime<Utc>,
    pub level: String,
    pub message: String,
}

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

    pub fn validation_file(mut self, validation_file: impl Into<Str<'a>>) -> Self {
        self.validation_file = Some(validation_file.into());
        self
    }
}

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
