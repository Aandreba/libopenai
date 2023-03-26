use crate::{
    error::{FallibleResponse, Result},
    file::File,
    Client,
};
use chrono::{DateTime, Utc};
use serde::Deserialize;

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
