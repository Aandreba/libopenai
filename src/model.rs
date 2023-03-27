use super::error::Result;
use crate::{error::FallibleResponse, Client};
use chrono::{DateTime, Utc};
use serde::Deserialize;

/// OpenAI module. Each module has different capabilities and price points.
#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct Model {
    pub id: String,
    pub owned_by: String,
    pub permission: Vec<Permission>,
    #[serde(default)]
    pub root: Option<String>,
    #[serde(default)]
    pub parent: Option<String>,
}

/// Permissions granted to a [`Model`]
#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct Permission {
    #[serde(with = "chrono::serde::ts_seconds")]
    pub created_at: DateTime<Utc>,
    pub id: String,
    pub allow_create_engine: bool,
    pub allow_sampling: bool,
    pub allow_lobprobs: bool,
    pub allow_search_indices: bool,
    pub allow_view: bool,
    pub allow_fine_tuning: bool,
    pub organization: String,
    pub is_blocking: bool,
}

impl Model {
    /// Retrieves a model instance, providing basic information about the model such as the owner and permissioning.
    pub async fn get(model: impl AsRef<str>, client: impl AsRef<Client>) -> Result<Model> {
        let models = client
            .as_ref()
            .get(format!(
                "https://api.openai.com/v1/models/{}",
                model.as_ref()
            ))
            .send()
            .await?
            .json::<FallibleResponse<Model>>()
            .await?
            .into_result()?;

        return Ok(models);
    }
}

/// Lists the currently available models, and provides basic information about each one such as the owner and availability.
pub async fn models(client: impl AsRef<Client>) -> Result<Vec<Model>> {
    #[derive(Debug, Clone, Deserialize)]
    pub struct Models {
        data: Vec<Model>,
    }

    let models = client
        .as_ref()
        .get("https://api.openai.com/v1/models")
        .send()
        .await?
        .json::<FallibleResponse<Models>>()
        .await?
        .into_result()?;

    return Ok(models.data);
}
