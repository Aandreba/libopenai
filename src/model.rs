use super::error::Result;
use crate::{error::FallibleResponse, Client};
use serde::Deserialize;

/// OpenAI module. Each module has different capabilities and price points.
#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct Model {
    pub id: String,
    pub object: String,
    pub owned_by: String,
    pub permission: Vec<serde_json::Value>,
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
