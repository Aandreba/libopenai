use super::error::Result;
use crate::error::FallibleResponse;
use reqwest::Client;
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
    pub async fn get(model: impl AsRef<str>, api_key: impl AsRef<str>) -> Result<Model> {
        let client = Client::new();
        let models = client
            .get(format!(
                "https://api.openai.com/v1/models/{}",
                model.as_ref()
            ))
            .bearer_auth(api_key.as_ref())
            .send()
            .await?
            .json::<FallibleResponse<Model>>()
            .await?
            .into_result()?;

        return Ok(models);
    }
}

/// Lists the currently available models, and provides basic information about each one such as the owner and availability.
pub async fn models(api_key: impl AsRef<str>) -> Result<Vec<Model>> {
    #[derive(Debug, Clone, Deserialize)]
    pub struct Models {
        data: Vec<Model>,
    }

    let client = Client::new();
    let models = client
        .get("https://api.openai.com/v1/models")
        .bearer_auth(api_key.as_ref())
        .send()
        .await?
        .json::<FallibleResponse<Models>>()
        .await?
        .into_result()?;

    return Ok(models.data);
}
