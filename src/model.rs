use crate::error::FallibleResponse;

use super::error::Result;
use reqwest::Client;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct Model {
    pub id: String,
    pub object: String,
    pub owned_by: String,
    pub permission: Vec<serde_json::Value>,
}

impl Model {
    pub async fn get(api_key: &str, model: &str) -> Result<Model> {
        let client = Client::new();
        let models = client
            .get(format!("https://api.openai.com/v1/models/{model}"))
            .bearer_auth(api_key)
            .send()
            .await?
            .json::<FallibleResponse<Model>>()
            .await?
            .into_result()?;

        return Ok(models);
    }
}

pub async fn models(api_key: &str) -> Result<Vec<Model>> {
    #[derive(Debug, Clone, Deserialize)]
    pub struct Models {
        data: Vec<Model>,
    }

    let client = Client::new();
    let models = client
        .get("https://api.openai.com/v1/models")
        .bearer_auth(api_key)
        .send()
        .await?
        .json::<FallibleResponse<Models>>()
        .await?
        .into_result()?;

    return Ok(models.data);
}
