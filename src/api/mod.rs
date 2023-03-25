use reqwest::Client;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct Model {
    pub id: String,
    pub object: String,
    pub owned_by: String,
    pub permision: Vec<serde_json::Value>,
}

pub async fn models(api_key: &str) -> anyhow::Result<Vec<Model>> {
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
        .json::<Models>()
        .await?;

    return Ok(models.data);
}
