use super::error::Result;
use crate::api::error::FallibleResponse;
use reqwest::Client;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Moderation {
    pub id: String,
    pub model: String,
    pub results: Vec<ModerationResult>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModerationResult {
    pub categories: Categories<bool>,
    pub category_scores: Categories<f64>,
    pub flagged: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Categories<T> {
    pub hate: T,
    #[serde(rename = "hate/threatening")]
    pub hate_threatening: T,
    #[serde(rename = "self-harm")]
    pub self_harm: T,
    pub sexual: T,
    #[serde(rename = "sexual/minors")]
    pub sexual_minors: T,
    pub violence: T,
    #[serde(rename = "violence/graphic")]
    pub violence_graphic: T,
}

impl Moderation {
    pub async fn new(
        input: impl AsRef<str>,
        model: Option<&str>,
        api_key: impl AsRef<str>,
    ) -> Result<Self> {
        let client = Client::new();
        let resp = client
            .post("https://api.openai.com/v1/moderations")
            .bearer_auth(api_key.as_ref())
            .json(&serde_json::json! {{
                "input": input.as_ref(),
                "model": model
            }})
            .send()
            .await?
            .json::<FallibleResponse<Self>>()
            .await?
            .into_result()?;

        return Ok(resp);
    }
}
