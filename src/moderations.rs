use super::error::Result;
use crate::{error::FallibleResponse, Client};
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
        client: impl AsRef<Client>,
    ) -> Result<Self> {
        let resp = client
            .as_ref()
            .post("https://api.openai.com/v1/moderations")
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
