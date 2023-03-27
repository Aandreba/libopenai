use super::error::Result;
use crate::{error::FallibleResponse, Client};
use serde::{Deserialize, Serialize};

/// Given a input text, outputs if the model classifies it as violating OpenAI's content policy.
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
    /// Classifies if text violates OpenAI's Content Policy
    pub async fn new(
        input: impl AsRef<str>,
        model: Option<&str>,
        client: impl AsRef<Client>,
    ) -> Result<Self> {
        #[derive(Debug, Serialize)]
        struct Body<'a> {
            input: &'a str,
            #[serde(skip_serializing_if = "Option::is_none")]
            model: Option<&'a str>,
        }

        let resp = client
            .as_ref()
            .post("https://api.openai.com/v1/moderations")
            .json(&Body {
                input: input.as_ref(),
                model,
            })
            .send()
            .await?
            .json::<FallibleResponse<Self>>()
            .await?
            .into_result()?;

        return Ok(resp);
    }
}
