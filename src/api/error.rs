use serde::Deserialize;
use std::fmt::Display;

pub type Result<T, E = Error> = ::core::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("OpenAI error: {0}")]
    OpenAI(#[from] OpenAiError),
    #[error("Reqwest error: {0}")]
    Reqwest(#[from] reqwest::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(in crate::api) enum FallibleResponse<T> {
    Ok(T),
    Err { error: OpenAiError },
}

impl<T> FallibleResponse<T> {
    #[inline]
    pub fn into_result(self) -> Result<T, OpenAiError> {
        match self {
            FallibleResponse::Ok(x) => Ok(x),
            FallibleResponse::Err { error } => Err(error),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct OpenAiError {
    pub message: String,
    #[serde(rename = "type")]
    pub ty: String,
    pub param: Option<serde_json::Value>,
    pub code: Option<serde_json::Value>,
}

impl Display for OpenAiError {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.message.fmt(f)
    }
}

impl std::error::Error for OpenAiError {}
