use crate::{
    common::Usage,
    error::{FallibleResponse, Result},
    Client, Str,
};
use serde::{Deserialize, Serialize};

/// Get a vector representation of a given input that can be easily consumed by machine learning models and algorithms.
#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct Embedding {
    pub embedding: Vec<f64>,
    pub index: u64,
}

/// A list of [`Embedding`]s
#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct EmbeddingResult {
    pub data: Vec<Embedding>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingBuilder<'a> {
    model: Str<'a>,
    input: Str<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<Str<'a>>,
}

impl Embedding {
    /// Creates an embedding vector representing the input text.
    #[inline]
    pub async fn new(
        model: impl AsRef<str>,
        input: impl AsRef<str>,
        client: impl AsRef<Client>,
    ) -> Result<EmbeddingResult> {
        return Self::builder(model.as_ref(), input.as_ref())
            .build(client)
            .await;
    }

    #[inline]
    pub fn builder<'a>(
        model: impl Into<Str<'a>>,
        input: impl Into<Str<'a>>,
    ) -> EmbeddingBuilder<'a> {
        EmbeddingBuilder::new(model, input)
    }
}

impl<'a> EmbeddingBuilder<'a> {
    #[inline]
    pub fn new(model: impl Into<Str<'a>>, input: impl Into<Str<'a>>) -> Self {
        return Self {
            model: model.into(),
            input: input.into(),
            user: None,
        };
    }

    /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
    #[inline]
    pub fn user(mut self, user: impl Into<Str<'a>>) -> Self {
        self.user = Some(user.into());
        self
    }

    /// Sends the request
    pub async fn build(self, client: impl AsRef<Client>) -> Result<EmbeddingResult> {
        let result = client
            .as_ref()
            .post("https://api.openai.com/v1/embeddings")
            .json(&self)
            .send()
            .await?
            .json::<FallibleResponse<EmbeddingResult>>()
            .await?
            .into_result()?;

        return Ok(result);
    }
}
