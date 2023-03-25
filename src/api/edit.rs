use super::{
    common::{Choice, Usage},
    error::Result,
    Str,
};
use crate::api::error::FallibleResponse;
use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::{borrow::Cow, ops::RangeInclusive};

#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct Edit {
    pub object: String,
    #[serde(with = "chrono::serde::ts_seconds")]
    pub created: DateTime<Utc>,
    pub model: String,
    pub choices: Vec<Choice>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Builder<'a> {
    model: Str<'a>,
    instruction: Str<'a>,
    input: Option<Str<'a>>,
    n: Option<u32>,
    temperature: Option<f64>,
    top_p: Option<f64>,
}

impl Edit {
    #[inline]
    pub async fn create(
        model: impl AsRef<str>,
        instruction: impl AsRef<str>,
        input: impl AsRef<str>,
        api_key: impl AsRef<str>,
    ) -> Result<Self> {
        return Self::builder(model.as_ref(), instruction.as_ref())
            .input(input.as_ref())
            .build(api_key.as_ref())
            .await;
    }

    #[inline]
    pub fn builder<'a>(model: impl Into<Str<'a>>, instruction: impl Into<Str<'a>>) -> Builder<'a> {
        return Builder::new(model, instruction);
    }
}

impl<'a> Builder<'a> {
    pub fn new(model: impl Into<Cow<'a, str>>, instruction: impl Into<Cow<'a, str>>) -> Self {
        return Self {
            model: model.into(),
            instruction: instruction.into(),
            input: None,
            n: None,
            temperature: None,
            top_p: None,
        };
    }

    pub fn input(mut self, input: impl Into<Str<'a>>) -> Self {
        self.input = Some(input.into());
        self
    }

    pub fn n(mut self, n: u32) -> Self {
        self.n = Some(n);
        self
    }

    /// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
    ///
    /// We generally recommend altering this or `top_p` but not both.
    pub fn temperature(mut self, temperature: f64) -> Result<Self, Self> {
        const RANGE: RangeInclusive<f64> = 0f64..=2f64;
        return match RANGE.contains(&temperature) {
            true => {
                self.temperature = Some(temperature);
                Ok(self)
            }
            false => Err(self),
        };
    }

    pub fn top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub async fn build(self, api_key: &str) -> Result<Edit> {
        let client = Client::new();
        let resp = client
            .post("https://api.openai.com/v1/edits")
            .bearer_auth(api_key)
            .json(&self)
            .send()
            .await?
            .json::<FallibleResponse<Edit>>()
            .await?
            .into_result()?;

        return Ok(resp);
    }
}
