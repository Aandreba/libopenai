use super::{
    common::Usage,
    completion::Choice,
    error::{BuilderError, Result},
    Str,
};
use crate::{error::FallibleResponse, Client};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{borrow::Cow, ops::RangeInclusive};

/// Given a prompt and an instruction, the model will return an edited version of the prompt.
#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct Edit {
    #[serde(with = "chrono::serde::ts_seconds")]
    pub created: DateTime<Utc>,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

/// Builder for [`Edit`]
#[derive(Debug, Clone, Serialize)]
pub struct EditBuilder<'a> {
    model: Str<'a>,
    instruction: Str<'a>,
    input: Option<Str<'a>>,
    n: Option<u64>,
    temperature: Option<f64>,
    top_p: Option<f64>,
}

impl Edit {
    /// Creates a new edit for the provided input, instruction, and parameters.
    #[inline]
    pub async fn new(
        model: impl AsRef<str>,
        input: impl AsRef<str>,
        instruction: impl AsRef<str>,
        client: impl AsRef<Client>,
    ) -> Result<Self> {
        return Self::builder(model.as_ref(), instruction.as_ref())
            .input(input.as_ref())
            .build(client)
            .await;
    }

    #[inline]
    pub fn builder<'a>(
        model: impl Into<Str<'a>>,
        instruction: impl Into<Str<'a>>,
    ) -> EditBuilder<'a> {
        return EditBuilder::new(model, instruction);
    }
}

impl<'a> EditBuilder<'a> {
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

    /// The input text to use as a starting point for the edit.
    pub fn input(mut self, input: impl Into<Str<'a>>) -> Self {
        self.input = Some(input.into());
        self
    }

    /// How many edits to generate for the input and instruction.
    pub fn n(mut self, n: u64) -> Self {
        self.n = Some(n);
        self
    }

    /// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
    ///
    /// We generally recommend altering this or `top_p` but not both.
    pub fn temperature(mut self, temperature: f64) -> Result<Self, BuilderError<Self>> {
        const RANGE: RangeInclusive<f64> = 0f64..=2f64;
        return match RANGE.contains(&temperature) {
            true => {
                self.temperature = Some(temperature);
                Ok(self)
            }
            false => Err(BuilderError::msg(
                self,
                format!("temperature out of range ({RANGE:?})"),
            )),
        };
    }

    /// An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
    ///
    /// We generally recommend altering this or `temperature` but not both.
    pub fn top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Sends the request.
    pub async fn build(self, client: impl AsRef<Client>) -> Result<Edit> {
        let resp = client
            .as_ref()
            .post("https://api.openai.com/v1/edits")
            .json(&self)
            .send()
            .await?
            .json::<FallibleResponse<Edit>>()
            .await?
            .into_result()?;

        return Ok(resp);
    }
}
