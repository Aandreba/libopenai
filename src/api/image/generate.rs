use super::{Images, ResponseFormat, Size};
use crate::api::{
    error::{Error, FallibleResponse, Result},
    Str,
};
use reqwest::Client;
use serde::Serialize;
use std::ops::RangeInclusive;

#[derive(Debug, Clone, Serialize)]
pub struct Builder<'a> {
    prompt: Str<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<Size>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<Str<'a>>,
}

impl Images {
    #[inline]
    pub async fn new(prompt: impl AsRef<str>, api_key: impl AsRef<str>) -> Result<Self> {
        return Self::generate(prompt.as_ref())?
            .build(api_key.as_ref())
            .await;
    }

    #[inline]
    pub fn generate<'a>(prompt: impl Into<Str<'a>>) -> Result<Builder<'a>> {
        return Builder::new(prompt);
    }
}

impl<'a> Builder<'a> {
    #[inline]
    pub fn new(prompt: impl Into<Str<'a>>) -> Result<Self> {
        let prompt: Str<'a> = prompt.into();
        if prompt.len() > 1000 {
            return Err(Error::msg("Message excedes character limit of 1000"));
        }

        return Ok(Self {
            prompt: prompt.into(),
            n: None,
            size: None,
            response_format: None,
            user: None,
        });
    }

    #[inline]
    pub fn n(mut self, n: u32) -> Result<Self, Self> {
        const RANGE: RangeInclusive<u32> = 1..=10;
        return match RANGE.contains(&n) {
            true => {
                self.n = Some(n);
                Ok(self)
            }
            false => Err(self),
        };
    }

    #[inline]
    pub fn size(mut self, size: Size) -> Self {
        self.size = Some(size);
        self
    }

    #[inline]
    pub fn response_format(mut self, response_format: ResponseFormat) -> Self {
        self.response_format = Some(response_format);
        self
    }

    #[inline]
    pub fn user(mut self, user: impl Into<Str<'a>>) -> Self {
        self.user = Some(user.into());
        self
    }

    pub async fn build(self, api_key: &str) -> Result<Images> {
        let client = Client::new();
        let resp = client
            .post("https://api.openai.com/v1/images/generations")
            .bearer_auth(api_key)
            .json(&self)
            .send()
            .await?
            .json::<FallibleResponse<Images>>()
            .await?
            .into_result()?;

        return Ok(resp);
    }
}
