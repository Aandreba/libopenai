use super::{Images, ResponseFormat, Size};
use crate::{
    error::{BuilderError, Error, FallibleResponse, Result},
    Client, Str,
};
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
    /// Creates an image given a prompt.
    #[inline]
    pub async fn new(prompt: impl AsRef<str>, client: impl AsRef<Client>) -> Result<Self> {
        return Self::generate(prompt.as_ref())?.build(client).await;
    }

    /// Creates an image given a prompt.
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

    /// The number of images to generate. Must be between 1 and 10.
    #[inline]
    pub fn n(mut self, n: u32) -> Result<Self, BuilderError<Self>> {
        const RANGE: RangeInclusive<u32> = 1..=10;
        return match RANGE.contains(&n) {
            true => {
                self.n = Some(n);
                Ok(self)
            }
            false => Err(BuilderError::msg(
                self,
                format!("n out of range ({RANGE:?})"),
            )),
        };
    }

    /// The size of the generated images.
    #[inline]
    pub fn size(mut self, size: Size) -> Self {
        self.size = Some(size);
        self
    }

    /// The format in which the generated images are returned.
    #[inline]
    pub fn response_format(mut self, response_format: ResponseFormat) -> Self {
        self.response_format = Some(response_format);
        self
    }

    /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
    #[inline]
    pub fn user(mut self, user: impl Into<Str<'a>>) -> Self {
        self.user = Some(user.into());
        self
    }

    /// Sends the request
    pub async fn build(self, client: impl AsRef<Client>) -> Result<Images> {
        let resp = client
            .as_ref()
            .post("https://api.openai.com/v1/images/generations")
            .json(&self)
            .send()
            .await?
            .json::<FallibleResponse<Images>>()
            .await?
            .into_result()?;

        return Ok(resp);
    }
}
