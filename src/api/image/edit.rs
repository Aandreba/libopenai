use super::{Image, ResponseFormat, Size};
use crate::api::error::{Error, FallibleResponse, Result};
use bytes::Bytes;
use futures::{future::try_join, TryStream};
use reqwest::{
    multipart::{Form, Part},
    Body, Client,
};
use std::{ops::RangeInclusive, path::Path};
use tokio_util::io::ReaderStream;

#[derive(Debug, Clone)]
pub struct Builder {
    prompt: String,
    n: Option<u32>,
    size: Option<Size>,
    response_format: Option<ResponseFormat>,
    user: Option<String>,
}

impl Image {
    #[inline]
    pub async fn edit(prompt: impl AsRef<str>, api_key: impl AsRef<str>) -> Result<Self> {
        return Self::generate_builder(prompt.as_ref())?
            .build(api_key.as_ref())
            .await;
    }

    #[inline]
    pub fn edit_builder<'a>(prompt: impl Into<String>) -> Result<Builder> {
        return Builder::new(prompt);
    }
}

impl Builder {
    #[inline]
    pub fn new(prompt: impl Into<String>) -> Result<Self> {
        let prompt: String = prompt.into();
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
    pub fn user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    pub async fn with_file(
        self,
        image: impl AsRef<Path>,
        mask: Option<&Path>,
        api_key: impl AsRef<str>,
    ) -> Result<Image> {
        let (image, mask) = match mask {
            Some(mask) => {
                let (image, mask) =
                    try_join(tokio::fs::File::open(image), tokio::fs::File::open(mask)).await?;
                (Body::from(image), Some(Body::from(mask)))
            }
            None => {
                let image = tokio::fs::File::open(image).await?;
                (Body::from(image), None)
            }
        };

        return self.with_body(image, mask, api_key).await;
    }

    pub async fn with_tokio_reader<I>(self, image: I, api_key: impl AsRef<str>) -> Result<Image>
    where
        I: 'static + Send + Sync + tokio::io::AsyncRead,
    {
        return self.with_stream(ReaderStream::new(image), api_key).await;
    }

    pub async fn with_stream<I>(self, image: I, api_key: impl AsRef<str>) -> Result<Image>
    where
        I: TryStream + Send + Sync + 'static,
        I::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
        Bytes: From<I::Ok>,
    {
        return self
            .with_body(Body::wrap_stream(image), None, api_key)
            .await;
    }

    pub async fn with_body(
        self,
        image: impl Into<Body>,
        mask: Option<Body>,
        api_key: impl AsRef<str>,
    ) -> Result<Image> {
        return self
            .with_part(Part::stream(image), mask.map(Part::stream), api_key)
            .await;
    }

    pub async fn with_part(
        self,
        image: Part,
        mask: Option<Part>,
        api_key: impl AsRef<str>,
    ) -> Result<Image> {
        let client = Client::new();

        let mut body = Form::new().text("prompt", self.prompt);
        body = body.part("image", image);

        if let Some(mask) = mask {
            body = body.part("mask", mask)
        }
        if let Some(n) = self.n {
            body = body.text("n", format!("{n}"))
        }
        if let Some(size) = self.size {
            body = body.text(
                "size",
                match serde_json::to_value(&size)? {
                    serde_json::Value::String(x) => x,
                    _ => return Err(Error::msg("Unexpected error")),
                },
            )
        }
        if let Some(response_format) = self.response_format {
            body = body.text(
                "response_format",
                match serde_json::to_value(&response_format)? {
                    serde_json::Value::String(x) => x,
                    _ => return Err(Error::msg("Unexpected error")),
                },
            )
        }
        if let Some(user) = self.user {
            body = body.text("user", user)
        }

        let resp = client
            .post("https://api.openai.com/v1/images/edits")
            .bearer_auth(api_key.as_ref())
            .multipart(body)
            .send()
            .await?
            .json::<FallibleResponse<Image>>()
            .await?
            .into_result()?;

        return Ok(resp);
    }
}
