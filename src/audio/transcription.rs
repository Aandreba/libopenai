use std::{borrow::Cow, ffi::OsStr, ops::RangeInclusive, path::Path};

use super::ResponseFormat;
use crate::error::{BuilderError, Error, OpenAiError, Result};
use bytes::Bytes;
use futures::TryStream;
use rand::random;
use reqwest::{
    multipart::{Form, Part},
    Body, Client,
};
use serde::Deserialize;
use tokio_util::io::ReaderStream;

#[derive(Debug, Clone)]
pub struct Transcription {
    prompt: Option<String>,
    response_format: Option<ResponseFormat>,
    temperature: Option<f64>,
    language: Option<String>,
}

impl Transcription {
    #[inline]
    pub fn new() -> Self {
        return Self {
            prompt: None,
            response_format: None,
            temperature: None,
            language: None,
        };
    }

    pub fn prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    pub fn response_format(mut self, response_format: ResponseFormat) -> Self {
        self.response_format = Some(response_format);
        self
    }

    pub fn temperature(mut self, temperature: f64) -> Result<Self, BuilderError<Self>> {
        const RANGE: RangeInclusive<f64> = 0f64..=1f64;
        match RANGE.contains(&temperature) {
            true => {
                self.temperature = Some(temperature);
                Ok(self)
            }
            false => Err(BuilderError::msg(
                self,
                format!("temperature out of range ({RANGE:?})"),
            )),
        }
    }

    pub fn language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    pub async fn with_file(
        self,
        image: impl AsRef<Path>,
        api_key: impl AsRef<str>,
    ) -> Result<String> {
        let image = image.as_ref();
        let name = image
            .file_name()
            .map(OsStr::to_string_lossy)
            .map(Cow::into_owned)
            .ok_or_else(|| Error::msg("File name not found"))?;

        let image = Body::from(tokio::fs::File::open(image).await?);
        let image = Part::stream(Body::from(image)).file_name(name);

        return self.with_part(image, api_key).await;
    }

    pub async fn with_tokio_reader<I>(
        self,
        image: I,
        extension: impl AsRef<str>,
        api_key: impl AsRef<str>,
    ) -> Result<String>
    where
        I: 'static + Send + Sync + tokio::io::AsyncRead,
    {
        return self
            .with_stream(ReaderStream::new(image), extension, api_key)
            .await;
    }

    pub async fn with_stream<I>(
        self,
        image: I,
        extension: impl AsRef<str>,
        api_key: impl AsRef<str>,
    ) -> Result<String>
    where
        I: TryStream + Send + Sync + 'static,
        I::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
        Bytes: From<I::Ok>,
    {
        return self
            .with_body(Body::wrap_stream(image), extension, api_key)
            .await;
    }

    pub async fn with_body(
        self,
        file: impl Into<Body>,
        extension: impl AsRef<str>,
        api_key: impl AsRef<str>,
    ) -> Result<String> {
        return self
            .with_part(
                Part::stream(file).file_name(format!("{}.{}", random::<u64>(), extension.as_ref())),
                api_key,
            )
            .await;
    }

    pub async fn with_part(self, file: Part, api_key: impl AsRef<str>) -> Result<String> {
        let client = Client::new();

        let mut body = Form::new().text("model", "whisper-1").part("file", file);

        if let Some(prompt) = self.prompt {
            body = body.text("prompt", prompt)
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
        if let Some(temperature) = self.temperature {
            body = body.text("temperature", format!("{temperature}"))
        }
        if let Some(language) = self.language {
            body = body.text("language", language)
        }

        let resp = client
            .post("https://api.openai.com/v1/audio/transcriptions")
            .bearer_auth(api_key.as_ref())
            .multipart(body)
            .send()
            .await?
            .bytes()
            .await?;

        if let Ok(err) = serde_json::from_slice::<OpenAiError>(&resp) {
            return Err(Error::OpenAI(err));
        }

        return match self.response_format {
            None | Some(ResponseFormat::Json) => {
                #[derive(Debug, Deserialize)]
                struct Body {
                    text: String,
                }

                let Body { text } = serde_json::from_slice::<Body>(&resp)?;
                Ok(text)
            }
            Some(_) => todo!(),
        };
    }
}
