use super::{parse_audio_response, AudioResponse, AudioResponseFormat};
use crate::{
    error::{BuilderError, Error, Result},
    Client,
};
use bytes::Bytes;
use futures::TryStream;
use rand::random;
use reqwest::{
    multipart::{Form, Part},
    Body,
};
use std::{borrow::Cow, ffi::OsStr, ops::RangeInclusive, path::Path};
use tokio_util::io::ReaderStream;

/// Translates audio into into English.
#[derive(Debug, Clone)]
pub struct TranslationBuilder {
    prompt: Option<String>,
    response_format: AudioResponseFormat,
    temperature: Option<f64>,
}

impl TranslationBuilder {
    #[inline]
    pub fn new() -> Self {
        return Self {
            prompt: None,
            response_format: AudioResponseFormat::VerboseJson,
            temperature: None,
        };
    }

    /// An optional text to guide the model's style or continue a previous audio segment. The prompt should be in English.
    pub fn prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    /// The format of the transcript output.
    pub fn response_format(mut self, response_format: AudioResponseFormat) -> Self {
        self.response_format = response_format;
        self
    }

    /// The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. If set to 0, the model will use log probability to automatically increase the temperature until certain thresholds are hit.
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

    /// Sends the request with the specified file.
    pub async fn with_file(
        self,
        image: impl AsRef<Path>,
        client: impl AsRef<Client>,
    ) -> Result<AudioResponse> {
        let image = image.as_ref();
        let name = image
            .file_name()
            .map(OsStr::to_string_lossy)
            .map(Cow::into_owned)
            .ok_or_else(|| Error::msg("File name not found"))?;

        let image = Body::from(tokio::fs::File::open(image).await?);
        let image = Part::stream(Body::from(image)).file_name(name);

        return self.with_part(image, client).await;
    }

    /// Sends the request with the specified file.
    pub async fn with_tokio_reader<I>(
        self,
        image: I,
        extension: impl AsRef<str>,
        client: impl AsRef<Client>,
    ) -> Result<AudioResponse>
    where
        I: 'static + Send + Sync + tokio::io::AsyncRead,
    {
        return self
            .with_stream(ReaderStream::new(image), extension, client)
            .await;
    }

    /// Sends the request with the specified file.
    pub async fn with_stream<I>(
        self,
        image: I,
        extension: impl AsRef<str>,
        client: impl AsRef<Client>,
    ) -> Result<AudioResponse>
    where
        I: TryStream + Send + Sync + 'static,
        I::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
        Bytes: From<I::Ok>,
    {
        return self
            .with_body(Body::wrap_stream(image), extension, client)
            .await;
    }

    /// Sends the request with the specified file.
    pub async fn with_body(
        self,
        file: impl Into<Body>,
        extension: impl AsRef<str>,
        client: impl AsRef<Client>,
    ) -> Result<AudioResponse> {
        return self
            .with_part(
                Part::stream(file).file_name(format!("{}.{}", random::<u64>(), extension.as_ref())),
                client,
            )
            .await;
    }

    /// Sends the request with the specified file.
    pub async fn with_part(self, file: Part, client: impl AsRef<Client>) -> Result<AudioResponse> {
        let mut body = Form::new()
            .text("model", "whisper-1")
            .text(
                "response_format",
                match serde_json::to_value(&self.response_format)? {
                    serde_json::Value::String(x) => x,
                    _ => return Err(Error::msg("Unexpected error")),
                },
            )
            .part("file", file);

        if let Some(prompt) = self.prompt {
            body = body.text("prompt", prompt)
        }
        if let Some(temperature) = self.temperature {
            body = body.text("temperature", format!("{temperature}"))
        }

        let resp = client
            .as_ref()
            .post("https://api.openai.com/v1/audio/translations")
            .multipart(body)
            .send()
            .await?;

        return parse_audio_response(resp, self.response_format).await;
    }
}
