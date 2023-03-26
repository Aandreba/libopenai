use super::{
    common::{Choice, Usage},
    error::{BuilderError, Result},
    Str,
};
use crate::api::{
    error::{Error, FallibleResponse, OpenAiError},
    trim_ascii_start,
};
use chrono::{DateTime, Utc};
use futures::{future::ready, ready, Stream, TryStreamExt};
use reqwest::{Client, Response};
use serde::{Deserialize, Serialize};
use std::{borrow::Cow, collections::HashMap, ops::RangeInclusive, pin::Pin};

type InnerStream = dyn Stream<Item = reqwest::Result<bytes::Bytes>>;

#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct Completion {
    pub id: String,
    pub object: String,
    #[serde(with = "chrono::serde::ts_seconds")]
    pub created: DateTime<Utc>,
    pub model: String,
    pub choices: Vec<Choice>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

pub struct CompletionStream {
    inner: Pin<Box<InnerStream>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Builder<'a> {
    model: Str<'a>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt: Option<Vec<Str<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    suffix: Option<Str<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    echo: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<Str<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    best_of: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logit_bias: Option<HashMap<Str<'a>, f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<Str<'a>>,
}

impl Completion {
    #[inline]
    pub async fn create(
        model: impl AsRef<str>,
        prompt: impl Into<String>,
        api_key: impl AsRef<str>,
    ) -> Result<Self> {
        return Self::builder(model.as_ref())
            .prompt([prompt.into()].as_slice())
            .build(api_key.as_ref())
            .await;
    }

    #[inline]
    pub async fn create_stream(
        model: impl AsRef<str>,
        prompt: impl Into<String>,
        api_key: impl AsRef<str>,
    ) -> Result<CompletionStream> {
        return CompletionStream::create(model, prompt, api_key).await;
    }

    #[inline]
    pub fn builder<'a>(model: impl Into<Str<'a>>) -> Builder<'a> {
        return Builder::new(model);
    }
}

impl Completion {
    #[inline]
    pub fn first(&self) -> Option<&str> {
        return Some(&self.choices.first()?.text);
    }

    #[inline]
    pub fn into_first(self) -> Option<String> {
        return Some(self.choices.into_iter().next()?.text);
    }
}

impl<'a> Builder<'a> {
    pub fn new(model: impl Into<Cow<'a, str>>) -> Self {
        return Self {
            model: model.into(),
            prompt: None,
            suffix: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            n: None,
            stream: false,
            logprobs: None,
            echo: None,
            presence_penalty: None,
            best_of: None,
            logit_bias: None,
            user: None,
            stop: None,
        };
    }

    pub fn prompt<I: IntoIterator>(mut self, prompt: I) -> Self
    where
        I::Item: Into<Str<'a>>,
    {
        self.prompt = Some(prompt.into_iter().map(Into::into).collect());
        self
    }

    pub fn suffix(mut self, suffix: impl Into<Str<'a>>) -> Self {
        self.suffix = Some(suffix.into());
        self
    }

    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
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

    pub fn top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub fn n(mut self, n: u32) -> Self {
        self.n = Some(n);
        self
    }

    pub fn logprobs(mut self, logprobs: u32) -> Self {
        self.logprobs = Some(logprobs);
        self
    }

    pub fn echo(mut self, echo: bool) -> Self {
        self.echo = Some(echo);
        self
    }

    pub fn stop<I: IntoIterator>(mut self, stop: I) -> Result<Self, BuilderError<Self>>
    where
        I::Item: Into<Str<'a>>,
    {
        const MAX_SIZE: usize = 4;

        let mut stop = stop.into_iter();
        let mut result = Vec::with_capacity(MAX_SIZE);

        while let Some(next) = stop.next() {
            if result.len() == result.capacity() {
                return Err(BuilderError::msg(
                    self,
                    format!("Interator exceeds size limit of {MAX_SIZE}"),
                ));
            }
            result.push(next.into());
        }

        self.stop = Some(result);
        return Ok(self);
    }

    pub fn presence_penalty(mut self, presence_penalty: f64) -> Result<Self, BuilderError<Self>> {
        const RANGE: RangeInclusive<f64> = -2f64..=2f64;
        return match RANGE.contains(&presence_penalty) {
            true => {
                self.presence_penalty = Some(presence_penalty);
                Ok(self)
            }
            false => Err(BuilderError::msg(
                self,
                format!("presence_penalty out of range ({RANGE:?})"),
            )),
        };
    }

    pub fn best_of(mut self, best_of: u32) -> Self {
        self.best_of = Some(best_of);
        self
    }

    pub fn logit_bias<K, I>(mut self, logit_bias: I) -> Self
    where
        K: Into<Str<'a>>,
        I: IntoIterator<Item = (K, f64)>,
    {
        self.logit_bias = Some(logit_bias.into_iter().map(|(k, v)| (k.into(), v)).collect());
        self
    }

    pub fn user(mut self, user: impl Into<Str<'a>>) -> Self {
        self.user = Some(user.into());
        self
    }

    pub async fn build(self, api_key: &str) -> Result<Completion> {
        let client = Client::new();
        let resp = client
            .post("https://api.openai.com/v1/completions")
            .bearer_auth(api_key)
            .json(&self)
            .send()
            .await?
            .json::<FallibleResponse<Completion>>()
            .await?
            .into_result()?;

        return Ok(resp);
    }

    pub async fn build_stream(mut self, api_key: &str) -> Result<CompletionStream> {
        self.stream = true;
        let client = Client::new();
        let resp = client
            .post("https://api.openai.com/v1/completions")
            .bearer_auth(api_key)
            .json(&self)
            .send()
            .await?;

        return Ok(CompletionStream::new(resp));
    }
}

impl CompletionStream {
    #[inline]
    pub async fn create(
        model: impl AsRef<str>,
        prompt: impl Into<String>,
        api_key: impl AsRef<str>,
    ) -> Result<CompletionStream> {
        return Completion::builder(model.as_ref())
            .prompt([prompt.into()].as_slice())
            .build_stream(api_key.as_ref())
            .await;
    }

    #[inline]
    fn new(resp: Response) -> Self {
        return Self {
            inner: Box::pin(resp.bytes_stream()),
        };
    }
}

impl CompletionStream {
    pub fn into_text_stream(self) -> impl Stream<Item = Result<String>> {
        return self
            .try_filter_map(|x| ready(Ok(x.choices.into_iter().next())))
            .map_ok(|x| x.text);
    }
}

impl Stream for CompletionStream {
    type Item = Result<Completion>; // Result<Completion>

    #[inline]
    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        const DONE: &[u8] = b"[DONE]";

        #[derive(Debug, Deserialize)]
        struct ChunkError {
            error: OpenAiError,
        }

        match ready!(self.inner.as_mut().poll_next(cx)) {
            Some(Ok(x)) => {
                // Check if chunk is error
                if let Ok(ChunkError { error }) = serde_json::from_slice::<ChunkError>(&x) {
                    return std::task::Poll::Ready(Some(Err(Error::from(error))));
                }

                // remove initial "data"
                let x: &[u8] = trim_ascii_start(&x[5..]);
                if x.starts_with(DONE) {
                    return std::task::Poll::Ready(None);
                }

                let json = serde_json::from_slice::<Completion>(x)?;
                return std::task::Poll::Ready(Some(Ok(json)));
            }
            Some(Err(e)) => return std::task::Poll::Ready(Some(Err(e.into()))),
            None => return std::task::Poll::Ready(None),
        }
    }
}
