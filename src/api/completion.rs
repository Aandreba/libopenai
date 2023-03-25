use super::{error::Result, Slice, Str};
use crate::api::{error::FallibleResponse, trim_ascii_start};
use chrono::{DateTime, Utc};
use futures::{future::ready, ready, Stream, StreamExt, TryStreamExt};
use reqwest::{Client, Response};
use serde::{Deserialize, Serialize};
use std::{borrow::Cow, ops::RangeInclusive, pin::Pin};

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

#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct Choice {
    pub text: String,
    pub index: u32,
    pub lobprogs: Option<Vec<serde_json::Value>>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

pub struct CompletionStream {
    inner: Pin<Box<InnerStream>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Builder<'a> {
    model: Str<'a>,
    prompt: Option<Slice<'a, String>>,
    suffix: Option<Str<'a>>,
    max_tokens: Option<u32>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    n: Option<u32>,
    stream: bool,
}

impl Completion {
    #[inline]
    pub async fn new(model: &str, prompt: impl Into<String>, api_key: &str) -> Result<Self> {
        return Self::builder(model)
            .set_prompt([prompt.into()].as_slice())
            .build(api_key)
            .await;
    }

    #[inline]
    pub async fn new_stream(
        model: &str,
        prompt: impl Into<String>,
        api_key: &str,
    ) -> Result<CompletionStream> {
        return CompletionStream::new(model, prompt, api_key).await;
    }

    #[inline]
    pub fn builder<'a>(model: impl Into<Str<'a>>) -> Builder<'a> {
        return Builder::new(model);
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
        };
    }

    pub fn set_prompt(mut self, prompt: impl Into<Slice<'a, String>>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    pub fn append_prompt(mut self, prompt: impl Into<Cow<'a, [String]>>) -> Self {
        let mut prompt: Cow<'a, [String]> = prompt.into();
        match self.prompt {
            Some(ref mut prev) => {
                let prev = Cow::to_mut(prev);
                match prompt {
                    Cow::Borrowed(x) => prev.extend_from_slice(x),
                    Cow::Owned(ref mut x) => prev.append(x),
                }
            }
            None => self.prompt = Some(prompt),
        }
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

    pub fn n(mut self, n: u32) -> Self {
        self.n = Some(n);
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

        return Ok(CompletionStream::create(resp));
    }
}

impl CompletionStream {
    #[inline]
    pub async fn new(
        model: &str,
        prompt: impl Into<String>,
        api_key: &str,
    ) -> Result<CompletionStream> {
        return Completion::builder(model)
            .set_prompt([prompt.into()].as_slice())
            .build_stream(api_key)
            .await;
    }

    #[inline]
    fn create(resp: Response) -> Self {
        return Self {
            inner: Box::pin(resp.bytes_stream()),
        };
    }
}

impl CompletionStream {
    pub fn into_text_stream(self) -> impl Stream<Item = Result<String>> {
        return self
            .try_filter_map(|mut x| ready(Ok(x.choices.drain(0..=0).next())))
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

        match ready!(self.inner.as_mut().poll_next(cx)) {
            Some(Ok(x)) => {
                // remove initial "data"
                let x: &[u8] = trim_ascii_start(&x[5..]);
                if x.starts_with(DONE) {
                    return std::task::Poll::Ready(None);
                }

                let json =
                    serde_json::from_slice::<FallibleResponse<Completion>>(x)?.into_result()?;
                return std::task::Poll::Ready(Some(Ok(json)));
            }
            Some(Err(e)) => return std::task::Poll::Ready(Some(Err(e.into()))),
            None => return std::task::Poll::Ready(None),
        }
    }
}
