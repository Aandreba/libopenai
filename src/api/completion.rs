use super::{
    common::{Choice, Usage},
    error::Result,
    Slice, Str,
};
use crate::api::{error::FallibleResponse, trim_ascii_start};
use chrono::{DateTime, Utc};
use futures::{future::ready, ready, Stream, TryStreamExt};
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
    pub async fn create(
        model: impl AsRef<str>,
        prompt: impl Into<String>,
        api_key: impl AsRef<str>,
    ) -> Result<Self> {
        return Self::builder(model.as_ref())
            .set_prompts([prompt.into()].as_slice())
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
        };
    }

    pub fn set_prompt(self, prompt: impl Into<String>) -> Self {
        self.set_prompts(vec![prompt.into()])
    }

    pub fn set_prompts(mut self, prompt: impl Into<Slice<'a, String>>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    pub fn append_prompts(mut self, prompt: impl Into<Cow<'a, [String]>>) -> Self {
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
            .set_prompts([prompt.into()].as_slice())
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
