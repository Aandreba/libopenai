use super::{
    common::Usage,
    error::{BuilderError, Result},
    Str,
};
use crate::{
    error::{Error, FallibleResponse, OpenAiError},
    trim_ascii_start,
};
use chrono::{DateTime, Utc};
use futures::{ready, Stream, TryStreamExt};
use reqwest::{Client, Response};
use serde::{Deserialize, Serialize};
use std::{borrow::Cow, collections::HashMap, future::ready, ops::RangeInclusive, pin::Pin};

/// Message role
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Role {
    #[default]
    User,
    System,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message<'a> {
    pub role: Role,
    pub content: Str<'a>,
}

#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct Choice {
    pub message: Message<'static>,
    pub index: u32,
    #[serde(default)]
    pub lobprogs: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// Given a chat conversation, the model will return a chat completion response.
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

/// Given a chat conversation, the model will return a chat completion response.
pub struct CompletionStream {
    inner: Pin<Box<dyn Stream<Item = reqwest::Result<bytes::Bytes>>>>,
}

/// [`Completion`]/[`CompletionStream`] request builder
#[derive(Debug, Clone, Serialize)]
pub struct Builder<'a> {
    model: Str<'a>,
    messages: Vec<Message<'a>>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<Str<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logit_bias: Option<HashMap<Str<'a>, f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<Str<'a>>,
}

impl<'a> Message<'a> {
    /// Creates a new message
    #[inline]
    pub fn new(role: Role, content: impl Into<Str<'a>>) -> Self {
        return Self {
            role,
            content: content.into(),
        };
    }

    /// Creates a new message with a role of [`User`](Role::User)
    #[inline]
    pub fn user(content: impl Into<Str<'a>>) -> Self {
        return Self::new(Role::User, content);
    }

    /// Creates a new message with a role of [`System`](Role::System)
    #[inline]
    pub fn system(content: impl Into<Str<'a>>) -> Self {
        return Self::new(Role::System, content);
    }

    /// Creates a new message with a role of [`Assistant`](Role::Assistant)
    #[inline]
    pub fn assistant(content: impl Into<Str<'a>>) -> Self {
        return Self::new(Role::Assistant, content);
    }
}

impl Completion {
    /// Creates a completion for the chat message
    #[inline]
    pub async fn create<'a, I: IntoIterator<Item = Message<'a>>>(
        model: impl Into<Str<'a>>,
        messages: I,
        api_key: impl AsRef<str>,
    ) -> Result<Self> {
        return Self::builder(model, messages).build(api_key.as_ref()).await;
    }

    /// Creates a completion for the chat message
    #[inline]
    pub async fn create_stream<'a, I: IntoIterator<Item = Message<'a>>>(
        model: impl Into<Str<'a>>,
        messages: I,
        api_key: impl AsRef<str>,
    ) -> Result<CompletionStream> {
        return CompletionStream::create(model, messages, api_key).await;
    }

    /// Creates a new chat completion request builder
    #[inline]
    pub fn builder<'a, I: IntoIterator<Item = Message<'a>>>(
        model: impl Into<Str<'a>>,
        messages: I,
    ) -> Builder<'a> {
        return Builder::new(model, messages);
    }
}

impl Completion {
    /// Returns a reference to the first [`Choice`]
    #[inline]
    pub fn first(&self) -> Option<&Choice> {
        return self.choices.first();
    }

    /// Returns the first [`Choice`]
    #[inline]
    pub fn into_first(self) -> Option<Choice> {
        return self.choices.into_iter().next();
    }
}

impl<'a> Builder<'a> {
    /// Creates a new chat completion request builder
    pub fn new<I: IntoIterator<Item = Message<'a>>>(
        model: impl Into<Cow<'a, str>>,
        messages: I,
    ) -> Self {
        return Self {
            model: model.into(),
            messages: messages.into_iter().collect(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            n: None,
            stream: false,
            presence_penalty: None,
            frequency_penalty: None,
            logit_bias: None,
            user: None,
            stop: None,
        };
    }

    /// The maximum number of tokens to generate in the chat completion.
    ///
    /// The total length of input tokens and generated tokens is limited by the model's context length.
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

    /// An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
    ///
    /// We generally recommend altering this or `temperature` but not both.
    pub fn top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// How many chat completion choices to generate for each input message.
    pub fn n(mut self, n: u32) -> Self {
        self.n = Some(n);
        self
    }

    /// Up to 4 sequences where the API will stop generating further tokens.
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

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
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

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
    pub fn frequency_penalty(mut self, frequency_penalty: f64) -> Result<Self, BuilderError<Self>> {
        const RANGE: RangeInclusive<f64> = -2f64..=2f64;
        return match RANGE.contains(&frequency_penalty) {
            true => {
                self.frequency_penalty = Some(frequency_penalty);
                Ok(self)
            }
            false => Err(BuilderError::msg(
                self,
                format!("frequency_penalty out of range ({RANGE:?})"),
            )),
        };
    }

    /// Modify the likelihood of specified tokens appearing in the completion.
    ///
    /// Accepts a json object that maps tokens (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token.
    pub fn logit_bias<K, I>(mut self, logit_bias: I) -> Self
    where
        K: Into<Str<'a>>,
        I: IntoIterator<Item = (K, f64)>,
    {
        self.logit_bias = Some(logit_bias.into_iter().map(|(k, v)| (k.into(), v)).collect());
        self
    }

    /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
    pub fn user(mut self, user: impl Into<Str<'a>>) -> Self {
        self.user = Some(user.into());
        self
    }

    /// Sends the request
    pub async fn build(self, api_key: &str) -> Result<Completion> {
        let client = Client::new();
        let resp = client
            .post("https://api.openai.com/v1/chat/completions")
            .bearer_auth(api_key)
            .json(&self)
            .send()
            .await?
            .json::<FallibleResponse<Completion>>()
            .await?
            .into_result()?;

        return Ok(resp);
    }

    /// Sends the stream request
    pub async fn build_stream(mut self, api_key: &str) -> Result<CompletionStream> {
        self.stream = true;
        let client = Client::new();
        let resp = client
            .post("https://api.openai.com/v1/chat/completions")
            .bearer_auth(api_key)
            .json(&self)
            .send()
            .await?;

        return Ok(CompletionStream::new(resp));
    }
}

impl CompletionStream {
    /// Creates a new edit for the provided input, instruction, and parameters.
    #[inline]
    pub async fn create<'a, I: IntoIterator<Item = Message<'a>>>(
        model: impl Into<Str<'a>>,
        messages: I,
        api_key: impl AsRef<str>,
    ) -> Result<Self> {
        return Completion::builder(model, messages)
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
    /// Converts [`Stream<Item = Result<Completion>>`] into [`Stream<Item = Result<Message<'static>>>`]
    pub fn into_message_stream(self) -> impl Stream<Item = Result<Message<'static>>> {
        return self
            .try_filter_map(|x| ready(Ok(x.choices.into_iter().next())))
            .map_ok(|x| x.message);
    }

    /// Converts [`Stream<Item = Result<Completion>>`] into [`Stream<Item = Result<Cow<'static, str>>>`]
    pub fn into_text_stream(self) -> impl Stream<Item = Result<Str<'static>>> {
        return self
            .try_filter_map(|x| ready(Ok(x.choices.into_iter().next())))
            .map_ok(|x| x.message.content);
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
