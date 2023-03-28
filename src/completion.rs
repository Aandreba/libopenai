use super::{
    common::Usage,
    error::{BuilderError, Result},
    Str,
};
use crate::{error::FallibleResponse, Client, OpenAiStream};
use chrono::{DateTime, Utc};
use futures::{future::ready, Stream, TryStreamExt};
use reqwest::Response;
use serde::{Deserialize, Serialize};
use std::{borrow::Cow, collections::HashMap, marker::PhantomData, ops::RangeInclusive};

#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct Choice {
    #[serde(with = "crate::serde_trim_string")]
    pub text: String,
    pub index: u64,
    #[serde(default)]
    pub logprobs: Option<Logprobs>,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct Logprobs {
    pub tokens: Vec<String>,
    pub token_logprobs: Vec<f64>,
    pub top_logprobs: Vec<HashMap<String, f64>>,
    pub text_offset: Vec<u64>,
}

/// Given a prompt, the model will return one or more predicted completions, and can also return the probabilities of alternative tokens at each position.
#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct Completion {
    pub id: String,
    #[serde(with = "chrono::serde::ts_seconds")]
    pub created: DateTime<Utc>,
    pub model: String,
    pub choices: Vec<Choice>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

/// Given a prompt, the model will return one or more predicted completions, and can also return the probabilities of alternative tokens at each position.
pub type CompletionStream = OpenAiStream<Completion>;

/// [`Completion`]/[`CompletionStream`] request builder
#[derive(Debug, Clone, Serialize)]
pub struct CompletionBuilder<'a> {
    model: Str<'a>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt: Option<Vec<Str<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    suffix: Option<Str<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    n: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    echo: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<Str<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    best_of: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logit_bias: Option<HashMap<Str<'a>, f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<Str<'a>>,
}

impl Completion {
    /// Creates a completion for the provided prompt and parameters
    #[inline]
    pub async fn new(
        model: impl AsRef<str>,
        prompt: impl AsRef<str>,
        client: impl AsRef<Client>,
    ) -> Result<Self> {
        return Self::builder(model.as_ref(), prompt.as_ref())
            .build(client)
            .await;
    }

    /// Creates a completion for the provided prompt and parameters
    #[inline]
    pub async fn new_stream(
        model: impl AsRef<str>,
        prompt: impl AsRef<str>,
        client: impl AsRef<Client>,
    ) -> Result<CompletionStream> {
        return CompletionStream::new(model, prompt, client).await;
    }

    /// Creates a completion request builder
    #[inline]
    pub fn builder<'a>(
        model: impl Into<Str<'a>>,
        prompt: impl Into<Str<'a>>,
    ) -> CompletionBuilder<'a> {
        return CompletionBuilder::new(model).prompt([prompt]);
    }

    /// Creates a completion request builder
    #[inline]
    pub fn raw_builder<'a>(model: impl Into<Str<'a>>) -> CompletionBuilder<'a> {
        return CompletionBuilder::new(model);
    }
}

impl Completion {
    /// Returns a reference to the completion's first choice
    #[inline]
    pub fn first(&self) -> Option<&Choice> {
        return self.choices.first();
    }

    /// Returns the completion's first choice
    #[inline]
    pub fn into_first(self) -> Option<Choice> {
        return self.choices.into_iter().next();
    }
}

impl<'a> CompletionBuilder<'a> {
    /// Creates a new completion builder
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
            frequency_penalty: None,
            presence_penalty: None,
            best_of: None,
            logit_bias: None,
            user: None,
            stop: None,
        };
    }

    /// The prompt(s) to generate completions for, encoded as a string, array of strings, array of tokens, or array of token arrays.
    ///
    /// Note that <|endoftext|> is the document separator that the model sees during training, so if a prompt is not specified the model will generate as if from the beginning of a new document.
    pub fn prompt<I: IntoIterator>(mut self, prompt: I) -> Self
    where
        I::Item: Into<Str<'a>>,
    {
        self.prompt = Some(prompt.into_iter().map(Into::into).collect());
        self
    }

    /// The suffix that comes after a completion of inserted text.
    pub fn suffix(mut self, suffix: impl Into<Str<'a>>) -> Self {
        self.suffix = Some(suffix.into());
        self
    }

    /// The maximum number of tokens to generate in the completion.
    ///
    /// The token count of your prompt plus `max_tokens` cannot exceed the model's context length. Most models have a context length of 2048 tokens (except for the newest models, which support 4096).
    pub fn max_tokens(mut self, max_tokens: u64) -> Self {
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
    /// We generally recommend altering this or temperature but not both.
    pub fn top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// How many completions to generate for each prompt.
    ///
    /// > **Note**: Because this parameter generates many completions, it can quickly consume your token quota. Use carefully and ensure that you have reasonable settings for `max_tokens` and `stop`.
    pub fn n(mut self, n: u64) -> Self {
        self.n = Some(n);
        self
    }

    /// Include the log probabilities on the logprobs most likely tokens, as well the chosen tokens. For example, if logprobs is 5, the API will return a list of the 5 most likely tokens. The API will always return the logprob of the sampled token, so there may be up to logprobs+1 elements in the response.
    ///
    /// The maximum value for logprobs is 5.
    pub fn logprobs(mut self, logprobs: u64) -> Result<Self, BuilderError<Self>> {
        const MAX: u64 = 5;
        match logprobs > MAX {
            true => Err(BuilderError::msg(
                self,
                format!("Exceeded maximum value of '{MAX}'"),
            )),
            false => {
                self.logprobs = Some(logprobs);
                Ok(self)
            }
        }
    }

    /// Echo back the prompt in addition to the completion
    pub fn echo(mut self, echo: bool) -> Self {
        self.echo = Some(echo);
        self
    }

    /// Up to 4 sequences where the API will stop generating further tokens. The returned text will not contain the stop sequence.
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

    /// Generates `best_of` completions server-side and returns the "best" (the one with the highest log probability per token). Results cannot be streamed.
    ///
    /// When used with `n`, `best_of` controls the number of candidate completions and `n` specifies how many to return – `best_of` must be greater than `n`.
    ///
    /// > **Note**: Because this parameter generates many completions, it can quickly consume your token quota. Use carefully and ensure that you have reasonable settings for `max_tokens` and `stop`.
    pub fn best_of(mut self, best_of: u64) -> Self {
        self.best_of = Some(best_of);
        self
    }

    /// Modify the likelihood of specified tokens appearing in the completion.
    ///
    /// Accepts a json object that maps tokens (specified by their token ID in the GPT tokenizer) to an associated bias value from -100 to 100. You can use this tokenizer tool (which works for both GPT-2 and GPT-3) to convert text to token IDs. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token.
    ///
    /// As an example, you can pass `{"50256": -100}` to prevent the <|endoftext|> token from being generated.
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
    pub async fn build(self, client: impl AsRef<Client>) -> Result<Completion> {
        let resp = client
            .as_ref()
            .post("https://api.openai.com/v1/completions")
            .json(&self)
            .send()
            .await?
            .json::<FallibleResponse<Completion>>()
            .await?
            .into_result()?;

        return Ok(resp);
    }

    /// Sends the request as a stream request
    pub async fn build_stream(mut self, client: impl AsRef<Client>) -> Result<CompletionStream> {
        self.stream = true;
        let resp = client
            .as_ref()
            .post("https://api.openai.com/v1/completions")
            .json(&self)
            .send()
            .await?;

        return Ok(CompletionStream::create(resp));
    }
}

impl CompletionStream {
    /// Creates a completion for the provided prompt and parameters
    #[inline]
    pub async fn new(
        model: impl AsRef<str>,
        prompt: impl AsRef<str>,
        client: impl AsRef<Client>,
    ) -> Result<CompletionStream> {
        return Completion::builder(model.as_ref(), prompt.as_ref())
            .build_stream(client)
            .await;
    }

    #[inline]
    fn create(resp: Response) -> Self {
        return Self {
            inner: Box::pin(resp.bytes_stream()),
            current_chunk: None,
            _phtm: PhantomData,
        };
    }
}

impl CompletionStream {
    /// Converts [`Stream<Item = Result<Completion>>`] into [`Stream<Item = Result<Choice>>`]
    pub fn into_choice_stream(self) -> impl Stream<Item = Result<Choice>> {
        return self.try_filter_map(|x| ready(Ok(x.choices.into_iter().next())));
    }

    /// Converts [`Stream<Item = Result<Completion>>`] into [`Stream<Item = Result<String>>`]
    pub fn into_text_stream(self) -> impl Stream<Item = Result<String>> {
        return self
            .try_filter_map(|x| ready(Ok(x.choices.into_iter().next())))
            .map_ok(|x| x.text);
    }
}
