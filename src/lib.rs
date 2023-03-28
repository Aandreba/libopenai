#![cfg_attr(docsrs, feature(doc_cfg))]

use crate::error::OpenAiError;
use bytes::Bytes;
use error::{Error, Result};
use futures::{ready, Stream};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};
use serde::{
    de::{DeserializeOwned, Visitor},
    Deserialize, Deserializer,
};
use std::{
    borrow::Cow,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    pin::Pin,
    time::Duration,
};

pub(crate) type Str<'a> = Cow<'a, str>;

/// Learn how to turn audio into text.
pub mod audio;
/// Given a chat conversation, the model will return a chat completion response.
pub mod chat;
/// Structures and methods commonly used throughout the library
pub mod common;
/// Given a prompt, the model will return one or more predicted completions, and can also return the probabilities of alternative tokens at each position.
pub mod completion;
/// Given a prompt and an instruction, the model will return an edited version of the prompt.
pub mod edit;
/// Get a vector representation of a given input that can be easily consumed by machine learning models and algorithms.
pub mod embeddings;
/// Library's error types
pub mod error;
/// Files are used to upload documents that can be used with features like fine-tuning.
pub mod file;
/// Manage fine-tuning jobs to tailor a model to your specific training data.
pub mod finetune;
/// Given a prompt and/or an input image, the model will generate a new image.
pub mod image;
/// List and describe the various models available in the API.
pub mod model;
/// Given a input text, outputs if the model classifies it as violating OpenAI's content policy.
pub mod moderations;

pub mod prelude {
    use super::*;

    pub use audio::transcription::TranscriptionBuilder;
    pub use audio::translation::TranslationBuilder;

    pub use chat::ChatCompletion;
    pub use chat::ChatCompletionStream;
    pub use chat::Message;

    pub use completion::{Choice, Completion, CompletionStream};

    pub use edit::Edit;

    pub use embeddings::{Embedding, EmbeddingResult};

    pub use error::Error;

    pub use file::File;

    pub use finetune::{FineTune, FineTuneEvent};

    pub use super::image::ImageData;
    pub use super::image::Images;

    pub use model::models;
    pub use model::Model;

    pub use moderations::Moderation;
}

/// A client that's used to connect to the OpenAI API
#[derive(Debug, Clone)]
pub struct Client(reqwest::Client);

impl Client {
    /// Creates a new client with a default [`reqwest::Client`] (restricted to HTTPS requests only).
    ///
    /// If `api_key` is `None`, the key will be taken from the enviroment variable `OPENAI_API_KEY`
    #[inline]
    pub fn new(api_key: Option<&str>, organization: Option<&str>) -> Result<Self> {
        Self::from_builder(
            reqwest::ClientBuilder::new().https_only(true),
            api_key,
            organization,
        )
    }

    /// Creates a new client with the specified [`reqwest::ClientBuilder`].
    ///
    /// If `api_key` is `None`, the key will be taken from the enviroment variable `OPENAI_API_KEY`
    pub fn from_builder(
        builder: reqwest::ClientBuilder,
        api_key: Option<&str>,
        organization: Option<&str>,
    ) -> Result<Self> {
        let api_key = match api_key {
            Some(x) => Str::Borrowed(x),
            None => Str::Owned(std::env::var("OPENAI_API_KEY")?),
        };

        let mut headers = HeaderMap::new();

        let mut bearer = HeaderValue::try_from(format!("Bearer {api_key}"))
            .map_err(|e| Error::Other(e.into()))?;
        bearer.set_sensitive(true);
        headers.append(AUTHORIZATION, bearer);

        if let Some(organization) = organization {
            let organization =
                HeaderValue::from_str(organization).map_err(|e| Error::Other(e.into()))?;
            headers.append("OpenAI-Organization", organization);
        }

        let client = builder.default_headers(headers).build()?;
        return Ok(Self(client));
    }
}

impl AsRef<Client> for Client {
    #[inline]
    fn as_ref(&self) -> &Client {
        self
    }
}

impl Deref for Client {
    type Target = reqwest::Client;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Client {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pin_project_lite::pin_project! {
    /// A [`Stream`] of [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events), sent by OpenAI
    pub struct OpenAiStream<T> {
        #[pin]
        inner: Pin<Box<dyn 'static + Stream<Item = reqwest::Result<Bytes>> + Send + Sync>>,
        current_chunk: Option<Bytes>,
        _phtm: PhantomData<T>,
    }
}

// Stream doesn't actually hold any value of type `T`
unsafe impl<T> Send for OpenAiStream<T> {}
unsafe impl<T> Sync for OpenAiStream<T> {}

impl<T: DeserializeOwned> Stream for OpenAiStream<T> {
    type Item = Result<T>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        const DONE: &[u8] = b"[DONE]";

        #[derive(Debug, Deserialize)]
        struct ChunkError {
            error: OpenAiError,
        }

        fn process_line(current: &mut Bytes) -> Bytes {
            let mut idx = None;
            for (i, b) in current.windows(2).enumerate() {
                match b {
                    b"\n\n" => {
                        idx = Some(i);
                        break;
                    }
                    _ => {}
                }
            }

            let split = current.split_off(idx.unwrap_or(current.len())).slice(2..);
            core::mem::replace(current, split)
        }

        loop {
            let line = match self.current_chunk {
                Some(ref mut current) => {
                    let next = process_line(current);
                    if current.is_empty() {
                        self.current_chunk = None
                    }
                    next
                }
                None => match ready!(self.inner.as_mut().poll_next(cx)) {
                    Some(Ok(mut x)) => {
                        let next = process_line(&mut x);
                        self.current_chunk = match x.is_empty() {
                            true => None,
                            false => Some(x),
                        };
                        next
                    }
                    Some(Err(e)) => return std::task::Poll::Ready(Some(Err(e.into()))),
                    None => return std::task::Poll::Ready(None),
                },
            };

            let line = trim_ascii(&line);
            if line.is_empty() {
                continue;
            }

            // Check if chunk is error
            if let Ok(ChunkError { error }) = serde_json::from_slice::<ChunkError>(&line) {
                return std::task::Poll::Ready(Some(Err(Error::from(error))));
            }

            // remove initial "data"
            let line: &[u8] = trim_ascii_start(&line[5..]);
            if line.starts_with(DONE) {
                return std::task::Poll::Ready(None);
            }

            let json = serde_json::from_slice::<T>(line)?;
            return std::task::Poll::Ready(Some(Ok(json)));
        }
    }
}

#[inline]
pub(crate) fn trim_ascii(ascii: &[u8]) -> &[u8] {
    return trim_ascii_end(trim_ascii_start(ascii));
}

#[allow(unused)]
pub(crate) fn trim_ascii_start(mut ascii: &[u8]) -> &[u8] {
    loop {
        match ascii.first() {
            Some(&x) if x.is_ascii_whitespace() => ascii = &ascii[1..],
            _ => break,
        }
    }
    return ascii;
}

#[allow(unused)]
pub(crate) fn trim_ascii_end(mut ascii: &[u8]) -> &[u8] {
    loop {
        match ascii.last() {
            Some(&x) if x.is_ascii_whitespace() => ascii = &ascii[..ascii.len() - 1],
            _ => break,
        }
    }
    return ascii;
}

pub(crate) fn error_to_io_error(e: Error) -> std::io::Error {
    match e {
        Error::Io(e) => e,
        Error::Other(e) => match e.downcast::<std::io::Error>() {
            Ok(e) => e,
            Err(other) => std::io::Error::new(std::io::ErrorKind::Other, other),
        },
        other => std::io::Error::new(std::io::ErrorKind::Other, other),
    }
}

pub(crate) mod serde_trim_string {
    use serde::{Deserialize, Deserializer, Serializer};

    #[allow(unused)]
    #[inline]
    pub fn serialize<S: Serializer>(this: impl AsRef<str>, ser: S) -> Result<S::Ok, S::Error> {
        ser.serialize_str(this.as_ref().trim())
    }

    #[inline]
    pub fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<String, D::Error> {
        let str = String::deserialize(de)?;
        let trim = str.trim();

        return match str.len() == trim.len() {
            true => Ok(str),
            false => Ok(trim.to_string()),
        };
    }
}

#[inline]
pub(crate) fn deserialize_duration_secs<'de, D: Deserializer<'de>>(
    de: D,
) -> Result<Duration, D::Error> {
    struct LocalVisitor;

    impl<'a> Visitor<'a> for LocalVisitor {
        type Value = Duration;

        #[inline]
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "a duration in seconds")
        }

        #[inline]
        fn visit_u64<E>(self, v: u64) -> std::result::Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            return Ok(Duration::from_secs(v));
        }

        fn visit_f32<E>(self, v: f32) -> std::result::Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            return Ok(Duration::from_secs_f32(v));
        }

        fn visit_f64<E>(self, v: f64) -> std::result::Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            return Ok(Duration::from_secs_f64(v));
        }
    }

    de.deserialize_any(LocalVisitor)
}
