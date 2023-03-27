use std::{
    borrow::Cow,
    ops::{Deref, DerefMut},
};

use error::{Error, Result};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};

pub(crate) type Str<'a> = Cow<'a, str>;

/// Learn how to turn audio into text.
pub mod audio;
/// Given a chat conversation, the model will return a chat completion response.
pub mod chat;
/// Structures and methods commonly used throughout the library
pub mod common;
/// Given a prompt, the model will return one or more predicted completions, and can also return the probabilities of alternative tokens at each position.
pub mod completion;
/// Get a vector representation of a given input that can be easily consumed by machine learning models and algorithms.
pub mod embeddings;
/// Library's error types
pub mod error;
/// Files are used to upload documents that can be used with features like fine-tuning.
pub mod file;
/// Given a prompt and/or an input image, the model will generate a new image.
pub mod image;
/// List and describe the various models available in the API.
pub mod model;
/// Given a input text, outputs if the model classifies it as violating OpenAI's content policy.
pub mod moderations;
// pub mod edit;
//pub mod finetune;

pub mod prelude {
    use super::*;

    pub use audio::transcription::TranscriptionBuilder;
    pub use audio::translation::TranslationBuilder;

    pub use chat::ChatCompletion;
    pub use chat::ChatCompletionStream;
    pub use chat::Message;

    pub use completion::{Choice, Completion, CompletionStream};

    // pub use edit::Edit;

    pub use embeddings::{Embedding, EmbeddingResult};

    pub use error::Error;

    pub use file::File;

    pub use super::image::Data;
    pub use super::image::Images;

    pub use model::models;
    pub use model::Model;

    pub use moderations::Moderation;
}

/// A client that's used to connect to the OpenAI API
#[derive(Debug, Clone)]
pub struct Client(reqwest::Client);

impl Client {
    /// Creates a new client with a default [`reqwest::Client`].
    ///
    /// If `api_key` is `None`, the key will be taken from the enviroment variable `OPENAI_API_KEY`
    #[inline]
    pub fn new(api_key: Option<&str>) -> Result<Self> {
        Self::from_builder(Default::default(), api_key)
    }

    /// Creates a new client with the specified [`reqwest::ClientBuilder`].
    ///
    /// If `api_key` is `None`, the key will be taken from the enviroment variable `OPENAI_API_KEY`
    pub fn from_builder(builder: reqwest::ClientBuilder, api_key: Option<&str>) -> Result<Self> {
        let api_key = match api_key {
            Some(x) => Str::Borrowed(x),
            None => Str::Owned(std::env::var("OPENAI_API_KEY")?),
        };

        let mut bearer = HeaderValue::try_from(format!("Bearer {api_key}"))
            .map_err(|e| Error::Other(e.into()))?;
        bearer.set_sensitive(true);

        let mut headers = HeaderMap::new();
        headers.append(AUTHORIZATION, bearer);

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

#[inline]
#[allow(unused)]
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
