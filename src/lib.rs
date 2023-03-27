use std::{
    borrow::Cow,
    ops::{Deref, DerefMut},
};

use error::{Error, Result};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};

pub(crate) type Str<'a> = Cow<'a, str>;

pub mod audio;
pub mod chat;
pub mod common;
pub mod completion;
pub mod embeddings;
pub mod error;
pub mod file;
pub mod image;
pub mod model;
pub mod moderations;
// pub mod edit;
//pub mod finetune;

pub mod prelude {
    use super::*;

    pub use audio::transcription::Transcription;
    pub use audio::translation::Translation;

    pub type ChatCompletion = chat::Completion;
    pub type ChatCompletionStream = chat::CompletionStream;
    pub use chat::Message;

    pub use completion::Choice;
    pub use completion::Completion;
    pub use completion::CompletionStream;

    // pub use edit::Edit;

    pub use error::Error;

    pub use super::image::Data;
    pub use super::image::Images;

    pub use model::models;
    pub use model::Model;

    pub use moderations::Moderation;
}

#[derive(Debug, Clone)]
pub struct Client(reqwest::Client);

impl Client {
    #[inline]
    pub fn new(api_key: Option<&str>) -> Result<Self> {
        Self::from_builder(Default::default(), api_key)
    }

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
