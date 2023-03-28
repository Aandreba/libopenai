use serde::Deserialize;
use std::fmt::{Debug, Display};

pub type Result<T, E = Error> = ::core::result::Result<T, E>;

/// An error returned by a builder
#[derive(Debug)]
pub struct BuilderError<T> {
    pub builder: T,
    pub err: Error,
}

/// An error of the `libopenai` library
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("OpenAI error: {0}")]
    OpenAI(#[from] OpenAiError),
    #[error("Enviroment error: {0}")]
    Env(#[from] std::env::VarError),
    #[error("Reqwest error: {0}")]
    Reqwest(#[from] reqwest::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Base64 error: {0}")]
    Base64(#[from] base64::DecodeError),
    #[error("Image error: {0}")]
    Image(#[from] image::error::ImageError),
    #[error("Srt error: {0}")]
    Srt(#[from] srtlib::ParsingError),
    #[error("Unknown error: {0}")]
    Other(#[from] anyhow::Error),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(crate) enum FallibleResponse<T> {
    Ok(T),
    Err { error: OpenAiError },
}

/// Error returned by an OpenAI API's endpoint
#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct OpenAiError {
    pub message: String,
    #[serde(rename = "type")]
    pub ty: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

impl Error {
    /// Creates a generic error with a custom message
    #[inline]
    pub fn msg<M: Display + Debug + Send + Sync + 'static>(msg: M) -> Self {
        Self::Other(anyhow::Error::msg(msg))
    }
}

impl<T> BuilderError<T> {
    #[inline]
    pub fn new(builder: T, error: impl Into<Error>) -> Self {
        return Self {
            builder,
            err: error.into(),
        };
    }

    #[inline]
    pub fn msg<M>(builder: T, msg: M) -> Self
    where
        M: Display + Debug + Send + Sync + 'static,
    {
        return Self {
            builder,
            err: Error::msg(msg),
        };
    }

    /// Returns the inner builder
    #[inline]
    pub fn into_inner(self) -> T {
        self.builder
    }

    /// Returns the inner error
    #[inline]
    pub fn into_error(self) -> Error {
        self.err
    }
}

impl<T> FallibleResponse<T> {
    #[inline]
    pub fn into_result(self) -> Result<T, OpenAiError> {
        match self {
            FallibleResponse::Ok(x) => Ok(x),
            FallibleResponse::Err { error } => {
                #[cfg(feature = "tracing")]
                tracing::error!("OpenAI error: {error}");
                Err(error)
            }
        }
    }
}

impl<T> From<BuilderError<T>> for Error {
    #[inline]
    fn from(value: BuilderError<T>) -> Self {
        value.into_error()
    }
}

impl Display for OpenAiError {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.message, f)
    }
}

impl<T> Display for BuilderError<T> {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.err, f)
    }
}

impl std::error::Error for OpenAiError {}
impl<T: Debug> std::error::Error for BuilderError<T> {
    #[inline]
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.err)
    }
}
