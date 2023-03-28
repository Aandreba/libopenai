use crate::{error::Result, Str};
use elor::Either;
use serde::{Deserialize, Serialize};
use srtlib::{Subtitle, Subtitles, Timestamp};
use std::time::Duration;

/// Transcribes audio into the input language.
pub mod transcription;
/// Translates audio into English.
pub mod translation;

/// The format of the transcript/translation output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum AudioResponseFormat {
    #[default]
    Json,
    Text,
    Srt,
    VerboseJson,
    // Vtt,
}

/// Response to a transcript/translation
#[derive(Debug)]
#[non_exhaustive]
pub enum AudioResponse {
    Json(JsonResponse),
    Text(String),
    Srt(Vec<Subtitle>),
    VerboseJson(VerboseJsonResponse),
    // Vtt
}

/// A generic segment, independent of [response format](AudioResponseFormat)
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GenericSegment<'a> {
    pub text: &'a str,
    pub start: Duration,
    pub end: Duration,
}

/// Response for [`Json`](AudioResponseFormat::Json) response format
#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct JsonResponse {
    pub text: String,
}

/// Response for [`VerboseJson`](AudioResponseFormat::VerboseJson) response format
#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct VerboseJsonResponse {
    pub task: String,
    pub language: String,
    #[serde(deserialize_with = "crate::deserialize_duration_secs")]
    pub duration: Duration,
    pub segments: Vec<VerboseJsonSegment>,
    pub text: String,
}

/// A [`VerboseJson`](AudioResponseFormat::VerboseJson) response segment
#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct VerboseJsonSegment {
    pub id: u64,
    pub seek: u64,
    #[serde(deserialize_with = "crate::deserialize_duration_secs")]
    pub start: Duration,
    #[serde(deserialize_with = "crate::deserialize_duration_secs")]
    pub end: Duration,
    pub text: String,
    pub tokens: Vec<u64>,
    pub temperature: f64,
    pub avg_logprob: f64,
    pub compression_ratio: f64,
    pub no_speech_prob: f64,
    pub transient: bool,
}

impl AudioResponse {
    /// Returns the underlying text response
    #[inline]
    pub fn text(&self) -> Str<'_> {
        match self {
            AudioResponse::Json(x) => Str::Borrowed(&x.text),
            AudioResponse::Text(x) => Str::Borrowed(x),
            AudioResponse::Srt(lines) => {
                let mut result = String::new();
                let mut lines = lines.iter().peekable();

                while let Some(line) = lines.next() {
                    result.push_str(&line.text);
                    if lines.peek().is_some() {
                        result.push(' ');
                    }
                }

                Str::Owned(result)
            }
            AudioResponse::VerboseJson(x) => Str::Borrowed(&x.text),
        }
    }

    /// Returns the language of the audio
    #[inline]
    pub fn language(&self) -> Option<&str> {
        match self {
            AudioResponse::VerboseJson(x) => Some(&x.language),
            _ => None,
        }
    }

    /// Returns the duration of the audio
    #[inline]
    pub fn duration(&self) -> Option<Duration> {
        match self {
            AudioResponse::VerboseJson(x) => Some(x.duration),
            AudioResponse::Srt(x) => match (x.first(), x.last()) {
                (Some(start), Some(end)) => timestamp_to_duration(end.end_time)
                    .checked_sub(timestamp_to_duration(start.start_time)),
                _ => Some(Duration::ZERO),
            },
            _ => None,
        }
    }

    /// Returns an iterator over the [`GenericSegment`] (the segments of the response)
    pub fn segments(&self) -> Option<impl Iterator<Item = GenericSegment<'_>>> {
        match self {
            AudioResponse::VerboseJson(x) => Some(
                Either::Left(x.segments.iter().map(|x| GenericSegment {
                    start: x.start,
                    end: x.end,
                    text: &x.text,
                }))
                .into_same_iter(),
            ),
            AudioResponse::Srt(x) => Some(
                Either::Right(x.iter().map(|x| GenericSegment {
                    start: timestamp_to_duration(x.start_time),
                    end: timestamp_to_duration(x.end_time),
                    text: &x.text,
                }))
                .into_same_iter(),
            ),
            _ => None,
        }
    }
}

impl GenericSegment<'_> {
    /// Returns the duration of the segment
    #[inline]
    pub fn duration(self) -> Duration {
        self.end - self.start
    }
}

/// Parses a [`reqwest::Response`] into a response of the specified format.
pub async fn parse_audio_response(
    resp: reqwest::Response,
    format: AudioResponseFormat,
) -> Result<AudioResponse> {
    return match format {
        AudioResponseFormat::Json => Ok(AudioResponse::Json(resp.json::<JsonResponse>().await?)),
        AudioResponseFormat::Text => Ok(AudioResponse::Text(resp.text().await?)),
        AudioResponseFormat::Srt => {
            let text = resp.text().await?;
            Ok(AudioResponse::Srt(
                Subtitles::parse_from_str(text)?.to_vec(),
            ))
        }
        AudioResponseFormat::VerboseJson => Ok(AudioResponse::VerboseJson(
            resp.json::<VerboseJsonResponse>().await?,
        )),
        // AudioResponseFormat::Vtt => Err(Error::msg("Vtt is currently unsuported")),
    };
}

#[inline]
fn timestamp_to_duration(ts: Timestamp) -> Duration {
    let (h, m, s, ms) = ts.get();
    let millis = (ms as u64) + 1000 * (s as u64) + 60000 * (m as u64) + 3600000 * (h as u64);
    Duration::from_millis(millis)
}
