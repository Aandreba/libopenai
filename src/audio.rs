use serde::Serialize;

/// Transcribes audio into the input language.
pub mod transcription;
/// Translates audio into into English.
pub mod translation;

/// The format of the transcript/translation output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormat {
    #[default]
    Json,
    Text,
    Srt,
    VerboseJson,
    Vtt,
}
