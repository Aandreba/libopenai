use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct Usage {
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: Option<u32>,
    pub total_tokens: u32,
}
