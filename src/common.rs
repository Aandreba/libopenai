use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    pub total_tokens: u64,
}
