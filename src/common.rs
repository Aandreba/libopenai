use std::collections::HashMap;

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct Usage {
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: Option<u64>,
    pub total_tokens: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Logprobs {
    pub tokens: Vec<String>,
    pub token_logprobs: Vec<f64>,
    pub top_logprobs: Vec<HashMap<String, f64>>,
    pub text_offset: Vec<u64>,
}
