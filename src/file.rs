use crate::{
    error::{FallibleResponse, Result},
    Str,
};
use chrono::{DateTime, Utc};
use rand::random;
use reqwest::{
    multipart::{Form, Part},
    Client, Response,
};
use serde::Deserialize;
use std::{ffi::OsStr, path::Path};

/// Files are used to upload documents that can be used with features like **Fine-tuning**.
#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct File {
    pub id: String,
    pub object: String,
    pub bytes: u64,
    #[serde(with = "chrono::serde::ts_seconds")]
    pub created_at: DateTime<Utc>,
    pub filename: String,
    pub purpose: String,
}

#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct Delete {
    pub id: String,
    pub object: String,
    pub deleted: bool,
}

impl File {
    /// Returns information about a specific file.
    pub async fn retreive(id: impl AsRef<str>, api_key: impl AsRef<str>) -> Result<Self> {
        let client = Client::new();
        let file = client
            .get(format!("https://api.openai.com/v1/files/{}", id.as_ref()))
            .bearer_auth(api_key.as_ref())
            .send()
            .await?
            .json::<FallibleResponse<Self>>()
            .await?
            .into_result()?;

        return Ok(file);
    }

    /// Returns the contents of the file
    #[inline]
    pub async fn content(&self, api_key: impl AsRef<str>) -> Result<Response> {
        return retreive_file_content(&self.id, api_key).await;
    }

    /// Delete the file.
    #[inline]
    pub async fn delete(self, api_key: impl AsRef<str>) -> Result<Delete> {
        return delete_file(self.id, api_key).await;
    }
}

/// Returns the contents of the specified file
pub async fn retreive_file_content(
    id: impl AsRef<str>,
    api_key: impl AsRef<str>,
) -> Result<Response> {
    let client = Client::new();
    let content = client
        .get(format!(
            "https://api.openai.com/v1/files/{}/content",
            id.as_ref()
        ))
        .bearer_auth(api_key.as_ref())
        .send()
        .await?;

    return Ok(content);
}

/// Delete a file.
pub async fn delete_file(id: impl AsRef<str>, api_key: impl AsRef<str>) -> Result<Delete> {
    let client = Client::new();
    let delete = client
        .delete(format!("https://api.openai.com/v1/files/{}", id.as_ref()))
        .bearer_auth(api_key.as_ref())
        .send()
        .await?
        .json::<FallibleResponse<Delete>>()
        .await?
        .into_result()?;

    return Ok(delete);
}

/// Upload a file that contains document(s) to be used across various endpoints/features. Currently, the size of all the files uploaded by one organization can be up to 1 GB.
pub async fn upload_file(
    file: impl AsRef<Path>,
    purpose: impl Into<Str<'static>>,
    api_key: impl AsRef<str>,
) -> Result<File> {
    let path: &Path = file.as_ref();
    let filename = match path.file_name().map(OsStr::to_string_lossy) {
        Some(x) => x.into_owned(),
        None => format!("{}.jsonl", random::<u64>()),
    };

    let file = Part::stream(tokio::fs::File::open(path).await?).file_name(filename);
    return upload_file_with_part(file, purpose, api_key).await;
}

/// Upload a file that contains document(s) to be used across various endpoints/features. Currently, the size of all the files uploaded by one organization can be up to 1 GB.
pub async fn upload_file_with_part(
    file: Part,
    purpose: impl Into<Str<'static>>,
    api_key: impl AsRef<str>,
) -> Result<File> {
    let client = Client::new();
    let body = Form::new().text("purpose", purpose).part("file", file);

    let file = client
        .post("https://api.openai.com/v1/files")
        .bearer_auth(api_key.as_ref())
        .multipart(body)
        .send()
        .await?
        .json::<FallibleResponse<File>>()
        .await?
        .into_result()?;

    return Ok(file);
}

/// Returns a list of files that belong to the user's organization.
pub async fn files(api_key: impl AsRef<str>) -> Result<Vec<File>> {
    #[derive(Debug, Deserialize)]
    struct Response {
        data: Vec<File>,
    }

    let client = Client::new();
    let files = client
        .get("https://api.openai.com/v1/files")
        .bearer_auth(api_key.as_ref())
        .send()
        .await?
        .json::<FallibleResponse<Response>>()
        .await?
        .into_result()?;

    return Ok(files.data);
}
