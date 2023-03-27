use crate::{
    error::{FallibleResponse, Result},
    Client, Str,
};
use bytes::Bytes;
use chrono::{DateTime, Utc};
use futures::{ready, Stream, StreamExt, TryStream, TryStreamExt};
use rand::random;
use reqwest::{
    multipart::{Form, Part},
    Body, Response,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    collections::VecDeque, ffi::OsStr, future::ready, marker::PhantomData, path::Path, pin::Pin,
    task::Poll,
};
use tokio_util::io::ReaderStream;

/// Files are used to upload documents that can be used with features like **Fine-tuning**.
#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct File {
    pub id: String,
    pub bytes: u64,
    #[serde(with = "chrono::serde::ts_seconds")]
    pub created_at: DateTime<Utc>,
    pub filename: String,
    pub purpose: String,
}

/// Result of deleting a file
#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct Delete {
    pub id: String,
    pub deleted: bool,
}

pin_project_lite::pin_project! {
    struct Contents<S, T> {
        #[pin]
        stream: S,
        buf: VecDeque<u8>,
        _phtm: PhantomData<T>,
    }
}

impl File {
    /// Upload a file that contains document(s) to be used across various endpoints/features. Currently, the size of all the files uploaded by one organization can be up to 1 GB.
    pub async fn upload(
        file: impl AsRef<Path>,
        purpose: impl Into<Str<'static>>,
        client: impl AsRef<Client>,
    ) -> Result<Self> {
        let path: &Path = file.as_ref();
        let filename = match path.file_name().map(OsStr::to_string_lossy) {
            Some(x) => x.into_owned(),
            None => format!("{}.jsonl", random::<u64>()),
        };

        let file = Part::stream(tokio::fs::File::open(path).await?).file_name(filename);
        return Self::upload_part(file, purpose, client).await;
    }

    /// Upload a file that contains document(s) to be used across various endpoints/features. Currently, the size of all the files uploaded by one organization can be up to 1 GB.
    pub async fn upload_byte_stream<S>(
        stream: S,
        filename: Option<String>,
        purpose: impl Into<Str<'static>>,
        client: impl AsRef<Client>,
    ) -> Result<Self>
    where
        S: futures::stream::TryStream + Send + Sync + 'static,
        S::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
        Bytes: From<S::Ok>,
    {
        let filename = filename.unwrap_or_else(|| format!("{}.jsonl", random::<u64>()));
        let file = Part::stream(Body::wrap_stream(stream)).file_name(filename);
        return Self::upload_part(file, purpose, client).await;
    }

    /// Upload a file that contains document(s) to be used across various endpoints/features. Currently, the size of all the files uploaded by one organization can be up to 1 GB.
    pub async fn upload_stream<T, S>(
        stream: S,
        filename: Option<String>,
        purpose: impl Into<Str<'static>>,
        client: impl AsRef<Client>,
    ) -> Result<Self>
    where
        T: Serialize,
        S: 'static + Send + Sync + Stream<Item = T>,
    {
        let stream = stream.map(|x| {
            serde_json::to_string(&x).map(|mut x| {
                x.push('\n');
                x.into_bytes()
            })
        });
        return Self::upload_byte_stream(stream, filename, purpose, client).await;
    }

    /// Upload a file that contains document(s) to be used across various endpoints/features. Currently, the size of all the files uploaded by one organization can be up to 1 GB.
    pub async fn try_upload_stream<T, S>(
        stream: S,
        filename: Option<String>,
        purpose: impl Into<Str<'static>>,
        client: impl AsRef<Client>,
    ) -> Result<Self>
    where
        S: 'static + Send + Sync + TryStream<Ok = T>,
        S::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
        T: Serialize,
    {
        let stream = stream.map_err(Into::into).and_then(|x| {
            ready({
                match serde_json::to_string(&x) {
                    Ok(mut x) => {
                        x.push('\n');
                        Ok(x.into_bytes())
                    }
                    Err(e) => Err(e.into()),
                }
            })
        });
        return Self::upload_byte_stream(stream, filename, purpose, client).await;
    }

    /// Upload a file that contains document(s) to be used across various endpoints/features. Currently, the size of all the files uploaded by one organization can be up to 1 GB.
    pub async fn upload_tokio_reader<R>(
        reader: R,
        filename: Option<String>,
        purpose: impl Into<Str<'static>>,
        client: impl AsRef<Client>,
    ) -> Result<Self>
    where
        R: 'static + Send + Sync + tokio::io::AsyncRead,
    {
        return Self::upload_byte_stream(ReaderStream::new(reader), filename, purpose, client)
            .await;
    }

    /// Upload a file that contains document(s) to be used across various endpoints/features. Currently, the size of all the files uploaded by one organization can be up to 1 GB.
    pub async fn upload_part(
        file: Part,
        purpose: impl Into<Str<'static>>,
        client: impl AsRef<Client>,
    ) -> Result<Self> {
        let body = Form::new().text("purpose", purpose).part("file", file);
        let file = client
            .as_ref()
            .post("https://api.openai.com/v1/files")
            .multipart(body)
            .send()
            .await?
            .json::<FallibleResponse<File>>()
            .await?
            .into_result()?;

        return Ok(file);
    }

    /// Returns information about a specific file.
    pub async fn retreive(id: impl AsRef<str>, client: impl AsRef<Client>) -> Result<Self> {
        let file = client
            .as_ref()
            .get(format!("https://api.openai.com/v1/files/{}", id.as_ref()))
            .send()
            .await?
            .json::<FallibleResponse<Self>>()
            .await?
            .into_result()?;

        return Ok(file);
    }

    /// Returns the contents of the file.
    #[inline]
    pub async fn content<T: DeserializeOwned>(
        &self,
        client: impl AsRef<Client>,
    ) -> Result<impl Stream<Item = Result<T>>> {
        let content = retreive_raw_file_content(&self.id, client).await?;
        return Ok(Contents {
            stream: content.bytes_stream(),
            buf: VecDeque::new(),
            _phtm: PhantomData,
        });
    }

    /// Returns the contents of the file.
    #[inline]
    pub async fn raw_content(&self, client: impl AsRef<Client>) -> Result<Response> {
        return retreive_raw_file_content(&self.id, client).await;
    }

    /// Delete the file.
    #[inline]
    pub async fn delete(self, client: impl AsRef<Client>) -> Result<Delete> {
        return delete_file(self.id, client).await;
    }
}

impl<S: Stream<Item = reqwest::Result<Bytes>>, T: DeserializeOwned> Stream for Contents<S, T> {
    type Item = Result<T>;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let mut this = self.project();

        loop {
            if let Some((idx, _)) = this.buf.iter().enumerate().find(|(_, &x)| x == b'\n') {
                let mut line = this.buf.split_off(idx);
                let item = serde_json::from_slice::<T>(line.make_contiguous())?;
                return Poll::Ready(Some(Ok(item)));
            }

            match ready!(this.stream.as_mut().poll_next(cx)) {
                Some(Ok(x)) => this.buf.extend(x),
                Some(Err(e)) => return Poll::Ready(Some(Err(e.into()))),
                None => return Poll::Ready(None),
            }
        }
    }
}

/// Returns the contents of the specified file
pub async fn retreive_file_content<T: DeserializeOwned>(
    id: impl AsRef<str>,
    client: impl AsRef<Client>,
) -> Result<impl Stream<Item = Result<T>>> {
    let content = retreive_raw_file_content(id, client).await?;
    return Ok(Contents {
        stream: content.bytes_stream(),
        buf: VecDeque::new(),
        _phtm: PhantomData,
    });
}

/// Returns the contents of the specified file
pub async fn retreive_raw_file_content(
    id: impl AsRef<str>,
    client: impl AsRef<Client>,
) -> Result<Response> {
    let content = client
        .as_ref()
        .get(format!(
            "https://api.openai.com/v1/files/{}/content",
            id.as_ref()
        ))
        .send()
        .await?;

    return Ok(content);
}

/// Delete a file.
pub async fn delete_file(id: impl AsRef<str>, client: impl AsRef<Client>) -> Result<Delete> {
    let delete = client
        .as_ref()
        .delete(format!("https://api.openai.com/v1/files/{}", id.as_ref()))
        .send()
        .await?
        .json::<FallibleResponse<Delete>>()
        .await?
        .into_result()?;

    return Ok(delete);
}

/// Returns a list of files that belong to the user's organization.
pub async fn files(client: impl AsRef<Client>) -> Result<Vec<File>> {
    #[derive(Debug, Deserialize)]
    struct Response {
        data: Vec<File>,
    }

    let files = client
        .as_ref()
        .get("https://api.openai.com/v1/files")
        .send()
        .await?
        .json::<FallibleResponse<Response>>()
        .await?
        .into_result()?;

    return Ok(files.data);
}
