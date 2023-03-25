use super::{
    common::StreamTokioAsyncRead,
    error::{Error, Result},
};
use base64::Engine;
use bytes::Bytes;
use chrono::{DateTime, Utc};
use elor::{Either, LeftRight};
use futures::{stream::FuturesUnordered, StreamExt, TryFutureExt, TryStream, TryStreamExt};
use rand::{distributions::Standard, thread_rng, Rng};
use serde::{Deserialize, Serialize};
use std::{
    future::ready,
    ops::Deref,
    panic::resume_unwind,
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::task::spawn_blocking;

pub mod edit;
pub mod generate;
pub mod variation;

#[derive(Debug, Clone, Deserialize)]
pub struct Images {
    #[serde(with = "chrono::serde::ts_seconds")]
    pub created: DateTime<Utc>,
    pub data: Vec<Data>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize)]
pub enum Size {
    #[serde(rename = "256x256")]
    P256,
    #[serde(rename = "512x512")]
    P512,
    #[serde(rename = "1024x1024")]
    #[default]
    P1024,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormat {
    #[default]
    Url,
    #[serde(rename = "b64_json")]
    B64Json,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Data {
    Url(Arc<str>),
    #[serde(rename = "b64_json")]
    B64Json(Arc<str>),
}

impl Images {
    pub async fn save_at(self, path: impl AsRef<Path>) -> Result<()> {
        let mut rng = thread_rng();
        let path: &Path = path.as_ref();

        let fut = self.save(|_| {
            let id = rng.sample::<u64, _>(Standard);
            path.join(format!("{id}")).with_extension("jpg")
        });

        return fut.await;
    }

    pub async fn save<F: FnMut(&Data) -> PathBuf>(self, mut f: F) -> Result<()> {
        let fut = futures::stream::iter(self.data.into_iter())
            .map(|data| {
                let path = f(&data);
                tokio::spawn(async move {
                    let mut w = tokio::fs::File::create(path).await?;
                    data.save_into_tokio(&mut w).await?;
                    return Result::<()>::Ok(());
                })
            })
            .collect::<FuturesUnordered<_>>()
            .await;

        fut.map(|x| match x {
            Ok(x) => x,
            Err(e) => resume_unwind(e.into_panic()),
        })
        .try_collect::<()>()
        .await?;

        return Ok(());
    }
}

impl Data {
    #[inline]
    pub fn as_str(&self) -> &str {
        match self {
            Data::Url(x) => x,
            Data::B64Json(x) => x,
        }
    }

    pub async fn into_stream(self) -> Result<impl TryStream<Ok = Bytes, Error = Error>> {
        let v = match self {
            Data::Url(url) => Either::Left(
                reqwest::get(url.deref())
                    .await?
                    .bytes_stream()
                    .map_err(Error::from),
            ),
            Data::B64Json(x) => {
                let fut = async move {
                    match spawn_blocking(move || {
                        base64::engine::general_purpose::STANDARD.decode(x.deref())
                    })
                    .await
                    {
                        Ok(Ok(x)) => return Ok(futures::stream::once(ready(Ok(Bytes::from(x))))),
                        Ok(Err(e)) => return Err(Error::from(e)),
                        Err(e) => std::panic::resume_unwind(e.into_panic()),
                    }
                };
                Either::Right(fut.try_flatten_stream())
            }
        };

        return Ok(futures::stream::StreamExt::map(v, LeftRight::into_inner));
    }

    pub async fn into_futures_reader(self) -> Result<impl futures::io::AsyncBufRead> {
        let stream = self.into_stream().await?.map_err(|e| match e {
            Error::Io(e) => e,
            Error::Other(e) => match e.downcast::<std::io::Error>() {
                Ok(e) => e,
                Err(other) => std::io::Error::new(std::io::ErrorKind::Other, other),
            },
            other => std::io::Error::new(std::io::ErrorKind::Other, other),
        });

        return Ok(stream.into_async_read());
    }

    pub async fn into_tokio_reader(self) -> Result<impl tokio::io::AsyncBufRead> {
        let stream = self.into_stream().await?.map_err(|e| match e {
            Error::Io(e) => e,
            Error::Other(e) => match e.downcast::<std::io::Error>() {
                Ok(e) => e,
                Err(other) => std::io::Error::new(std::io::ErrorKind::Other, other),
            },
            other => std::io::Error::new(std::io::ErrorKind::Other, other),
        });

        return Ok(StreamTokioAsyncRead::new(stream));
    }

    pub async fn save_into_tokio<W: ?Sized + Unpin + tokio::io::AsyncWrite>(
        self,
        w: &mut W,
    ) -> Result<()> {
        let reader = self.into_tokio_reader().await?;
        futures::pin_mut!(reader);
        tokio::io::copy_buf(&mut reader, w).await?;
        return Ok(());
    }

    pub async fn save_into_futures<W: ?Sized + Unpin + futures::io::AsyncWrite>(
        self,
        w: &mut W,
    ) -> Result<()> {
        let reader = self.into_futures_reader().await?;
        futures::pin_mut!(reader);
        futures::io::copy_buf(&mut reader, w).await?;
        return Ok(());
    }
}
