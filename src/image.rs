use super::error::{Error, Result};
use crate::error_to_io_error;
use base64::Engine;
use bytes::Bytes;
use chrono::{DateTime, Utc};
use elor::{Either, LeftRight};
use futures::{stream::FuturesUnordered, StreamExt, TryFutureExt, TryStream, TryStreamExt};
use image::{
    codecs::png::PngDecoder, ExtendedColorType, GenericImage, GenericImageView, ImageBuffer,
    ImageDecoder, ImageFormat, ImageOutputFormat, Rgba,
};
use image::{io::Reader as ImageReader, DynamicImage};
use rand::{distributions::Standard, thread_rng, Rng};
use reqwest::Body;
use serde::{Deserialize, Serialize};
use std::{
    future::ready,
    io::{Cursor, Read, Seek, SeekFrom},
    ops::Deref,
    panic::resume_unwind,
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::task::spawn_blocking;
use tokio_util::io::StreamReader;

pub mod edit;
pub mod generate;
pub mod variation;

/// Result from an images request
#[derive(Debug, Clone, Deserialize)]
pub struct Images {
    #[serde(with = "chrono::serde::ts_seconds")]
    pub created: DateTime<Utc>,
    pub data: Vec<Data>,
}

/// The size of the generated images.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize)]
pub enum Size {
    /// 256-by-256 pixels
    #[serde(rename = "256x256")]
    P256,
    /// 512-by-512 pixels
    #[serde(rename = "512x512")]
    P512,
    /// 1024-by-1024 pixels
    #[serde(rename = "1024x1024")]
    #[default]
    P1024,
}

/// The format in which the generated images are returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormat {
    /// URL that points to an image hosted by OpenAI
    #[default]
    Url,
    /// Base64-encoded image data
    #[serde(rename = "b64_json")]
    B64Json,
}

/// Image data
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Data {
    /// URL that points to an image hosted by OpenAI
    Url(String),
    /// Base64-encoded image data
    #[serde(rename = "b64_json")]
    B64Json(Arc<String>),
}

impl Images {
    /// Saves all the images in the response into the specified directory
    pub async fn save_at(self, path: impl AsRef<Path>) -> Result<()> {
        let mut rng = thread_rng();
        let path: &Path = path.as_ref();

        let fut = self.save(|_| {
            let id = rng.sample::<u64, _>(Standard);
            path.join(format!("{id}")).with_extension("jpg")
        });

        return fut.await;
    }

    /// Saves all the images in the response into the path provided for each one by `f`
    pub async fn save<F: FnMut(&Data) -> PathBuf>(self, mut f: F) -> Result<()> {
        let fut = futures::stream::iter(self.data.into_iter())
            .map(|data| {
                let path = f(&data);
                tokio::spawn(async move {
                    let mut w = tokio::fs::File::create(path).await?;
                    data.write_into_tokio(&mut w).await?;
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
    /// Returns the response's value as a [`str`] slice
    #[inline]
    pub fn as_str(&self) -> &str {
        match self {
            Data::Url(x) => x,
            Data::B64Json(x) => x,
        }
    }

    /// Returns a bytes [`Stream`](futures::Stream) with the contents of the image
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

    /// Returns an [`futures::io::AsyncBufRead`] with the contents of the image
    pub async fn into_futures_reader(self) -> Result<impl futures::io::AsyncBufRead> {
        let stream = self.into_stream().await?.map_err(error_to_io_error);
        return Ok(stream.into_async_read());
    }

    /// Returns an [`tokio::io::AsyncBufRead`] with the contents of the image
    pub async fn into_tokio_reader(self) -> Result<impl tokio::io::AsyncBufRead> {
        let stream = self.into_stream().await?.map_err(error_to_io_error);
        return Ok(StreamReader::new(stream));
    }

    /// Writes the image's content into the specified [`tokio::io::AsyncWrite`] writer
    pub async fn write_into_tokio<W: ?Sized + Unpin + tokio::io::AsyncWrite>(
        self,
        w: &mut W,
    ) -> Result<()> {
        let reader = self.into_tokio_reader().await?;
        futures::pin_mut!(reader);
        tokio::io::copy_buf(&mut reader, w).await?;
        return Ok(());
    }

    /// Writes the image's content into the specified [`futures::io::AsyncWrite`] writer
    pub async fn write_into_futures<W: ?Sized + Unpin + futures::io::AsyncWrite>(
        self,
        w: &mut W,
    ) -> Result<()> {
        let reader = self.into_futures_reader().await?;
        futures::pin_mut!(reader);
        futures::io::copy_buf(&mut reader, w).await?;
        return Ok(());
    }
}

/// Loads the image from `path` and transforms it into a format valid to be sent to an OpenAI endpoint.
///
/// If the image is already in a valid format, no conversion will be done and it's byte stream will be directly returned.
///
/// > **Note**: This is a **blocking** method and should not be used in async contexts
pub fn load_image(path: impl AsRef<Path>) -> Result<Body> {
    let mut image = std::fs::File::open(path)?;

    // Read file magic number and seek back to start
    let mut magic = [0; 8];
    image.read_exact(&mut magic)?;
    image.seek(SeekFrom::Start(0))?;

    return match image::guess_format(&magic) {
        // Image is a PNG
        Ok(ImageFormat::Png) => {
            let decoder = PngDecoder::new(&mut image)?;
            let (width, height) = decoder.dimensions();

            // Make image square (by adding transparent background)
            if width != height {
                let size = u32::max(width, height);
                let image = DynamicImage::from_decoder(decoder)?;

                let mut extended = ImageBuffer::<Rgba<u8>, _>::new(size, size);
                extended.copy_from(&image, (size - width) / 2, (size - height) / 2)?;

                let mut result = Cursor::new(Vec::new());
                extended.write_to(&mut result, ImageOutputFormat::Png)?;
                return Ok(Body::from(result.into_inner()));
            }

            // Check image color type
            match decoder.original_color_type() {
                // Image has RGBA color, pass directly for streaming.
                ExtendedColorType::Rgba8 => {
                    image.seek(SeekFrom::Start(0))?;
                    Ok(Body::from(tokio::fs::File::from_std(image)))
                }

                // Transform image to RGBA PNG
                _ => {
                    let image = DynamicImage::from_decoder(decoder)?.to_rgba8();
                    let mut result = Cursor::new(Vec::new());
                    image.write_to(&mut result, ImageOutputFormat::Png)?;
                    Ok(Body::from(result.into_inner()))
                }
            }
        }

        // Image isn't a PNG
        _ => {
            let image = ImageReader::new(std::io::BufReader::new(image))
                .with_guessed_format()?
                .decode()?;
            let (width, height) = image.dimensions();

            // Make image square (by adding transparent background)
            if width != height {
                let size = u32::max(width, height);

                let mut extended = ImageBuffer::<Rgba<u8>, _>::new(size, size);
                extended.copy_from(&image, (size - width) / 2, (size - height) / 2)?;

                let mut result = Cursor::new(Vec::new());
                extended.write_to(&mut result, ImageOutputFormat::Png)?;
                return Ok(Body::from(result.into_inner()));
            }

            let mut output = Cursor::new(Vec::new());
            image
                .into_rgba8()
                .write_to(&mut output, ImageOutputFormat::Png)?;

            Result::<Body>::Ok(Body::from(output.into_inner()))
        }
    };
}
