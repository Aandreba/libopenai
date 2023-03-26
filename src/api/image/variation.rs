use super::{Images, ResponseFormat, Size};
use crate::api::error::{Error, FallibleResponse, Result};
use bytes::{Bytes, BytesMut};
use futures::TryStream;
use image::codecs::png::PngDecoder;
use image::io::Reader as ImageReader;
use image::{DynamicImage, ImageDecoder, ImageFormat, ImageOutputFormat};
use rand::random;
use reqwest::{
    multipart::{Form, Part},
    Body, Client,
};
use std::io::Cursor;
use std::mem::MaybeUninit;
use std::path::PathBuf;
use std::{
    ffi::OsStr,
    io::{Read, Seek, SeekFrom},
    ops::RangeInclusive,
};
use tokio::task::spawn_blocking;
use tokio_util::io::ReaderStream;

#[derive(Debug, Clone)]
pub struct Builder {
    n: Option<u32>,
    size: Option<Size>,
    response_format: Option<ResponseFormat>,
    user: Option<String>,
}

impl Images {
    #[inline]
    pub fn variation() -> Builder {
        return Builder::new();
    }
}

impl Builder {
    #[inline]
    pub fn new() -> Self {
        return Self {
            n: None,
            size: None,
            response_format: None,
            user: None,
        };
    }

    #[inline]
    pub fn n(mut self, n: u32) -> Result<Self, Self> {
        const RANGE: RangeInclusive<u32> = 1..=10;
        return match RANGE.contains(&n) {
            true => {
                self.n = Some(n);
                Ok(self)
            }
            false => Err(self),
        };
    }

    #[inline]
    pub fn size(mut self, size: Size) -> Self {
        self.size = Some(size);
        self
    }

    #[inline]
    pub fn response_format(mut self, response_format: ResponseFormat) -> Self {
        self.response_format = Some(response_format);
        self
    }

    #[inline]
    pub fn user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    pub async fn with_file(
        self,
        image: impl Into<PathBuf>,
        api_key: impl AsRef<str>,
    ) -> Result<Images> {
        let image_path: PathBuf = image.into();
        let my_image_path = image_path.clone();

        let image = spawn_blocking(move || {
            let mut image = std::fs::File::open(my_image_path)?;

            // Read file magic number and seek back to start
            let mut magic = [0; 8];
            image.read_exact(&mut magic)?;
            image.seek(SeekFrom::Start(0))?;

            return match image::guess_format(&magic) {
                Ok(ImageFormat::Png) => {
                    let decoder = PngDecoder::new(&mut image)?;
                    match decoder.color_type() {
                        image::ColorType::Rgba8
                        | image::ColorType::Rgba16
                        | image::ColorType::Rgba32F => {
                            Ok(Body::from(tokio::fs::File::from_std(image)))
                        }
                        _ => {
                            let len = match usize::try_from(decoder.total_bytes()) {
                                Ok(len) => len,
                                Err(e) => return Err(Error::Other(anyhow::Error::new(e))),
                            };

                            let mut bytes = vec![0; len];
                            decoder.read_image(&mut bytes)?;
                            Ok(Body::from(Bytes::from(bytes)))
                        }
                    }
                }
                _ => {
                    let mut output = Vec::new();
                    match ImageReader::new(std::io::BufReader::new(image))
                        .with_guessed_format()?
                        .decode()?
                    {
                        DynamicImage::ImageRgba8(x) => {
                            x.write_to(&mut Cursor::new(&mut output), ImageOutputFormat::Png)
                        }
                        DynamicImage::ImageRgba16(x) => {
                            x.write_to(&mut Cursor::new(&mut output), ImageOutputFormat::Png)
                        }
                        DynamicImage::ImageRgba32F(x) => {
                            x.write_to(&mut Cursor::new(&mut output), ImageOutputFormat::Png)
                        }
                        x => x
                            .to_rgba8()
                            .write_to(&mut Cursor::new(&mut output), ImageOutputFormat::Png),
                    }?;

                    Result::<Body>::Ok(Body::from(output))
                }
            };
        })
        .await
        .unwrap()?;

        let name = match image_path.file_name().map(OsStr::to_string_lossy) {
            Some(x) => x.into_owned(),
            None => format!("{}.png", random::<u64>()),
        };

        let image = Part::stream(image).file_name(name);
        return self.with_part(image, api_key).await;
    }

    pub async fn with_tokio_reader<I>(self, image: I, api_key: impl AsRef<str>) -> Result<Images>
    where
        I: 'static + Send + Sync + tokio::io::AsyncRead,
    {
        return self.with_stream(ReaderStream::new(image), api_key).await;
    }

    pub async fn with_stream<I>(self, image: I, api_key: impl AsRef<str>) -> Result<Images>
    where
        I: TryStream + Send + Sync + 'static,
        I::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
        Bytes: From<I::Ok>,
    {
        return self.with_body(Body::wrap_stream(image), api_key).await;
    }

    pub async fn with_body(
        self,
        image: impl Into<Body>,
        api_key: impl AsRef<str>,
    ) -> Result<Images> {
        return self
            .with_part(
                Part::stream(image).file_name(format!("{}.png", random::<u64>())),
                api_key,
            )
            .await;
    }

    pub async fn with_part(self, image: Part, api_key: impl AsRef<str>) -> Result<Images> {
        let client = Client::new();

        let mut body = Form::new().part("image", image);

        if let Some(n) = self.n {
            body = body.text("n", format!("{n}"))
        }
        if let Some(size) = self.size {
            body = body.text(
                "size",
                match serde_json::to_value(&size)? {
                    serde_json::Value::String(x) => x,
                    _ => return Err(Error::msg("Unexpected error")),
                },
            )
        }
        if let Some(response_format) = self.response_format {
            body = body.text(
                "response_format",
                match serde_json::to_value(&response_format)? {
                    serde_json::Value::String(x) => x,
                    _ => return Err(Error::msg("Unexpected error")),
                },
            )
        }
        if let Some(user) = self.user {
            body = body.text("user", user)
        }

        let resp = client
            .post("https://api.openai.com/v1/images/variations")
            .bearer_auth(api_key.as_ref())
            .multipart(body)
            .send()
            .await?
            .json::<FallibleResponse<Images>>()
            .await?
            .into_result()?;

        return Ok(resp);
    }
}
