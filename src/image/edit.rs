use super::{load_image, Images, ResponseFormat, Size};
use crate::{
    error::{BuilderError, Error, FallibleResponse, Result},
    Client,
};
use bytes::Bytes;
use futures::{future::try_join, FutureExt, TryStream};
use rand::{distributions::Standard, random, thread_rng, Rng};
use reqwest::{
    multipart::{Form, Part},
    Body,
};
use std::{ffi::OsStr, ops::RangeInclusive, path::PathBuf};
use tokio::task::spawn_blocking;
use tokio_util::io::ReaderStream;

#[derive(Debug, Clone)]
pub struct ImageEditBuilder {
    prompt: String,
    n: Option<u64>,
    size: Option<Size>,
    response_format: Option<ResponseFormat>,
    user: Option<String>,
}

impl Images {
    /// Creates an edited or extended image given an original image and a prompt.
    #[inline]
    pub fn edit(prompt: impl Into<String>) -> Result<ImageEditBuilder> {
        return ImageEditBuilder::new(prompt);
    }
}

impl ImageEditBuilder {
    #[inline]
    pub fn new(prompt: impl Into<String>) -> Result<Self> {
        let prompt: String = prompt.into();
        if prompt.len() > 1000 {
            return Err(Error::msg("Message excedes character limit of 1000"));
        }

        return Ok(Self {
            prompt: prompt.into(),
            n: None,
            size: None,
            response_format: None,
            user: None,
        });
    }

    /// The number of images to generate. Must be between 1 and 10.
    #[inline]
    pub fn n(mut self, n: u64) -> Result<Self, BuilderError<Self>> {
        const RANGE: RangeInclusive<u64> = 1..=10;
        return match RANGE.contains(&n) {
            true => {
                self.n = Some(n);
                Ok(self)
            }
            false => Err(BuilderError::msg(
                self,
                format!("n out of range ({RANGE:?})"),
            )),
        };
    }

    /// The size of the generated images.
    #[inline]
    pub fn size(mut self, size: Size) -> Self {
        self.size = Some(size);
        self
    }

    /// The format in which the generated images are returned.
    #[inline]
    pub fn response_format(mut self, response_format: ResponseFormat) -> Self {
        self.response_format = Some(response_format);
        self
    }

    /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
    #[inline]
    pub fn user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    /// Sends the request with the specified files.
    ///
    /// If the images do not conform to OpenAI's requirements, they will be adapted before they are sent
    pub async fn with_file(
        self,
        image: impl Into<PathBuf>,
        mask: Option<PathBuf>,
        client: impl AsRef<Client>,
    ) -> Result<Images> {
        let (image, mask) = match mask {
            Some(mask) => {
                let mut rng = thread_rng();
                let image: PathBuf = image.into();

                let image_name = match image.file_name().map(OsStr::to_string_lossy) {
                    Some(x) => x.into_owned(),
                    None => format!("{}.png", rng.sample::<u64, _>(Standard)),
                };
                let mask_name = match mask.file_name().map(OsStr::to_string_lossy) {
                    Some(x) => x.into_owned(),
                    None => format!("{}.png", rng.sample::<u64, _>(Standard)),
                };

                let (image, mask) = try_join(
                    spawn_blocking(move || load_image(image)).map(Result::unwrap),
                    spawn_blocking(move || load_image(mask)).map(Result::unwrap),
                )
                .await?;
                (
                    Part::stream(Body::from(image)).file_name(image_name),
                    Some(Part::stream(Body::from(mask)).file_name(mask_name)),
                )
            }
            None => {
                let image: PathBuf = image.into();
                let name = match image.file_name().map(OsStr::to_string_lossy) {
                    Some(x) => x.into_owned(),
                    None => format!("{}.png", random::<u64>()),
                };

                let image = spawn_blocking(move || load_image(image)).await.unwrap()?;
                (Part::stream(Body::from(image)).file_name(name), None)
            }
        };

        return self.with_part(image, mask, client).await;
    }

    /// Sends the request with the specified file.
    pub async fn with_tokio_reader<I>(self, image: I, client: impl AsRef<Client>) -> Result<Images>
    where
        I: 'static + Send + Sync + tokio::io::AsyncRead,
    {
        return self.with_stream(ReaderStream::new(image), client).await;
    }

    /// Sends the request with the specified file.
    pub async fn with_stream<I>(self, image: I, client: impl AsRef<Client>) -> Result<Images>
    where
        I: TryStream + Send + Sync + 'static,
        I::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
        Bytes: From<I::Ok>,
    {
        return self.with_body(Body::wrap_stream(image), None, client).await;
    }

    /// Sends the request with the specified files.
    pub async fn with_body(
        self,
        image: impl Into<Body>,
        mask: Option<Body>,
        client: impl AsRef<Client>,
    ) -> Result<Images> {
        let mut rng = thread_rng();

        return self
            .with_part(
                Part::stream(image).file_name(format!("{}.png", rng.sample::<u64, _>(Standard))),
                mask.map(|mask| {
                    Part::stream(mask).file_name(format!("{}.png", rng.sample::<u64, _>(Standard)))
                }),
                client,
            )
            .await;
    }

    /// Sends the request with the specified files.
    pub async fn with_part(
        self,
        image: Part,
        mask: Option<Part>,
        client: impl AsRef<Client>,
    ) -> Result<Images> {
        let mut body = Form::new().text("prompt", self.prompt).part("image", image);

        if let Some(mask) = mask {
            body = body.part("mask", mask)
        }
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
            .as_ref()
            .post("https://api.openai.com/v1/images/edits")
            .multipart(body)
            .send()
            .await?
            .json::<FallibleResponse<Images>>()
            .await?
            .into_result()?;

        return Ok(resp);
    }
}
