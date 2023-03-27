use crate::{error::Result, file::File, Client};
use futures::{Stream, StreamExt, TryStream, TryStreamExt};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingData {
    pub prompt: String,
    pub completion: String,
}

#[derive(Debug, Clone)]
pub struct TrainingDataBuilder {
    filename: Option<String>,
    prefix: Option<&'static str>,
    suffix: Option<&'static str>,
}

impl TrainingData {
    #[inline]
    pub fn new(prompt: impl Into<String>, completion: impl Into<String>) -> Self {
        return Self {
            prompt: prompt.into(),
            completion: completion.into(),
        };
    }

    #[inline]
    pub fn builder() -> TrainingDataBuilder {
        TrainingDataBuilder::new()
    }
}

impl TrainingDataBuilder {
    #[inline]
    pub fn new() -> Self {
        return Self {
            filename: None,
            prefix: Some(" "),
            suffix: Some("\n"),
        };
    }

    #[inline]
    pub fn filename(mut self, filename: impl Into<String>) -> Self {
        self.filename = Some(filename.into());
        self
    }

    #[inline]
    pub fn prefix(mut self, prefix: Option<&'static str>) -> Self {
        self.prefix = prefix;
        self
    }

    #[inline]
    pub fn suffix(mut self, suffix: Option<&'static str>) -> Self {
        self.suffix = suffix;
        self
    }

    pub async fn save_iter<I>(self, data: I, client: impl AsRef<Client>) -> Result<File>
    where
        I: IntoIterator<Item = TrainingData>,
        I::IntoIter: 'static + Send + Sync,
    {
        return self.save_stream(futures::stream::iter(data), client).await;
    }

    pub async fn try_save_iter<I, E>(self, data: I, client: impl AsRef<Client>) -> Result<File>
    where
        I: IntoIterator<Item = Result<TrainingData, E>>,
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
        I::IntoIter: 'static + Send + Sync,
    {
        return self
            .try_save_stream(futures::stream::iter(data), client)
            .await;
    }

    pub async fn save_stream<S>(self, data: S, client: impl AsRef<Client>) -> Result<File>
    where
        S: 'static + Send + Sync + Stream<Item = TrainingData>,
    {
        let data = data.map(move |mut x| {
            if let Some(prefix) = self.prefix {
                x.completion.insert_str(0, prefix)
            }

            if let Some(suffix) = self.suffix {
                x.completion.push_str(suffix)
            }

            return x;
        });

        return File::upload_stream(data, self.filename, "fine-tune", client).await;
    }

    pub async fn try_save_stream<S>(self, data: S, client: impl AsRef<Client>) -> Result<File>
    where
        S: 'static + Send + Sync + TryStream<Ok = TrainingData>,
        S::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        let data = data.map_ok(move |mut x| {
            if let Some(prefix) = self.prefix {
                x.completion.insert_str(0, prefix)
            }

            if let Some(suffix) = self.suffix {
                x.completion.push_str(suffix)
            }

            return x;
        });

        return File::try_upload_stream(data, self.filename, "fine-tune", client).await;
    }
}

impl Default for TrainingDataBuilder {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}
