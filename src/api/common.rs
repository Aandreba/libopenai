use futures::{ready, TryStream};
use serde::Deserialize;
use std::{collections::VecDeque, task::Poll};

#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct Choice {
    pub text: String,
    pub index: u32,
    #[serde(default)]
    pub lobprogs: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

pin_project_lite::pin_project! {
    pub struct StreamTokioAsyncRead<S> {
        #[pin]
        inner: S,
        buffer: VecDeque<u8>,
    }
}

impl<S: TryStream<Error = std::io::Error>> StreamTokioAsyncRead<S>
where
    S::Ok: AsRef<[u8]>,
{
    #[inline]
    pub const fn new(inner: S) -> Self {
        return Self {
            inner,
            buffer: VecDeque::new(),
        };
    }
}

impl<S: TryStream<Error = std::io::Error>> tokio::io::AsyncRead for StreamTokioAsyncRead<S>
where
    S::Ok: AsRef<[u8]>,
{
    fn poll_read(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        let mut this = self.project();
        let mut polled = false;

        loop {
            if !this.buffer.is_empty() {
                let buf_len = buf.remaining();
                let this_len = this.buffer.len();

                let chunk = this.buffer.split_off(usize::min(buf_len, this_len));
                let (left, right) = chunk.as_slices();

                buf.put_slice(&left);
                buf.put_slice(&right);
                polled = true
            }

            if buf.remaining() > 0 {
                match this.inner.as_mut().try_poll_next(cx) {
                    Poll::Ready(Some(Ok(bytes))) => {
                        this.buffer.extend(bytes.as_ref());
                        polled = true;
                        continue;
                    }
                    Poll::Ready(Some(Err(e))) => return Poll::Ready(Err(e)),
                    Poll::Ready(None) => polled = true,
                    Poll::Pending => {}
                }
            }

            return match polled {
                true => Poll::Ready(Ok(())),
                false => Poll::Pending,
            };
        }
    }
}

impl<S: TryStream<Error = std::io::Error>> tokio::io::AsyncBufRead for StreamTokioAsyncRead<S>
where
    S::Ok: AsRef<[u8]>,
{
    fn poll_fill_buf(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<std::io::Result<&[u8]>> {
        let this = self.project();

        if !this.buffer.is_empty() {
            return Poll::Ready(Ok(this.buffer.make_contiguous()));
        }

        match ready!(this.inner.try_poll_next(cx)) {
            Some(Ok(bytes)) => {
                this.buffer.extend(bytes.as_ref());
                return Poll::Ready(Ok(this.buffer.make_contiguous()));
            }
            Some(Err(e)) => return Poll::Ready(Err(e)),
            None => return Poll::Ready(Ok(&[])),
        }
    }

    #[inline]
    fn consume(self: std::pin::Pin<&mut Self>, amt: usize) {
        let this = self.project();
        let _ = this.buffer.drain(0..amt);
    }
}
