use api::completion::Completion;
use futures::TryStreamExt;
pub mod api;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::dotenv()?;
    let api_key = std::env::var("API_KEY")?;

    let mut stream = Completion::new_stream(
        "text-davinci-003",
        "Whats the square root of two?",
        &api_key,
    )
    .await?
    .into_text_stream();

    while let Some(c) = stream.try_next().await? {
        print!("{}", c)
    }

    return Ok(());
}
