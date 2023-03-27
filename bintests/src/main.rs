use libopenai::{embeddings::Embeddings, Client};

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    dotenv::dotenv()?;
    let client = Client::new(None)?;

    embed(&client).await?;
    return Ok(());
}
