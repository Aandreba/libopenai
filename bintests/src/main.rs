use libopenai::{embeddings::Embeddings, Client};

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    dotenv::dotenv()?;
    let client = Client::new(None)?;

    embed(&client).await?;
    return Ok(());
}

async fn embed(client: &Client) -> anyhow::Result<()> {
    let data = Embeddings::new(
        "text-embedding-ada-002",
        "What's the square root of two?",
        client,
    )
    .await?;

    println!("{data:#?}");
    return Ok(());
}
