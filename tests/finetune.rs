use libopenai::{file::File, Client};

#[tokio::test]
async fn fine() -> anyhow::Result<()> {
    let client = Client::new(dotenv::var("API_KEY").ok().as_deref())?;
    let file = File::retreive("test", &client).await;
    println!("{file:?}");
    return Ok(());
}
