use libopenai::{image::Size, prelude::Images, Client};

#[tokio::test(flavor = "multi_thread")]
async fn generate() -> anyhow::Result<()> {
    dotenv::dotenv()?;
    let client = Client::new(None)?;

    let result = Images::generate("depressed programmer")?
        .size(Size::P512)
        .build(&client)
        .await?;

    println!("{result:?}");
    return Ok(());
}
