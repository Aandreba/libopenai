use libopenai::{image::Size, prelude::Images, Client};

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    dotenv::dotenv()?;
    let client = Client::new(None)?;

    let result = Images::generate("depressed programmer")?
        .size(Size::P512)
        .build(&client)
        .await?;

    println!("{result:?}");
    return Ok(());
}
