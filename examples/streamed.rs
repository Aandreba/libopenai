use futures::TryStreamExt;
use libopenai::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv::dotenv().unwrap();
    tracing_subscriber::fmt::init();

    let client = Client::new(None, None)?;

    let mut stream = Completion::builder(
        "text-davinci-003",
        "What's the best way to calculate a factorial?",
    )
    .max_tokens(256)
    .build_stream(&client)
    .await?;

    while let Some(completion) = stream.try_next().await? {
        println!("{completion:#?}");
    }

    return Ok(());
}
