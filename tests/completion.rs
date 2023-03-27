use libopenai::{error::Result, prelude::Completion, Client};

#[tokio::test]
async fn basic() -> Result<()> {
    dotenv::dotenv().unwrap();
    let client = Client::new(None)?;

    let basic = Completion::new(
        "text-ada-001",
        "What's the most effitient way to calculate a factorial?",
        &client,
    )
    .await?;

    println!("{:#?}", basic);
    return Ok(());
}

#[tokio::test]
async fn logprob() -> Result<()> {
    dotenv::dotenv().unwrap();
    let client = Client::new(None)?;

    let basic = Completion::builder("text-ada-001")
        .prompt(["What's the most effitient way to calculate a factorial?"])
        .build(&client)
        .await?;

    println!("{:#?}", basic);
    return Ok(());
}
