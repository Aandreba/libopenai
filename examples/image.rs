use libopenai::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv::dotenv().unwrap();
    tracing_subscriber::fmt::init();

    let client = Client::new(None, None)?;

    Images::create("Nintendo Switch playing The Last of Us")?
        .n(2)?
        .build(&client)
        .await?
        .save_at("./media/out")
        .await?;

    return Ok(());
}
