use libopenai::Client;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    dotenv::dotenv()?;
    let client = Client::new(None)?;

    return Ok(());
}
