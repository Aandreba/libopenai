use api::completion::Completion;
use futures::TryStreamExt;
pub mod api;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let api_key = dotenv::var("API_KEY")?;

    let mut result = Completion::builder("text-davinci-003")
        .stop(["hello"])?
        .prompt(["Say hi!"])
        .build_stream(&api_key)
        .await?
        .into_text_stream();

    while let Some(x) = result.try_next().await? {
        print!("{x}")
    }
    println!();

    return Ok(());
}
