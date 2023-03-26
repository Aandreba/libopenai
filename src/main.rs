use crate::api::{chat::Message, prelude::ChatCompletion};

pub mod api;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let api_key = dotenv::var("API_KEY")?;

    let chat = ChatCompletion::builder("gpt-3.5-turbo", [Message::user("Say hi!")])
        .stop(["Hi"])?
        .build(&api_key)
        .await?;
    println!("{chat:#?}");

    return Ok(());
}
