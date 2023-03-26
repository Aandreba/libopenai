use crate::api::image::{self, Images};
use api::completion::Completion;
pub mod api;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let api_key = dotenv::var("API_KEY")?;

    let similar = Images::edit("Sad man")?
        .size(image::Size::P512)
        .with_file("media/me.jpg", None, &api_key)
        .await?;

    println!("{similar:#?}");
    return Ok(());
}

pub async fn complete_translate_to(target_lang: &str, input: &str) -> anyhow::Result<String> {
    let api_key = dotenv::var("API_KEY")?;
    let trans = Completion::builder("text-davinci-003")
        .set_prompts([format!("translate \"{input}\" to {target_lang}")])
        .temperature(0.0)
        .unwrap()
        .build(&api_key)
        .await?
        .into_first()
        .ok_or_else(|| anyhow::Error::msg("No choices found"))?;

    return Ok(trans);
}
