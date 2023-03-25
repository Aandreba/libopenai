use crate::api::{
    audio::{transcription::Transcription, translation::Translation},
    moderations::Moderation,
};
use api::{completion::Completion, edit::Edit};
pub mod api;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let api_key = dotenv::var("API_KEY")?;

    let text = Transcription::new()
        .with_file("media/audio.mp3", &api_key)
        .await?;

    let moderation = Moderation::new(&text, None, &api_key).await?;

    println!("{moderation:#?}");
    return Ok(());
}

//Maybe broken?
pub async fn edit_translate_to(target_lang: &str, input: &str) -> anyhow::Result<String> {
    let api_key = dotenv::var("API_KEY")?;
    let mut edit = Edit::create(
        "text-davinci-edit-003",
        format!("translate to {target_lang}"),
        input,
        &api_key,
    )
    .await?;

    let first = edit
        .choices
        .first_mut()
        .ok_or_else(|| anyhow::Error::msg("No choices found"))?;

    return Ok(core::mem::take(&mut first.text));
}

pub async fn complete_translate_to(target_lang: &str, input: &str) -> anyhow::Result<String> {
    let api_key = dotenv::var("API_KEY")?;
    let trans = Completion::builder("text-davinci-003")
        .set_prompt(format!("translate \"{input}\" to {target_lang}"))
        .temperature(0.0)
        .unwrap()
        .build(&api_key)
        .await?
        .into_first()
        .ok_or_else(|| anyhow::Error::msg("No choices found"))?;

    return Ok(trans);
}
