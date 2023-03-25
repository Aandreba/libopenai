use api::{
    completion::Completion,
    edit::Edit,
    image::{Image, ResponseFormat, Size},
};
use std::path::PathBuf;
pub mod api;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let api_key = dotenv::var("API_KEY")?;
    // let api_key = dotenv::var("API_KEY")?;
    // let models = models(&api_key).await?;
    // for model in models {
    //     if !model.id.contains("edit") {
    //         continue;
    //     }
    //     println!("{}: {:#?}", model.id, model.permission);
    // }

    let img = Image::edit_builder("remove his mustache")?
        .n(2)
        .unwrap()
        .size(Size::P512)
        .with_file("img/primeagen.jpg", None, &api_key)
        .await?;

    img.save_at(PathBuf::default()).await?;
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
