use libopenai::{
    error::Result,
    prelude::{Completion, Images, TranscriptionBuilder},
    Client,
};

#[tokio::test]
async fn basic() -> Result<()> {
    dotenv::dotenv().unwrap();
    let client = Client::new(None, None)?;

    let basic = Completion::new(
        "text-ada-001",
        "The best way to calculate a factorial is ",
        &client,
    )
    .await?;

    println!("{:#?}", basic);
    return Ok(());
}

#[tokio::test]
async fn stream() -> Result<()> {
    dotenv::dotenv().unwrap();
    let client = Client::new(None, None)?;

    let basic = Completion::builder("text-ada-001")
        .prompt(["The best way to calculate a factorial is "])
        .max_tokens(256)
        .build(&client)
        .await?;

    println!("{:#?}", &basic.choices);
    return Ok(());
}

#[tokio::test]
async fn audio() -> Result<()> {
    dotenv::dotenv().unwrap();
    let client = Client::new(None, None)?;

    let srt = TranscriptionBuilder::new()
        .response_format(libopenai::audio::AudioResponseFormat::Srt)
        .temperature(0.0)?
        .with_file("./media/audio.mp3", &client)
        .await?;

    println!("{:#?}", srt.text());
    println!("{:#?}", srt.duration());
    println!("{:#?}", srt.segments().map(Iterator::collect::<Vec<_>>));

    let verbose = TranscriptionBuilder::new()
        .response_format(libopenai::audio::AudioResponseFormat::VerboseJson)
        .temperature(0.0)?
        .with_file("./media/audio.mp3", &client)
        .await?;

    println!("{:#?}", verbose.text());
    println!("{:#?}", verbose.duration());
    println!("{:#?}", verbose.segments().map(Iterator::collect::<Vec<_>>));

    return Ok(());
}

#[tokio::test]
async fn image() -> Result<()> {
    dotenv::dotenv().unwrap();
    let client = Client::new(None, None)?;

    Images::create("A shark eating yo mama")?
        .n(2)?
        .build(&client)
        .await?
        .save_at("./media/out")
        .await?;

    return Ok(());
}
