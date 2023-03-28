use libopenai::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv::dotenv().unwrap();
    tracing_subscriber::fmt::init();

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
