use futures::{StreamExt, TryStreamExt};
use libopenai::{
    error::Result,
    file::TemporaryFile,
    finetune::{data::TrainingData, FineTune},
    prelude::{Completion, Images, TranscriptionBuilder},
    Client,
};

#[tokio::test]
async fn basic() -> Result<()> {
    dotenv::dotenv().unwrap();
    tracing_subscriber::fmt::init();
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
    tracing_subscriber::fmt::init();
    let client = Client::new(None, None)?;

    let mut stream = Completion::builder(
        "text-davinci-003",
        "Whats' the best way to calculate a factorial?",
    )
    .n(2)
    .max_tokens(256)
    .build_stream(&client)
    .await?
    .completions()
    .try_for_each(|mut entry| async move {
        while let Some(entry) = entry.next().await {
            println!("{:?}", entry.text);
        }
        return Ok(());
    })
    .await?;

    return Ok(());
}

#[tokio::test]
async fn audio() -> Result<()> {
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

#[tokio::test]
async fn image() -> Result<()> {
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

#[tokio::test]
async fn ft() -> Result<()> {
    const MODEL: &str = "text-ada-003";

    dotenv::dotenv().unwrap();
    tracing_subscriber::fmt::init();
    let client = Client::new(None, None)?;

    let ft = match FineTune::retreive("ft-8KUqxC0IcvgVe707j93Xoa6D", &client).await {
        Ok(x) => x,
        Err(_) => {
            tracing::info!("Fine-Tune not found. Generating new one");

            let questions = Completion::builder(MODEL, "Give me a math question")
                .n(10)
                .max_tokens(100)
                .build(&client)
                .await?
                .choices;

            let answers = Completion::raw_builder(MODEL)
                .echo(true)
                .prompt(questions.iter().map(|x| x.text.as_str()))
                .max_tokens(256)
                .build(&client)
                .await?
                .choices
                .into_iter()
                .filter_map(|x| {
                    if let Some((prompt, completion)) = x.text.split_once("\n\n") {
                        return Some(TrainingData::new(prompt, completion));
                    }
                    return None;
                });

            let training_file = TemporaryFile::new(
                TrainingData::save_iter(answers, &client).await?,
                client.clone(),
            );

            FineTune::new(&training_file.id, &client).await?
        }
    };

    let mut events = ft.event_stream(&client).await.unwrap();

    let handle = tokio::spawn(async move {
        while let Some(event) = events.try_next().await.unwrap() {
            println!("{event:#?}");
        }
        return Result::<()>::Ok(());
    });

    let example = Completion::new(ft.fine_tuned_model()?, "square root of two", client)
        .await
        .unwrap();
    println!("{example:#?}");

    handle.await.unwrap()?;
    return Ok(());
}
