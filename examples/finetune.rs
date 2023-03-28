use futures::TryStreamExt;
use libopenai::{file::TemporaryFile, prelude::*};

#[tokio::main]
async fn main() -> Result<()> {
    const MODEL: &str = "text-davinci-003";

    dotenv::dotenv().unwrap();
    tracing_subscriber::fmt::init();
    let client = Client::new(None, None)?;

    let ft = match FineTune::retreive("ft-8KUqxC0IcvgVe707j93Xoa6D", &client).await {
        Ok(x) => x,
        Err(_) => {
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

            let training_file = TemporaryFile::from_file(
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
