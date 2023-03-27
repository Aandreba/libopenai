use libopenai::{finetune::data::TrainingData, Client};

#[tokio::test]
async fn fine() -> anyhow::Result<()> {
    let client = Client::new(None)?;
    let data =
        TrainingData::builder().save_iter([TrainingData::new("sqrt", "square root")], client);

    return Ok(());
}
