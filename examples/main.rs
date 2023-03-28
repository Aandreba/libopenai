use libopenai::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv::dotenv().unwrap();

    let client = Client::new(
        None, // Gets api key from `OPENAI_API_KEY` enviroment variable
        None, // No organization specified
    )?;

    // Send basic completion request
    let basic = Completion::new(
        "text-davinci-003",
        "Whats the best way to calculate a factorial?",
        &client,
    )
    .await?;

    // Print the result
    println!("{:#?}", basic);
    return Ok(());
}
