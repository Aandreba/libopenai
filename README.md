![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/Aandreba/libopenai/Rust)

## libopenai - Rust client to interact with OpenAI's API

## How to use

To add `libopenai` to your project, you just need to run the following command on your project's main foler:

`cargo add libopenai`

### Example

```rust
use libopenai::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let client = Client::new(
        // Gets api key from `OPENAI_API_KEY` enviroment variable
        None,
        // No organization specified
        None
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
```

## Features

-   tracing: Enables logging
