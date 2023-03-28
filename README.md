![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/Aandreba/libopenai/rust.yml)
![GitHub](https://img.shields.io/github/license/Aandreba/libopenai)
![Crates.io](https://img.shields.io/crates/v/libopenai)
![docs.rs](https://img.shields.io/docsrs/libopenai)

## libopenai - Rust client to interact with OpenAI's API

Rust client for OpenAI's [API](https://platform.openai.com/docs/api-reference), written with [tokio](https://github.com/tokio-rs/tokio) and [reqwest](https://github.com/seanmonstar/reqwest)

## How to use

To add `libopenai` to your project, you just need to run the following command on your project's main foler:

```bash
cargo add libopenai
```

## Example

```rust
use libopenai::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    # dotenv::dotenv().unwrap();

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

```

## Features

Currently, the only feature available is **tracing**, which enables some minor logging
