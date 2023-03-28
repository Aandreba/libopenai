[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/Aandreba/libopenai/rust.yml)](https://github.com/Aandreba/libopenai)
[![GitHub](https://img.shields.io/github/license/Aandreba/libopenai)](https://github.com/Aandreba/libopenai/blob/master/LICENSE.md)
[![Crates.io](https://img.shields.io/crates/v/libopenai)](https://crates.io/crates/libopenai)
[![docs.rs](https://img.shields.io/docsrs/libopenai)](https://docs.rs/libopenai/latest/libopenai)

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
    // OPTIONAL: Load variables in a `.env` file into the enviroment
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
```

## Features

-   Text generation
    -   [Regular completion](https://docs.rs/libopenai/latest/libopenai/completion)
    -   [Chat completion](https://docs.rs/libopenai/latest/libopenai/chat)
    -   [Edit](https://docs.rs/libopenai/latest/libopenai/edit)
-   [Embeddings](https://docs.rs/libopenai/latest/libopenai/embeddings)
-   [Moderations](https://docs.rs/libopenai/latest/libopenai/moderations)
-   [Files](https://docs.rs/libopenai/latest/libopenai/file) and [fine-tuning](https://docs.rs/libopenai/latest/libopenai/finetune)
-   [Image generation](https://docs.rs/libopenai/latest/libopenai/image) with automatic conversion to desired formats
-   [Audio-to-text](https://docs.rs/libopenai/latest/libopenai/audio) conversions
-   Support for streaming

## Cargo features

Currently, the only feature available is **tracing**, which enables some minor logging
