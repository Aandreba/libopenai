use libopenai::prelude::Completion;

#[tokio::test]
async fn basic() {
    let basic = Completion::new(model, prompt, client);
}
