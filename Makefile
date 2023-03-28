check:
	cargo check
	cargo check --all-features
	cargo check --tests
	cargo check --tests --all-features

doc:
	cargo +nightly rustdoc --all-features --open -- --cfg docsrs