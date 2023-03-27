doc:
	cargo +nightly rustdoc --all-features --open -- --cfg docsrs

test:
	cd bintests && cargo run && cd ..