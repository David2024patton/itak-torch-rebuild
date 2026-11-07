# tokenizer package: byte-level BPE and SentencePiece BPE,
# selected by the model's own GGUF metadata, mirroring the Rust
# reference in crates/tokenizer/ (see tokenizer.mojo for the
# semantics contract and CONFORMANCE parity notes there).
# Modules: byte_level (the GPT-2 byte alphabet + UTF-8 codec),
# pretokenize (the GPT-4o / llama4 scanner), spm (SentencePiece),
# tokenizer (construction, encode, decode), gguf_vocab (the
# temporary GGUF metadata seam; see its header).
