cd "$OMNI_TREE/sglang_omni_router/rust"
CARGO=$(command -v cargo || echo "$HOME/.cargo/bin/cargo")
"$CARGO" build --locked -q 2>/dev/null
"$CARGO_TARGET_DIR/debug/sgl-omni-router" --version
