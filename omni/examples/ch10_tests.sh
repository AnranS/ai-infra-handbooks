cd "$OMNI_TREE/sglang_omni_router/rust"
CARGO=$(command -v cargo || echo "$HOME/.cargo/bin/cargo")
"$CARGO" test --locked -q 2>/dev/null | grep -E '^test result' | awk '{p += $4; f += $6} END {printf "通过 %d，失败 %d\n", p, f}'
echo "集成测试文件："
ls tests/ | sed 's/^/  /'
