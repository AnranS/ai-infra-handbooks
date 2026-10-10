cd "$OMNI_TREE"
for d in sglang_omni sglang_omni_router sglang_omni_mlx tests benchmarks docs examples playground scripts Voxt OmniTyper; do
  files=$(find "$d" -type f -not -path '*/__pycache__/*' | wc -l)
  lines=$(find "$d" -type f \( -name '*.py' -o -name '*.rs' -o -name '*.swift' -o -name '*.md' -o -name '*.yaml' -o -name '*.toml' \) -not -path '*/__pycache__/*' -print0 | xargs -0 cat | wc -l)
  printf '%-20s %5d 个文件 %7d 行\n' "$d" "$files" "$lines"
done
