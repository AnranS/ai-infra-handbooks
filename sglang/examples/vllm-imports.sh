REF=${REF:-29f6d408c0}
printf '%-11s %6s %6s %8s\n' 版本 文件数 语句数 srt文件数
for t in v0.1.5 v0.2.0 v0.3.0 v0.4.0 v0.4.6 v0.5.0rc0 "$REF"; do
  files=$(git grep -l 'from vllm\|import vllm' "$t" -- python/sglang/srt | wc -l)
  lines=$(git grep -h 'from vllm\|import vllm' "$t" -- python/sglang/srt | wc -l)
  total=$(git ls-tree -r --name-only "$t" -- python/sglang/srt | grep -c '\.py$')
  printf '%-11s %6d %6d %8d\n' "$t" "$files" "$lines" "$total"
done
