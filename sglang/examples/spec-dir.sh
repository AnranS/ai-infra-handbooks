REF=${REF:-29f6d408c0}
for t in v0.4.6 v0.5.0rc0 "$REF"; do
  printf '%-11s %3d 个文件：' "$t" "$(git ls-tree -r --name-only "$t" -- python/sglang/srt/speculative | grep -c '\.py$')"
  git ls-tree -r --name-only "$t" -- python/sglang/srt/speculative | grep '\.py$' | grep -c '_worker' | xargs printf '%s 个带 worker 的文件；'
  git ls-tree -r --name-only "$t" -- python/sglang/srt/speculative | grep '_worker.*\.py$' | sed 's|.*/||; s|\.py||' | tr '\n' ' ' | cut -c1-120; echo
done
