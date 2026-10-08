REF=${REF:-29f6d408c0}
for t in v0.4.6 v0.5.0rc0 "$REF"; do
  printf '%-11s %3d 个文件：' "$t" "$(git ls-tree -r --name-only "$t" -- python/sglang/srt/disaggregation | grep -c '\.py$')"
  git ls-tree -r --name-only "$t" -- python/sglang/srt/disaggregation | sed 's|python/sglang/srt/disaggregation/||' | awk -F/ '{print (NF > 1 ? $1 "/" : $1)}' | sort -u | tr '\n' ' ' | cut -c1-130; echo
done
