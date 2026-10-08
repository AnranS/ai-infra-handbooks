REF=${REF:-29f6d408c0}
for t in v0.4.0 v0.4.6 v0.5.0rc0 "$REF"; do
  printf '%-11s layers/moe %3d 个文件，eplb %2d 个，two_batch_overlap.py %4d 行\n' "$t" "$(git ls-tree -r --name-only "$t" -- python/sglang/srt/layers/moe | grep -c '\.py$')" "$(git ls-tree -r --name-only "$t" -- python/sglang/srt/eplb | grep -c '\.py$')" "$(git show "$t:python/sglang/srt/two_batch_overlap.py" 2>/dev/null | wc -l)"
done
echo "今天 batch_overlap/：$(git ls-tree -r --name-only "$REF" -- python/sglang/srt/batch_overlap | sed 's|.*/||' | tr '\n' ' ')"
