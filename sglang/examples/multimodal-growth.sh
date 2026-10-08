REF=${REF:-29f6d408c0}
git log --reverse --date=short --format='%ad  %h  %s' "$REF" -- python/sglang/srt/multimodal | head -1 | cut -c1-96
for t in v0.5.0rc0 "$REF"; do
  printf '%-11s multimodal/ %3d 个文件，processors/ %3d 个\n' "$t" "$(git ls-tree -r --name-only "$t" -- python/sglang/srt/multimodal | grep -c '\.py$')" "$(git ls-tree -r --name-only "$t" -- python/sglang/srt/multimodal/processors | grep -c '\.py$')"
done
echo "今天 processors/ 里的模型族：$(git ls-tree -r --name-only "$REF" -- python/sglang/srt/multimodal/processors | grep '\.py$' | sed 's|.*/||; s|\.py||' | grep -v '__init__\|base_processor' | tr '\n' ' ' | cut -c1-230)"
