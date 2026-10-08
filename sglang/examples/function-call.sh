REF=${REF:-29f6d408c0}
git log --reverse --date=short --format='%ad  %h  %s' "$REF" -- python/sglang/srt/function_call | head -1 | cut -c1-96
echo "今天 $(git ls-tree -r --name-only "$REF" -- python/sglang/srt/function_call | grep -c '\.py$') 个文件，其中 detector：$(git ls-tree -r --name-only "$REF" -- python/sglang/srt/function_call | grep -c '_detector\.py$') 个"
git ls-tree -r --name-only "$REF" -- python/sglang/srt/function_call | grep '_detector\.py$' | sed 's|.*/||; s|_detector\.py||' | tr '\n' ' ' | cut -c1-200; echo
