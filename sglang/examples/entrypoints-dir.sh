REF=${REF:-29f6d408c0}
git log --reverse --date=short --format='%ad  %h  %s' "$REF" -- python/sglang/srt/entrypoints | head -1 | cut -c1-96
for t in v0.4.6 v0.5.0rc0 "$REF"; do
  printf '%-11s %2d 个文件：' "$t" "$(git ls-tree -r --name-only "$t" -- python/sglang/srt/entrypoints | grep -c '\.py$')"
  git ls-tree -r --name-only "$t" -- python/sglang/srt/entrypoints | sed 's|python/sglang/srt/entrypoints/||' | awk -F/ '{print (NF > 1 ? $1 "/" : $1)}' | sort -u | tr '\n' ' ' | cut -c1-150; echo
done
