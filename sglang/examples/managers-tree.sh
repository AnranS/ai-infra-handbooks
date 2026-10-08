REF=${REF:-29f6d408c0}
for t in v0.1.5 v0.2.0 v0.3.0 v0.4.0; do
  echo "== $t"; git ls-tree -r --name-only $t -- python/sglang/srt/managers | sed 's|python/sglang/srt/managers/||' | tr '\n' ' '; echo
done
echo "== $REF：$(git ls-tree -r --name-only "$REF" -- python/sglang/srt/managers | grep -c '\.py$') 个文件"
