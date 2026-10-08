REF=${REF:-29f6d408c0}
for r in 22085081bb v0.1.12 v0.4.0 "$REF"; do
  echo "== $r"; git ls-tree --name-only "$r" python/sglang/srt/constrained/ | sed 's|python/sglang/srt/constrained/|   |'
done
