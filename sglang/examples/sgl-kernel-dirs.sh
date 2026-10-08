REF=${REF:-29f6d408c0}
for t in v0.4.6 v0.5.0rc0; do
  echo "== $t：$(git ls-tree -r --name-only $t -- sgl-kernel | wc -l) 个文件，.cu $(git ls-tree -r --name-only $t -- sgl-kernel | grep -c '\.cu$') 个"
  git ls-tree -r --name-only $t -- sgl-kernel/csrc | awk -F/ 'NF > 3 {print $3}' | sort | uniq -c | awk '{printf "   %3d %s\n", $1, $2}'
done
echo "== $REF：python/sglang/kernels/ 有 $(git ls-tree -r --name-only "$REF" -- python/sglang/kernels | wc -l) 个文件，.cu $(git ls-tree -r --name-only "$REF" -- python/sglang/kernels | grep -c '\.cu$') 个"
