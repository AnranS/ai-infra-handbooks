REF=${REF:-29f6d408c0}
echo "sgl-model-gateway/src/ 一级：$(git ls-tree --name-only "$REF" sgl-model-gateway/src/ | sed 's|.*/||' | tr '\n' ' ')"
echo "policies/：$(git ls-tree -r --name-only "$REF" -- sgl-model-gateway/src/policies | sed 's|.*/||; s|\.rs||' | tr '\n' ' ')"
echo "routers/ 一级：$(git ls-tree --name-only "$REF" sgl-model-gateway/src/routers/ | sed 's|.*/||' | tr '\n' ' ')"
echo "rust/ 下的 crate：$(git ls-tree --name-only "$REF" rust/ | sed 's|rust/||' | grep -v '\.' | tr '\n' ' ')"
