REF=${REF:-29f6d408c0}
for spec in "v0.4.0 rust" "v0.4.6 sgl-router" "v0.5.0rc0 sgl-router" "$REF sgl-model-gateway" "$REF rust"; do
  set -- $spec
  printf '%-11s %-18s %4d 个文件，.rs %4d 个，%6d 行 Rust\n' "$1" "$2" "$(git ls-tree -r --name-only "$1" -- "$2" | wc -l)" "$(git ls-tree -r --name-only "$1" -- "$2" | grep -c '\.rs$')" "$(git ls-tree -r --name-only "$1" -- "$2" | grep '\.rs$' | while read -r f; do git show "$1:$f"; done | wc -l)"
done
echo "gateway 的 tag：$(git tag | grep -c '^gateway-v')，最早 $(git for-each-ref --sort=creatordate --format='%(creatordate:short) %(refname:short)' refs/tags | grep gateway | head -1)，最新 $(git for-each-ref --sort=creatordate --format='%(creatordate:short) %(refname:short)' refs/tags | grep gateway | tail -1)"
