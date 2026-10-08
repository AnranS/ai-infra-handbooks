REF=${REF:-29f6d408c0}
for spec in "v0.4.0 rust" "v0.4.6 sgl-router" "v0.5.0rc0 sgl-router" "$REF sgl-model-gateway"; do
  set -- $spec
  printf '%-11s %-18s %4d 个文件，其中 .rs %3d 个，共 %6d 行 Rust\n' "$1" "$2" "$(git ls-tree -r --name-only "$1" -- "$2" | wc -l)" "$(git ls-tree -r --name-only "$1" -- "$2" | grep -c '\.rs$')" "$(git ls-tree -r --name-only "$1" -- "$2" | grep '\.rs$' | while read -r f; do git show "$1:$f"; done | wc -l)"
done
