for f in $(git ls-tree --name-only "$REF" .github/workflows/); do
  printf '%-40s %s\n' "$(basename "$f")" "$(git show "$REF:$f" | grep -m1 '^name:' | cut -c7-)"
done
