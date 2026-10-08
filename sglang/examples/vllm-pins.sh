REF=${REF:-29f6d408c0}
for t in v0.1.5 v0.2.0 v0.3.0 v0.4.0 v0.4.6 v0.5.0rc0 "$REF"; do
  printf '%-11s %s\n' "$t" "$(git show "$t:python/pyproject.toml" | grep -o '"vllm[^"]*"' | tr '\n' ' ')"
done
