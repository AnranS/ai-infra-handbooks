REF=${REF:-29f6d408c0}
for y in 2024 2025 2026; do
  printf '%s  提交 %5d  作者 %4d\n' "$y" "$(git log --date=short --format=%ad "$REF" | grep -c "^$y")" "$(git log --date=short --format='%ad %aN' "$REF" | grep "^$y" | cut -c12- | sort -u | wc -l)"
done
echo "累计作者：$(git log --format=%aN "$REF" | sort -u | wc -l)"
echo "srt/ 的 .py 文件：$(git ls-tree -r --name-only "$REF" -- python/sglang/srt | grep -c '\.py$')；test/ 的 .py 文件：$(git ls-tree -r --name-only "$REF" -- test | grep -c '\.py$')"
echo "2026 年的版本 tag（不含网关）：$(git for-each-ref --sort=creatordate --format='%(creatordate:short) %(refname:short)' refs/tags | grep '^2026' | grep -vc gateway) 个，从 $(git for-each-ref --sort=creatordate --format='%(creatordate:short) %(refname:short)' refs/tags | grep '^2026' | grep -v gateway | head -1) 到 $(git for-each-ref --sort=creatordate --format='%(creatordate:short) %(refname:short)' refs/tags | grep '^2026' | grep -v gateway | tail -1)"
