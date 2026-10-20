REF=${REF:-29f6d408c0}
for y in 2024 2025 2026; do
  printf '%s  提交 %5d  作者 %4d\n' "$y" "$(git log --date=short --format=%ad "$REF" | grep -c "^$y")" "$(git log --date=short --format='%ad %aN' "$REF" | grep "^$y" | cut -c12- | sort -u | wc -l)"
done
echo "累计作者：$(git log --format=%aN "$REF" | sort -u | wc -l)"
echo "srt/ 的 .py 文件：$(git ls-tree -r --name-only "$REF" -- python/sglang/srt | grep -c '\.py$')；test/ 的 .py 文件：$(git ls-tree -r --name-only "$REF" -- test | grep -c '\.py$')"
BASE=$(git log -1 --date=short --format=%ad "$REF")                   # 只数基准日及以前打的 tag，上游以后再发版输出不变
TAGS=$(git for-each-ref --sort=creatordate --format='%(creatordate:short) %(refname:short)' refs/tags | awk -v d="$BASE" '$1 <= d' | grep '^2026' | grep -v gateway)
echo "2026 年的版本 tag（不含网关）：$(echo "$TAGS" | wc -l) 个，从 $(echo "$TAGS" | head -1) 到 $(echo "$TAGS" | tail -1)"
