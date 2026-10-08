REF=${REF:-29f6d408c0}
f=python/sglang/srt/mem_cache/radix_cache.py
echo "提交数：$(git log --follow --format=%h "$REF" -- $f | wc -l)"
for y in 2024 2025 2026; do echo "  $y：$(git log --follow --date=short --format=%ad "$REF" -- $f | grep -c "^$y")"; done
echo "今天的行数：$(git show "$REF:$f" | wc -l)"
echo "mem_cache/ 的 Python 文件数：$(git ls-tree -r --name-only "$REF" -- python/sglang/srt/mem_cache | grep -c '\.py$')"
