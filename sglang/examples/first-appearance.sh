REF=${REF:-29f6d408c0}
first() { git log --reverse --date=short --format='%ad %h %s' "$REF" -- "$1" | head -1 | cut -c1-80; }
printf '%-46s %s\n' python/sglang/srt/mem_cache/ "$(first python/sglang/srt/mem_cache)"
printf '%-46s %s\n' python/sglang/srt/disaggregation/ "$(first python/sglang/srt/disaggregation)"
printf '%-46s %s\n' python/sglang/srt/mem_cache/radix_cache.py "$(first python/sglang/srt/mem_cache/radix_cache.py)"
echo "-- 用 --follow 追踪 radix_cache.py 的改名："
git log --follow --reverse --date=short --format='%ad %h %s' "$REF" -- python/sglang/srt/mem_cache/radix_cache.py | head -1 | cut -c1-80
