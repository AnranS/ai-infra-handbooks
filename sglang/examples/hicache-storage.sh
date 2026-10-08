REF=${REF:-29f6d408c0}
echo "mem_cache/storage/ 下的后端目录："
git ls-tree -r --name-only "$REF" -- python/sglang/srt/mem_cache/storage | sed 's|python/sglang/srt/mem_cache/storage/||' | awk -F/ 'NF > 1 {print $1}' | sort | uniq -c | awk '{printf "   %2d 个文件  %s\n", $1, $2}'
printf 'hiradix_cache.py 行数：v0.4.6 %d，v0.5.0rc0 %d\n' "$(git show v0.4.6:python/sglang/srt/mem_cache/hiradix_cache.py | wc -l)" "$(git show v0.5.0rc0:python/sglang/srt/mem_cache/hiradix_cache.py | wc -l)"
echo "hiradix_cache.py 的最后一个提交：$(git log -1 --date=short --format='%ad %h %s' "$REF" -- python/sglang/srt/mem_cache/hiradix_cache.py | cut -c1-80)"
echo "今天的统一树实现：$(git ls-tree -r --name-only "$REF" -- python/sglang/srt/mem_cache | grep -E 'unified_radix_cache.py|unified_cache/|rust_tree_core/' | wc -l) 个文件（unified_radix_cache.py、unified_cache/、rust_tree_core/）"
