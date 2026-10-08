REF=${REF:-29f6d408c0}
echo "带 vLLM 来源说明的文件：$(git grep -l 'Adapted from.*vllm\|adapted from vllm\|Copyright contributors to the vLLM project' "$REF" -- python/sglang/srt | wc -l)"
echo "仍然 import vllm 的语句："
git grep -h 'from vllm\|import vllm' "$REF" -- python/sglang/srt | sed 's/^ *//' | sort | uniq -c | sort -rn | head -6
