REF=${REF:-29f6d408c0}
echo "io_struct.py 里和权重更新有关的请求类型："
git show "$REF:python/sglang/srt/managers/io_struct.py" | grep -E '^class (Update|Init|Get|Release|Resume|Destroy).*(Weight|Memory|Parameter).*:' | sed 's/^class //; s/[(:].*//' | tr '\n' ' '; echo
echo "weight_sync/：$(git ls-tree -r --name-only "$REF" -- python/sglang/srt/weight_sync | sed 's|.*/||' | tr '\n' ' ')"
git show "$REF:python/sglang/srt/weight_sync/tensor_bucket.py" | grep -E '^class |^    def ' | sed 's/^ *//; s/(.*//' | tr '\n' ' '; echo
