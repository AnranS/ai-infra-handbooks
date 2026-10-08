REF=${REF:-29f6d408c0}
echo "v0.4.0：$(git show v0.4.0:python/sglang/srt/layers/attention/__init__.py | grep -c '    def ') 个方法"
echo "今天：$(git show "$REF:python/sglang/srt/layers/attention/base_attn_backend.py" | grep -c '    def ') 个方法，其中和 cuda_graph 有关的 $(git show "$REF:python/sglang/srt/layers/attention/base_attn_backend.py" | grep '    def ' | grep -ci 'graph') 个、和投机解码有关的 $(git show "$REF:python/sglang/srt/layers/attention/base_attn_backend.py" | grep '    def ' | grep -ciE 'draft|verify') 个"
git show "$REF:python/sglang/srt/layers/attention/base_attn_backend.py" | grep '    def ' | sed 's/^ *def //; s/(.*//' | tr '\n' ' ' | cut -c1-400; echo
