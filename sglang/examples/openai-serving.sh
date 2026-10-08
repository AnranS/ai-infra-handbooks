REF=${REF:-29f6d408c0}
for f in $(git ls-tree -r --name-only "$REF" -- python/sglang/srt/entrypoints/openai | grep '\.py$'); do printf '%5d  %s\n' "$(git show "$REF:$f" | wc -l)" "${f#python/sglang/srt/entrypoints/openai/}"; done
echo "-- serving_base.py 的方法："; git show "$REF:python/sglang/srt/entrypoints/openai/serving_base.py" | grep -E '^    (async )?def ' | sed 's/^ *//; s/(.*//' | tr '\n' ' '; echo
