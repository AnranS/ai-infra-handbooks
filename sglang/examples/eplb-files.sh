REF=${REF:-29f6d408c0}
for f in $(git ls-tree -r --name-only v0.5.0rc0 -- python/sglang/srt/eplb | grep '\.py$'); do printf '%5d  %s\n' "$(git show "v0.5.0rc0:$f" | wc -l)" "${f#python/sglang/srt/}"; done
echo "今天：$(git ls-tree -r --name-only "$REF" -- python/sglang/srt/eplb | grep -c '\.py$') 个文件"
