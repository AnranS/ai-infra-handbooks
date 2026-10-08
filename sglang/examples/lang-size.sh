REF=${REF:-29f6d408c0}
lines() { git ls-tree -r --name-only "$1" -- "$2" | grep '\.py$' | while read -r f; do git show "$1:$f"; done | wc -l; }
echo "lang/ 行数：初版 $(lines 22085081bb python/sglang/lang)，今天 $(lines "$REF" python/sglang/lang)"
echo "srt/ 行数：初版 $(lines 22085081bb python/sglang/srt)，今天 $(lines "$REF" python/sglang/srt)"
for y in 2024 2025 2026; do
  echo "$y 年改动 lang/ 的提交：$(git log --date=short --format=%ad "$REF" -- python/sglang/lang python/sglang/api.py | grep -c "^$y")，改动 srt/managers/ 的：$(git log --date=short --format=%ad "$REF" -- python/sglang/srt/managers | grep -c "^$y")"
done
