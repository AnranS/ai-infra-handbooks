total=0
for f in $(git ls-tree -r --name-only v0.2.0 -- python/sglang/srt/managers/controller); do
  n=$(git show "v0.2.0:$f" | wc -l); total=$((total + n)); printf '%5d  %s\n' "$n" "${f#python/sglang/srt/managers/}"
done
echo "合计 $total 行"
