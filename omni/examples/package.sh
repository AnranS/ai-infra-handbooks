cd "$OMNI_TREE/sglang_omni"
for d in */; do
  d=${d%/}
  [ "$d" = __pycache__ ] && continue
  n=$(find "$d" -name '*.py' -not -path '*/__pycache__/*' -print0 | xargs -0 cat | wc -l)
  echo "$n $d"
done | sort -rn | awk '{printf "%-14s %7d\n", $2, $1}'
