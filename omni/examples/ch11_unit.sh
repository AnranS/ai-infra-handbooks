cd "$OMNI_TREE"
for d in pipeline config relay; do
  line=$("$PYTHON" -m pytest -q -p no:cacheprovider --basetemp="/tmp/omni-ut-$d" \
           -m "not benchmark and not accelerator" "tests/unit_test/$d" 2>/dev/null | tail -1)
  printf '%-10s %s\n' "$d" "$(echo "$line" | sed -E 's/ in [0-9.]+s.*//; s/, [0-9]+ warnings?//')"
done
