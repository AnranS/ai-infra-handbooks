REF=${REF:-29f6d408c0}
for t in v0.4.0 v0.4.6 v0.5.0rc0 "$REF"; do
  printf '%-11s tp_worker_overlap_thread.py：%4d 行   scheduler.py：%5d 行\n' "$t" "$(git show "$t:python/sglang/srt/managers/tp_worker_overlap_thread.py" 2>/dev/null | wc -l)" "$(git show "$t:python/sglang/srt/managers/scheduler.py" | wc -l)"
done
