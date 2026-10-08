REF=${REF:-29f6d408c0}
for d in compilation elastic_ep checkpoint_engine dllm hardware_backend observability arg_groups kv_canary weight_cache beam_search rust_extensions; do
  printf '%-18s %s\n' "$d" "$(git log --reverse --date=short --format='%ad  %h  %s' "$REF" -- python/sglang/srt/$d | head -1 | cut -c1-78)"
done
printf '%-18s %s\n' "kernels (aot)" "$(git log --reverse --date=short --format='%ad  %h  %s' "$REF" -- python/sglang/kernels | head -1 | cut -c1-78)"
printf '%-18s %s\n' "scheduler_components" "$(git log --reverse --date=short --format='%ad  %h  %s' "$REF" -- python/sglang/srt/managers/scheduler_components | head -1 | cut -c1-78)"
