REF=${REF:-29f6d408c0}
printf '%-46s %6s\n' '今天的文件' '行数'
for f in python/sglang/srt/entrypoints/http_server.py python/sglang/srt/managers/tokenizer_manager.py python/sglang/srt/managers/scheduler.py \
         python/sglang/srt/managers/schedule_batch.py python/sglang/srt/managers/schedule_policy.py python/sglang/srt/managers/tp_worker.py \
         python/sglang/srt/model_executor/model_runner.py python/sglang/srt/mem_cache/radix_cache.py python/sglang/srt/mem_cache/memory_pool.py \
         python/sglang/srt/layers/radix_attention.py python/sglang/srt/managers/detokenizer_manager.py; do
  printf '%-46s %6d\n' "${f#python/sglang/srt/}" "$(git show "$REF:$f" | wc -l)"
done
