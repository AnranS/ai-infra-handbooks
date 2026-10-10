sgl-omni serve --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --tts_engine.engine.mem_fraction_static 0.5 \
    --tts_engine.engine.max_running_requests 16 \
    --tts_engine.engine.cuda_graph_max_bs 16 \
    --port 8000
