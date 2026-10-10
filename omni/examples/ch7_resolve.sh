work=$(mktemp -d) && cd "$work"
cat > tts.yaml <<'EOF'
config_cls: Qwen3TTSPipelineConfig
model_path: Qwen/Qwen3-TTS-12Hz-0.6B-Base
stages:
  tts_engine:
    engine:
      mem_fraction_static: 0.6
  vocoder:
    factory:
      max_batch_size: 16
EOF
omni() { COLUMNS=100 "$PYTHON" -m sglang_omni.cli "$@" 2>/dev/null; }
echo "== --show diff"
omni config resolve --config tts.yaml --vocoder.factory.max_batch_size 32 --mem-fraction-static 0.7 --show diff
echo "== config explain"
omni config explain tts_engine.engine.mem_fraction_static --config tts.yaml --mem-fraction-static 0.7
echo "== 写错 stage 名"
COLUMNS=200 "$PYTHON" -m sglang_omni.cli config resolve --config tts.yaml --vocodr.factory.dtype float16 2>&1 \
  | grep -oE "Invalid value: .*Qwen3TTSPipelineConfig|stages of this pipeline: [a-z_, ]+[a-z_]|did you mean: [a-z_]+"
