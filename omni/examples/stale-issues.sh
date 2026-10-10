echo "#2269 说 audio_encoder 引用了不存在的 get_feat_extract_output_lengths，基准提交里是："
git grep -n 'feat_extract_output_lengths' "$REF" -- sglang_omni/models/qwen3_omni/components/audio_encoder.py | sed "s/^$REF://"
echo "#1147 说编码器固定等 50 ms，基准提交里的默认值："
git show "$REF:sglang_omni/models/qwen3_omni/stages.py" | sed -n '356,361p'
