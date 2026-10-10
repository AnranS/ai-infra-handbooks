import warnings

warnings.filterwarnings("ignore")
from sglang_omni.models.qwen3_tts.streaming_vocoder import (
    DEFAULT_QWEN3_TTS_LEFT_CONTEXT_FRAMES as LEFT,
    DEFAULT_QWEN3_TTS_STREAM_CHUNK_RAMP as RAMP,
    DEFAULT_QWEN3_TTS_STREAM_FOLLOWUP_STRIDE as STEADY,
    decode_graph_frame_counts,
)

FRAME_MS = 80                                   # 12.5 Hz：每帧 80 ms 音频
ramp = (RAMP[0], *(min(s, STEADY) for s in RAMP[1:]))
schedule = list(ramp) + [STEADY] * 4
print("每次解码的新帧数：", schedule)
done = 0
for i, frames in enumerate(schedule[:6]):
    done += frames
    print(f"  第 {i + 1} 段：talker 累计生成 {done:2d} 帧时解码，这段音频 {frames * FRAME_MS:4d} ms，"
          f"累计 {done * FRAME_MS:4d} ms")
print("解码窗口（新帧 + 最多 16 帧左上下文）会出现的帧数：")
print(" ", decode_graph_frame_counts(left_context=LEFT, initial_chunk_frames=ramp[0],
                                     followup_stride_ramp=ramp[1:], steady_stride=STEADY))
