# 案例一：Qwen3-TTS——三个 stage 与首包延迟

<p class="lead">Qwen3-TTS 是 omni 里最短的一条"真"流水线：预处理 → TTS 引擎 → 声码器，三个 stage 默认住在同一个进程里。它很适合当第一个精读的模型——短，却把前面几章的机制全用上了：线程池调度器、嵌着 SGLang 的自回归引擎、反馈式的多码本生成、按 chunk 流式解码的声码器。这一章沿着一条请求读下去，重点放在语音服务最在意的指标上：<b>首包音频延迟（TTFA）</b>，看 omni 用哪些手段把它压下来。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. Qwen3-TTS 的三个 stage 各用什么调度器？为什么声码器在配置列表里排在 TTS 引擎前面？
    2. talker 每一步生成什么？"多码本"是在哪里生成的？
    3. 默认配置下，声码器收到几帧码就开始出第一段音频？后面的 chunk 怎么变化？
    4. 声音克隆时，参考音频的码为什么要拼在第一个流式 chunk 前面？
    5. 0.6B 模型有一类请求会一直生成到 `max_new_tokens`，这是什么问题？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 预处理用 `ThreadedSimpleScheduler`（8 个工作线程，让参考音频的编码能攒批）；TTS 引擎用 `OmniScheduler`（嵌 SGLang）；声码器用 `Qwen3TTSStreamingVocoderScheduler`（继承 `StreamingVocoderBase`）。stage 按列表顺序构建，声码器先建好，它的权重和 CUDA Graph 已经占住显存，TTS 引擎再按剩下的显存去分 KV 池，不会互相抢。
    2. talker（28 层的自回归主干）每一步生成第一个码本的 token；模型内部的小网络 code predictor（5 层）再在同一步里把其余 15 个码本补齐，凑成一帧 16 个码。每帧对应 80 ms 音频（12.5 Hz）。
    3. 默认的 chunk 递增序列是 1、2、4 帧，之后稳定在每 8 帧解码一次：生成出第 1 帧就能出第一段音频（80 ms），之后逐步放大，兼顾首包延迟和解码效率。
    4. 声码器是卷积网络，解码一段码需要前面若干帧作为"左上下文"才能和前一段无缝衔接。第一个 chunk 前面没有已生成的帧，就用参考音频的码当上下文，克隆出的音色从第一个字就是对的。
    5. 码的 EOS（结束符）"饿死"：模型把话说完后进入一种近乎静音的稳定状态，EOS 的概率掉到 1e-14 量级、排名在 top-k 之外，再也采样不到。issue #1179 统计过约 0.34% 的请求会这样，根因在模型本身，服务侧能做的是正确传播 `finish_reason` 并设合理的长度上限。

## 拓扑

```python title="sglang_omni/models/qwen3_tts/config.py @ 921ea2c8 L66-97"
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            # Note (Jiaxin Deng): no gpu declaration here. Sharing the engine's
            # process the stage holds no GPU budget of its own, and declaring one
            # makes every layout that shares the card demand a fraction for it. A
            # split frontend passes --preprocessing.gpu with its own fraction.
            next="tts_engine",
        ),
        # note(ratish): stages are built in list order. The vocoder comes before
        # the engine so its weights and graphs are resident when the KV pool is sized.
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_vocoder_executor",
            factory=FactoryArgs(dtype="bfloat16"),
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
        EngineStageConfig(
            name="tts_engine",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory=FactoryArgs(dtype="bfloat16"),
            gpu=0,
            next="vocoder",
            stream_to=["vocoder"],
        ),
    ]
```

![图：Qwen3-TTS 的三个 stage](../assets/figures/omni-qwen3-tts.svg){.aig-svg}

三个 stage 都写 `process="pipeline"`：预处理是纯 CPU 工作，声码器和 TTS 引擎共享 GPU0，彼此之间直传 Python 对象（第六章）。两条注释值得记住：

- 预处理没有声明 `gpu`：它和引擎住在一起，不单独占显存预算；如果把它拆到自己的进程，它要自己加载一份提示词前端（`load_frontend=True`）；
- 声码器排在引擎前面：stage 按列表顺序构建，先让声码器的权重和 CUDA Graph 落地，引擎再按剩下的显存分 KV 池。

模型本身（来自模型的 `config.json`）：talker 主干 28 层、隐藏维 1024；每帧 16 个码本（`num_code_groups=16`），码本大小 2048；code predictor 5 层；码的 EOS 是 2150。

## 第一站：预处理

```python title="sglang_omni/models/qwen3_tts/stages.py @ 921ea2c8 L186-218"
def create_preprocessing_executor(
    model_path: str,
    *,
    max_concurrency: int = 8,
    stream_codec_output: bool = True,
    load_frontend: bool = False,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    attn_implementation: str | None = None,
) -> ThreadedSimpleScheduler[StagePayload, StagePayload]:
    if load_frontend:
        load_standalone_preprocessing_context(
            model_path,
            device=device,
            gpu_id=gpu_id,
            dtype=dtype,
            attn_implementation=attn_implementation,
        )
    else:
        pass
    # note (luojiaxuan): preprocessing must admit several requests at once. A
    # serial executor keeps at most one reference-code request in flight, so
    # the speech-tokenizer batcher would only ever see batches of one; the
    # default matches the batcher's max_batch_size.
    return ThreadedSimpleScheduler(
        functools.partial(
            preprocess_qwen3_tts_payload,
            default_stream_codec_output=stream_codec_output,
        ),
        max_concurrency=max_concurrency,
        abort_callback=cleanup_prepared_qwen3_tts_request,
    )
```

预处理要做的事：把文本、语言、音色参数整理成 talker 的输入 embedding；声音克隆时还要把参考音频编码成码（`ref_code`）和说话人向量。参考音频的编码在 GPU 上，单个请求跑很浪费，所以有一个"speech tokenizer 攒批器"。注释说明了为什么要 8 个线程：串行执行时同一时刻最多只有一个请求在等编码，攒批器永远只能看到大小为 1 的批。这是"调度器选型影响下游批大小"的一个好例子。

## 第二站：TTS 引擎

`create_sglang_tts_engine_executor`（第五章看过）用 `Qwen3TtsEngineBuilder` 搭出一个 `OmniScheduler`。它的模型 runner 是反馈式的：每一步 talker 主干出第一个码本，code predictor 在同一次 `forward()` 里补齐其余码本，上一步的整帧码要写回 buffer 作为下一步的输入。

每一步算完，`OmniScheduler` 调模型提供的 `stream_output_builder` 把这一步的码变成流式消息：

```python title="sglang_omni/models/qwen3_tts/request_builders.py @ 921ea2c8 L1939-1987"
    def stream_output_builder(
        request_id: str,
        data: Qwen3TTSSGLangRequestData,
        req_output: RequestOutput,
    ) -> list[OutgoingMessage]:
        del req_output
        params = data.stage_payload.request.params
        if (
            not data.stream_codec_output
            or not isinstance(params, dict)
            or not params.get("stream")
        ):
            return []
        else:
            pass

        codes = data.latest_stream_code_chunk
        if codes is None:
            return []
        else:
            pass
        data.latest_stream_code_chunk = None
        if codes.ndim == 1:
            codes = codes.unsqueeze(0)
        elif codes.ndim != 2:
            raise ValueError(
                f"Qwen3-TTS stream codes must be [Q] or [T, Q], got {tuple(codes.shape)}"
            )
        else:
            pass

        metadata: dict[str, object] = {
            "modality": "audio_codes",
            "stream": True,
            "num_quantizers": int(codes.shape[-1]),
        }
        if not data.stream_ref_sent:
            ref_code = data.ref_code
            ref_code_len = 0
            if ref_code is not None and ref_code.numel() > 0:
                ref_code = ref_code.to(device=codes.device, dtype=torch.long)
                if ref_code.ndim != 2 or ref_code.shape[-1] != codes.shape[-1]:
                    raise ValueError(
                        "Qwen3-TTS reference codes must have shape [T, Q] matching "
                        f"stream codes, got {tuple(ref_code.shape)} and "
                        f"{tuple(codes.shape)}"
                    )
                else:
                    pass
```

几个细节：

- 只有请求要了流式（`params["stream"]`）且没关掉 `stream_codec_output` 时才发；非流式请求等全部生成完，声码器一次解码整段；
- 码的形状是 `[T, Q]`（T 帧、Q 个码本），元数据标着 `modality: audio_codes`；
- 第一次发送时把参考音频的码 `ref_code` 拼在前面——这就是自测第 4 题说的"左上下文"。

## 第三站：流式声码器

声码器决定什么时候解码、一次解码多少帧。默认值：

```python title="sglang_omni/models/qwen3_tts/streaming_vocoder.py @ 921ea2c8 L62-67"
DEFAULT_QWEN3_TTS_STREAM_STRIDE = 16
DEFAULT_QWEN3_TTS_STREAM_FOLLOWUP_STRIDE = 8
DEFAULT_QWEN3_TTS_STREAM_INITIAL_FOLLOWUP_STRIDE = 8
DEFAULT_QWEN3_TTS_INITIAL_CHUNK_FRAMES = 8
DEFAULT_QWEN3_TTS_STREAM_CHUNK_RAMP = (1, 2, 4)
DEFAULT_QWEN3_TTS_LEFT_CONTEXT_FRAMES = 16
```

"什么时候解码"的判断非常短：

```python title="sglang_omni/models/qwen3_tts/streaming_vocoder.py @ 921ea2c8 L1351-1365"
    def should_decode(self, state: Qwen3TTSStreamState, *, is_final: bool) -> bool:
        if is_final:
            return True
        else:
            pass
        generated_frames = state.total_frames - state.ref_frames
        next_frames = self.next_decode_threshold(state)
        return generated_frames >= next_frames

    def next_decode_threshold(self, state: Qwen3TTSStreamState) -> int:
        if state.next_decode_generated_frames:
            return state.next_decode_generated_frames
        else:
            pass
        return state.initial_chunk_frames or self.stream_stride
```

生成的帧数（总帧数减去参考帧）达到下一个阈值就解码一次。阈值序列由 chunk 递增表 `(1, 2, 4)` 和稳定步长决定。在本书的 CPU 环境里，可以直接调用声码器模块里的纯函数，把默认配置下的 chunk 计划算出来：

```python title="ch8_chunks.py"
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
```

```text title="输出"
每次解码的新帧数： [1, 2, 4, 8, 8, 8, 8]
  第 1 段：talker 累计生成  1 帧时解码，这段音频   80 ms，累计   80 ms
  第 2 段：talker 累计生成  3 帧时解码，这段音频  160 ms，累计  240 ms
  第 3 段：talker 累计生成  7 帧时解码，这段音频  320 ms，累计  560 ms
  第 4 段：talker 累计生成 15 帧时解码，这段音频  640 ms，累计 1200 ms
  第 5 段：talker 累计生成 23 帧时解码，这段音频  640 ms，累计 1840 ms
  第 6 段：talker 累计生成 31 帧时解码，这段音频  640 ms，累计 2480 ms
解码窗口（新帧 + 最多 16 帧左上下文）会出现的帧数：
  (1, 3, 7, 15, 17, 18, 19, 20, 21, 22, 23, 24)
```

读这个结果：

- **首包只等 1 帧**。如果第一段要攒满 8 帧，用户要多等 7 个 talker 步才能听到声音；递增序列让第一段立刻出来，再逐步放大到 8 帧一段，后面每段 640 ms 的音频只需解码一次；
- 只要播放速度不超过生成速度——每 80 ms 音频需要的 talker 一步加上摊到每帧的解码时间不超过 80 ms——递增的 chunk 就不会让播放断流；
- 解码窗口的帧数是有限的几种：前几段是"累计帧数"（1、3、7、15），左上下文填满之后是 16 + 新帧数（17～24）。声码器为这些长度预先捕获 CUDA Graph（`decode_graph_frame_counts` 的文档字符串解释了为什么要覆盖整个范围：步长会随到达时间抖动），绝大多数解码都走图回放而不是即时执行。

一条请求的首包延迟大致是：

> TTFA ≈ 预处理 + talker 的 prefill + 第一段需要的 talker 步数 × 每步时间 + 第一段的声码器解码 + 传输

omni 在每一项上都有手段：预处理多线程攒批、prefill 合并（第五章的 `prefill_coalesce_requests`）、递增 chunk（第一段只要 1 步）、声码器的 CUDA Graph 和高优先级 CUDA 流（`vocoder_decode_stream_priority`）、同进程直传。每个请求还可以用 `initial_codec_chunk_frames` 参数覆盖第一段的帧数：

```python title="sglang_omni/scheduling/streaming_vocoder.py @ 921ea2c8 L58-88"
def resolve_initial_codec_chunk_frames(
    params: Mapping[str, object] | None,
    *,
    steady_chunk_frames: int,
    default_frames: int = 0,
) -> int:
    """Resolve the model default or request override, clamped to steady size."""
    if steady_chunk_frames <= 0:
        raise ValueError(
            f"steady_chunk_frames must be positive, got {steady_chunk_frames}"
        )
    else:
        pass
    value = params.get(INITIAL_CODEC_CHUNK_FRAMES_PARAM) if params is not None else None
    if value is None:
        value = default_frames
    else:
        pass

    try:
        frames = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{INITIAL_CODEC_CHUNK_FRAMES_PARAM} must be an integer"
        ) from exc
    if frames < 0:
        raise ValueError(f"{INITIAL_CODEC_CHUNK_FRAMES_PARAM} must be >= 0")
    else:
        pass

    return min(frames, int(steady_chunk_frames))
```

这个公共函数在 `scheduling/streaming_vocoder.py` 里，Higgs TTS、MOSS-TTS Local 也在用——issue #1148 想把 Qwen3-Omni 的声码器也接上它，这就是"一个模型上做过的优化推广到另一个模型"这类贡献的典型样子。

## 一个真实问题：说完了却停不下来

issue #1179 记录了 0.6B 模型的一个问题：约 0.34% 的请求（全量 SeedTTS 评测里 60,424 个请求中有 207 个）在把句子完整说完之后进入近乎静音的稳定状态，一直生成到 `max_new_tokens`（2048 个 token，163.84 秒的音频）。分析结论是模型内在的：在那个状态里 EOS（2150）的概率塌到 1e-14、排名在 top-k=50 之外，重复惩罚也碰不到它。服务侧能做的是：

- 让这类请求的 `finish_reason` 正确地传到 API（`length` 而不是 `stop`），客户端才知道发生了什么——"`finish_reason` 在到达 `/v1/audio/speech` 之前丢了"是单独记录的 issue #1178（已关闭）；
- 给生成长度设和文本长度相称的上限，而不是固定的 2048。

这个 issue 的分析过程（复现、统计比例、看 EOS 的概率和排名、区分"模型问题"和"服务问题"）本身就是一份很好的排查范本，第十二章会再提到它。

## 在 16 GB 的卡上

Qwen3-TTS 的两个尺寸都很小：0.6B 的权重约 2.5 GB，1.7B 约 4.5 GB（HF 上的文件大小）。在一张 16 GB 的消费级卡上，主要要调的是 TTS 引擎的显存参数（第七章的路径语言）：

```bash title="tts-16g.sh" run="no"
sgl-omni serve --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --tts_engine.engine.mem_fraction_static 0.5 \
    --tts_engine.engine.max_running_requests 16 \
    --tts_engine.engine.cuda_graph_max_bs 16 \
    --port 8000
```

这几个值是起点而不是结论：声码器的 CUDA Graph、增量解码的状态槽（`codec_state_slots`）也要显存，实际能开多大要在卡上量。第七章说过，`sgl-omni config resolve` 能先确认这些覆盖都写对了。

## 练习

**1. 算一个更保守的 chunk 计划。** 把 `ch8_chunks.py` 改成"首包攒满 8 帧"（`initial_chunk_frames=8`，不用递增表），比较两种计划下第一段音频之前需要的 talker 步数，以及前 2 秒音频需要解码几次。

??? success "参考思路"
    递增表：第一段 1 步，前 2 秒（25 帧）需要解码 1、2、4、8、8 共 5 次左右；首包 8 帧：第一段 8 步，前 2 秒需要 8、8、8、（1）约 4 次。首包延迟差 7 个 talker 步，总解码次数相差不大——这就是递增表划算的原因。

**2. 找出声码器的"高优先级流"。** 读 `vocoder_decode_stream_priority`，说明它在 CUDA 上返回什么，为什么声码器的解码要放在高优先级流上。

??? success "参考思路"
    `grep -n 'def vocoder_decode_stream_priority' -A12 sglang_omni/scheduling/streaming_vocoder.py`：在支持流优先级的设备上取最高优先级（CUDA 里数值越小优先级越高）。声码器和 talker 共享一张卡，解码 kernel 排在 talker 的大 kernel 后面就会拖慢首包；高优先级流让它插队。

**3. 读参考码的拼接。** 在 `stream_output_builder` 里，`ref_code` 只在第一次发送时拼接（`stream_ref_sent`）。如果每次都拼，声码器会出什么问题？

??? success "参考思路"
    声码器按"总帧数 − 参考帧数"计算生成了多少帧、按左上下文切窗口。每次都拼参考码会让帧数虚增、窗口错位，解出来的音频里会反复出现参考音频的片段。`Qwen3TTSStreamState.ref_frames` 只在第一个 chunk 记一次。

!!! interview "怎么讲清楚"
    讲一个 TTS 服务的首包延迟，先拆公式：预处理 + prefill + 第一段需要的生成步数 × 每步时间 + 第一段解码 + 传输。再逐项讲 omni 的手段：预处理用线程池让参考音频编码能攒批；引擎里 prefill 合并摊薄固定开销；声码器用 1、2、4、8 帧的递增 chunk，首包只等 1 帧，稳定后 8 帧一段；解码窗口的长度有限，预先捕获 CUDA Graph；声码器在高优先级流上插队；三个 stage 同进程直传。最后讲一个模型侧的坑：EOS 饿死让少量请求跑满长度上限，服务侧要正确传播 `finish_reason`。

## 小结

- [x] 拓扑：预处理（`ThreadedSimpleScheduler`）→ TTS 引擎（`OmniScheduler`）→ 声码器（流式），同一个进程；声码器先构建以免和 KV 池抢显存。
- [x] talker 每步出第一个码本，code predictor 在同一次前向里补齐 16 个码本；每帧 80 ms。
- [x] 每步的码经 `stream_output_builder` 流式发出，第一次发送前面拼参考码作为左上下文。
- [x] 声码器默认 chunk 计划 1、2、4、8、8……帧，首包只等 1 帧；解码窗口长度有限，预捕获 CUDA Graph。
- [x] 0.6B 模型约 0.34% 的请求 EOS 饿死（#1179），根因在模型，服务侧负责传播 `finish_reason`。
