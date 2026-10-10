# 案例二：Qwen3-Omni——七个 stage、流式衔接与一致的路由

<p class="lead">Qwen3-Omni 是 omni 这个项目存在的理由：一个模型，能看图、听音频、看视频，用文字和语音同时回答。它的语音流水线有七个 stage，用到了前面讲过的每一种机制——扇出和投影、带动态条件的扇入、两个嵌 SGLang 的自回归引擎、thinker 到 talker 再到声码器的两级流式、按请求决定的终点。这一章先把七个 stage 摆出来，再看三件最有意思的事：所有动态路由怎么做到"不会对不上"，talker 为什么不等 thinker 说完就开工，以及 talker 和声码器为了首包延迟做了哪些特殊处理。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 语音流水线的七个 stage 是哪些？哪两个是终点？
    2. 一个只要文字输出的请求，会经过 talker 吗？是哪几个函数一起保证了这一点？
    3. thinker 往 talker 推的是什么？talker 什么时候开始生成？
    4. talker 关掉了 SGLang 的哪些功能？为什么？
    5. Qwen3-Omni 的声码器第一段要等几帧？和 Qwen3-TTS 比怎么样？

??? success "自测参考答案（先自己答，再展开对照）"
    1. `preprocessing` → `image_encoder` / `audio_encoder` → `thinker` → `decode`（终点，文本）；`thinker` 流式 → `talker_ar` → 流式 → `code2wav`（终点，音频）。
    2. 不会。路由、扇入、流结束信号、终点四类决定全都基于同一个判断 `should_generate_audio_output`（看请求的 `output_modalities`）：编码器的 `route_fn` 不发给 talker、thinker 的 `stream_done_to_fn` 不通知 talker、Coordinator 的终点只有 `decode`。
    3. thinker 每生成一段，就把这段 token 和对应的隐状态流式推给 talker（以及 decode）。talker 开启了 partial start：攒够配置的可用 chunk 数（talker 的工厂参数里是 5，默认下限 3）就开始构建请求、开始生成，不必等 thinker 说完。
    4. RadixCache（`disable_radix_cache`）、分块 prefill（`chunked_prefill_size=0`），开启反馈时还关掉重叠调度。talker 的输入是 thinker 的隐状态投影出来的 embedding，每个请求都不同，前缀缓存没有意义；反馈式生成要求每一步的结果在下一步之前就绪。
    5. 默认等满一个 `stream_chunk_size`（10 帧），左上下文 25 帧；Qwen3-TTS 默认第一段只等 1 帧。issue #1148 想把首段帧数做成可配置——这正是"把一个模型的优化推广到另一个模型"的机会。

## 七个 stage

第七章用 `prepare_pipeline_runtime` 看过它的部署，这里看拓扑本身。`speech_stages` 把七个 stage 串起来：

```python title="sglang_omni/models/qwen3_omni/config.py @ 921ea2c8 L274-310"
def speech_stages(
    *,
    thinker_gpu: int,
    talker_gpu: int,
    process_by_stage: dict[str, str],
    enable_partial_start: bool,
) -> list[StageConfig]:
    return [
        preprocessing_stage(
            process=process_by_stage["preprocessing"],
            speech_enabled=True,
        ),
        image_encoder_stage(
            gpu=thinker_gpu,
            process=process_by_stage["image_encoder"],
            speech_enabled=True,
        ),
        audio_encoder_stage(
            gpu=thinker_gpu,
            process=process_by_stage["audio_encoder"],
            speech_enabled=True,
        ),
        thinker_stage(
            gpu=thinker_gpu,
            speech_enabled=True,
            process=process_by_stage["thinker"],
        ),
        decode_stage(process=process_by_stage["decode"]),
        talker_stage(
            gpu=talker_gpu,
            process=process_by_stage["talker_ar"],
            enable_partial_start=enable_partial_start,
        ),
        code2wav_stage(gpu=thinker_gpu, process=process_by_stage["code2wav"]),
    ]


```

![图：Qwen3-Omni 的语音流水线](../assets/figures/omni-qwen3-omni.svg){.aig-svg}

| stage | 调度器 | 默认位置 | 做什么 |
| --- | --- | --- | --- |
| `preprocessing` | 简单调度器 | CPU | 解析消息、下载和解码图像 / 音频 / 视频、分词、拼 prompt |
| `image_encoder` / `audio_encoder` | `SimpleScheduler`（可攒批） | GPU0 | 视觉编码器（ViT）、音频编码器 |
| `thinker` | `OmniScheduler` + `ThinkerModelRunner` | GPU0 | 30B-A3B 的 MoE 主干，生成文本 token；prefill 前注入多模态 embedding |
| `decode` | 流式反分词 | CPU | token → 文本，流式发给客户端；**终点** |
| `talker_ar` | `QwenTalkerScheduler`（`OmniScheduler` 子类） | GPU1 | 由 thinker 的隐状态生成 codec 码，多码本反馈 |
| `code2wav` | `Code2WavScheduler`（流式） | GPU0 | codec 码 → 24 kHz 波形，流式发给客户端；**终点** |

注意 thinker 的工厂参数 `enable_async_decode=True`：第五章讲过的 omni 自己的"先发射、后处理"decode 循环默认就开在 thinker 上。

## 一致的路由：一个判断，四处使用

第三章的实验里提醒过：`route_fn` 和终点如果对不上，请求会永远挂着。Qwen3-Omni 有四类动态决定——编码器发给谁、扇入等谁、thinker 的流结束通知谁、Coordinator 等哪些终点——它们是这样保持一致的：

```python title="sglang_omni/models/qwen3_omni/request_builders.py @ 921ea2c8 L110-197"
def should_generate_audio_output(
    payload_or_request: StagePayload | OmniRequest | None,
) -> bool:
    request = (
        payload_or_request.request
        if isinstance(payload_or_request, StagePayload)
        else payload_or_request
    )
    modalities = output_modalities(request)
    return modalities is None or "audio" in modalities


def resolve_thinker_next_stages(
    request_id: str, output: StagePayload
) -> str | list[str]:
    del request_id, output
    return DECODE_STAGE


def resolve_encoder_next_stages(
    request_id: str, output: StagePayload
) -> str | list[str]:
    del request_id
    if should_generate_audio_output(output):
        return [THINKER_STAGE, TALKER_STAGE]
    else:
        pass
    return THINKER_STAGE


def resolve_thinker_stream_done_targets(
    request_id: str, output: StagePayload
) -> list[str]:
    del request_id
    if should_generate_audio_output(output):
        return [TALKER_STAGE, DECODE_STAGE]
    else:
        pass
    return [DECODE_STAGE]


def resolve_terminal_stages(request: OmniRequest) -> list[str]:
    if should_generate_audio_output(request):
        return [DECODE_STAGE, CODE2WAV_STAGE]
    else:
        pass
    return [DECODE_STAGE]


def resolve_preprocessing_next_stages(
    request_id: str, output: StagePayload
) -> list[str]:
    del request_id
    state = Qwen3OmniPipelineState.from_dict(output.data)
    return [
        *encoder_stages_with_model_inputs(state.encoder_inputs),
        MM_AGGREGATE_STAGE,
    ]


def resolve_preprocessing_next_stages_speech(
    request_id: str, output: StagePayload
) -> list[str]:
    del request_id
    state = Qwen3OmniPipelineState.from_dict(output.data)
    targets = [
        *encoder_stages_with_model_inputs(state.encoder_inputs),
        THINKER_STAGE,
    ]
    if should_generate_audio_output(output):
        targets.append(TALKER_STAGE)
    else:
        pass
    return targets


def resolve_mm_aggregate_wait_sources(
    request_id: str,
    from_stage: str,
    payload: StagePayload,
) -> list[str] | None:
    del request_id
    if from_stage != "preprocessing":
        return None
    else:
        pass
    state = Qwen3OmniPipelineState.from_dict(payload.data)
    return ["preprocessing", *active_encoder_stages(state.encoder_inputs)]
```

全部建立在 `should_generate_audio_output` 这一个判断上：请求没写 `output_modalities`，或者写了且包含 `audio`，就走语音路径。在本书的 CPU 环境里，可以直接调用这些函数，看三种请求各自的决定：

```python title="ch9_routing.py"
import warnings

warnings.filterwarnings("ignore")
from sglang_omni.models.qwen3_omni import request_builders as rb
from sglang_omni.proto import OmniRequest, StagePayload

for modalities in (None, ["text"], ["text", "audio"]):
    metadata = {} if modalities is None else {"output_modalities": modalities}
    request = OmniRequest(inputs="描述一下这张图", metadata=metadata)
    payload = StagePayload("r1", request, {})
    print(f"output_modalities={modalities}")
    print("  编码器发给：      ", rb.resolve_encoder_next_stages("r1", payload))
    print("  thinker 流结束通知：", rb.resolve_thinker_stream_done_targets("r1", payload))
    print("  Coordinator 等待：  ", rb.resolve_terminal_stages(request))
```

```text title="输出"
output_modalities=None
  编码器发给：       ['thinker', 'talker_ar']
  thinker 流结束通知： ['talker_ar', 'decode']
  Coordinator 等待：   ['decode', 'code2wav']
output_modalities=['text']
  编码器发给：       thinker
  thinker 流结束通知： ['decode']
  Coordinator 等待：   ['decode']
output_modalities=['text', 'audio']
  编码器发给：       ['thinker', 'talker_ar']
  thinker 流结束通知： ['talker_ar', 'decode']
  Coordinator 等待：   ['decode', 'code2wav']
```

只要文字时，talker 收不到编码器的结果、收不到 thinker 的流结束信号，Coordinator 也不等 `code2wav`——三处决定同时变化，不会出现"有人在等一个永远不会来的输入"。还有第四处：扇入的 `resolve_mm_aggregate_wait_sources` 按预处理的结果决定 thinker 和 talker 要等哪几个编码器（没有图片就不等图像编码器）。

这是写新模型时值得照搬的习惯：**把"走哪条路"收敛成一个函数，所有的 `route_fn`、`wait_for_fn`、`stream_done_to_fn`、`terminal_stages_fn` 都调用它**，而不是各写各的条件。

## thinker → talker：不等说完就开工

talker 需要两样东西：预处理和编码器给的多模态上下文（扇入），以及 thinker 生成的内容（流式）。thinker 生成的 token 和隐状态边生成边推给 talker；talker 用它们构建自己的请求：

```python title="sglang_omni/models/qwen3_omni/request_builders.py @ 921ea2c8 L872-886"
    """Build SGLang AR request for the Talker from thinker hidden states.

    Uses dummy input_ids of matching length for position tracking, while the
    request data keeps a device-backed FIFO of future text rows for decode.

    Stores the original tensor on SGLangARRequestData.prefill_input_embeds
    (projected or not), so the model runner consumes it directly and no
    CPU list conversion happens on the request path.

    Args:
        thinker_hidden_states: Embed layer hidden states [seq_len, hidden_size].
        thinker_layer_hidden: Optional layer-N hidden states for dual-layer mode.
        thinker_token_ids: Optional thinker output token ids aligned with hidden states.
    """
    from sglang.srt.managers.schedule_batch import MultimodalInputs, Req
```

问题是：talker 什么时候开始？等 thinker 说完再开始，语音的首包就要等整段文字生成完；thinker 一出 token 就开始，又可能信息不够。`QwenTalkerScheduler` 的折中是 partial start：

```python title="sglang_omni/models/qwen3_omni/talker_scheduler.py @ 921ea2c8 L107-124"
        self, payload: StagePayload, *, pending_stream_done: bool
    ) -> bool:
        if pending_stream_done:
            return True
        else:
            pass
        if not self.enable_partial_start:
            return False
        else:
            pass
        prefetched = getattr(payload, "prefetched_chunks", None) or []
        usable = self.count_usable_prefetched_chunks(prefetched)
        if self.talker_start_topology:
            return usable >= TALKER_START_MIN_CHUNKS
        else:
            pass
        return usable >= self.partial_start_min_chunks

```

收到流结束信号（thinker 已经说完）当然可以开始；否则开启 partial start 时，攒够若干个可用的 chunk 就开始（`count_usable_prefetched_chunks` 会把最后一个 `<|im_end|>` 排除在外）。语音配置里 talker 的工厂参数是 `partial_start_min_chunks=5`，代码里的下限是 3：

```python title="sglang_omni/models/qwen3_omni/config.py @ 921ea2c8 L23-31"
# Note (wenyao): Config and pre-boot gates preserve the legacy three-chunk floor;
# TALKER_START_MIN_CHUNKS reflects the prompt topology's one-chunk minimum.
MIN_PARTIAL_START_CHUNKS = 3

# Note (wenyao): vLLM-Omni qwen3_omni.py::_get_talker_assistant_parts needs one chunk
# for a 9-row tail (3 template + 4 pad + BOS + text); later chunks feed decode.
TALKER_START_MIN_CHUNKS = 1

ENABLE_TALKER_START_TOPOLOGY = False
```

注释里提到了 vLLM-Omni 的对应实现：talker 的提示词模板尾部需要一个 chunk 就能拼出来，所以"一个 chunk 就开工"的新拓扑（`ENABLE_TALKER_START_TOPOLOGY`）已经写好，但默认还没打开。一个默认关闭的新路径，通常意味着还在验证正确性和性能——这类开关附近往往有可以帮忙的测试和评测工作。

开工之后，thinker 后续的 chunk 仍然源源不断地进来，talker 每一步 decode 前要确认需要的文本行已经到了——这就是 `PendingStreamIngress` 和"后续 chunk 按步把关 decode"的逻辑。

## talker：为反馈式生成关掉的东西

```python title="sglang_omni/models/qwen3_omni/talker_scheduler.py @ 921ea2c8 L28-46"

def configure_talker_server_args(
    server_args: ServerArgs, *, feedback_enabled: bool = True
) -> bool:
    """Apply talker-specific scheduler/runtime defaults.

    Returns whether CUDA graphs were requested so the caller can capture them
    after the model worker is constructed.
    """
    from sglang.srt.arg_groups.model_override_base import resolved_view

    cfg = resolved_view(server_args)
    want_cuda_graph = not bool(cfg.disable_cuda_graph)
    overrides = {"disable_radix_cache": True, "chunked_prefill_size": 0}
    if feedback_enabled:
        overrides["disable_overlap_schedule"] = True
    else:
        pass
    override_server_args(server_args, "qwen3_omni.talker", **overrides)
```

- **关掉 RadixCache**：talker 的 prefill 输入是 thinker 隐状态投影出来的 embedding，每个请求都不同，没有可以复用的前缀；
- **关掉分块 prefill**：talker 的 prefill 要整体构建；
- **开启反馈时关掉重叠调度**：每一步主干出一个码本、小头补齐其余码本、整帧写回作为下一步输入——下一步必须等这一步的结果，"提前一步准备"没有意义。

另外，talker 的 `max_seq_len` 被调到 32768，注释记录了原因：talker 的 prefill 会把 thinker 的整段提示词按投影后的 embedding 重放一遍，一个 30 帧的视频提示词就有约 22K 个位置，8192 会溢出并在 `FusedAddRMSNorm` 里触发非法内存访问。这类"一个默认值背后是一次线上事故"的注释，是读代码时最值得注意的地方。

## code2wav：攒批、CUDA Graph 与首段

声码器的工厂参数：

```python title="sglang_omni/models/qwen3_omni/components/code2wav_scheduler.py @ 921ea2c8 L1379-1394"
def create_code2wav_scheduler(
    model_path: str,
    *,
    device: str | None = None,
    dtype: str | None = None,
    gpu_id: int | None = None,
    stream_chunk_size: int = 10,
    left_context_size: int = 25,
    enable_batching: bool = False,
    initial_codec_chunk_frames: int = 0,
    max_batch_wait_ms: int = 0,
    batch_floor: int = 2,
    batch_ceiling: int = 8,
    enable_output_overlap: bool = True,
    enable_cuda_graph: bool = False,
    total_gpu_memory_fraction: float | None = None,
```

每 10 帧解码一次、带 25 帧左上下文；`initial_codec_chunk_frames=0` 表示第一段也等满 10 帧。和 Qwen3-TTS 的 1、2、4、8 递增序列相比，这里的首包要多等几步——issue #1148 提出把首段帧数做成可配置（复用第八章那个 `resolve_initial_codec_chunk_frames`），并验证跨段拼接是否无缝。talker 的工厂参数里也有对应的 `codec_coalesce_*`：talker 把码攒成 10 帧一组再发，让声码器收到的窗口长度落在预先捕获的 CUDA Graph 形状（10 / 20 / 30 / 35 帧）上。

两边要一起改，正是这类优化"看起来改一个参数、其实要动三个地方"的原因。

## 只要文字、或者只有一张卡

`Variants` 列出了三种变体：

```python title="sglang_omni/models/qwen3_omni/config.py @ 921ea2c8 L456-462"
EntryClass = Qwen3OmniSpeechPipelineConfig

Variants = {
    "text": Qwen3OmniPipelineConfig,
    "speech": Qwen3OmniSpeechPipelineConfig,
    "speech-colocated": Qwen3OmniSpeechColocatedPipelineConfig,
}
```

- `text`：六个 stage，编码器 → `mm_aggregate` → thinker → decode，没有 talker 和声码器（`sgl-omni serve --text-only`）；
- `speech`：默认，两张卡；
- `speech-colocated`：一张卡，声码器住进 talker 的进程（第七章）。

## 练习

**1. 第四处一致性。** 构造一个没有图片、只有音频的预处理结果，调用 `resolve_mm_aggregate_wait_sources`，确认它不等图像编码器。（提示：先读 `active_encoder_stages` 需要的 `encoder_inputs` 长什么样。）

??? success "参考思路"
    读 `request_builders.py` 里的 `active_encoder_stages` 和 `Qwen3OmniPipelineState`：`encoder_inputs` 是按编码器名分开的字典，只有有输入的编码器才出现在里面。构造 `Qwen3OmniPipelineState(encoder_inputs={"audio_encoder": {...}}).to_dict()` 作为 payload 的 `data`，`from_stage="preprocessing"` 时返回 `["preprocessing", "audio_encoder"]`。

**2. 读 talker 的"按步把关"。** 找到 talker 在 decode 前检查"需要的 thinker 文本行是否已到"的代码，说明它在文本行没到时怎么做。

??? success "参考思路"
    在 `talker_scheduler.py` 和 `talker_model_runner.py` 里搜 `chunk_wait` 和 `pending`：文本行没到时，这个请求在本步不参与 decode（被跳过），计数 `chunk_wait_steps` 并定期打日志，等下一个 chunk 到了再继续。这避免了 talker 跑到 thinker 前面。

**3. 评估 #1148 的改动面。** 列出让 Qwen3-Omni 的声码器支持"首段 N 帧"需要改的地方：配置、声码器调度器、talker 的码攒批、CUDA Graph 形状。

??? success "参考思路"
    `config.py` 的 `code2wav_stage` 传 `factory` 参数；`code2wav_scheduler.py` 读请求参数并决定首段大小（`resolve_initial_codec_chunk_frames`），保证第一段用整个首段做左上下文、跨段拼接无缝；talker 的 `codec_coalesce_first_frames` 和首段对齐；CUDA Graph 的捕获形状要包含新的首段长度（否则首段走即时执行）；再加一个验证拼接无缝的单测和一次 TTFA 的对比。

!!! interview "怎么讲清楚"
    讲一个全模态模型的服务，先摆拓扑：预处理 → 两个编码器 → thinker（嵌 SGLang 的 MoE，async decode）→ decode 出文本；thinker 流式 → talker（反馈式多码本，关掉前缀缓存、分块 prefill、重叠调度）→ 流式 → 声码器出音频，两个终点由 Coordinator 合并。再讲两个设计：所有动态路由都基于同一个"要不要语音"的判断，保证路由、扇入、流结束、终点一致；talker 用 partial start，攒够几个 thinker chunk 就开工，后续 chunk 按步把关。最后讲还能优化的地方：声码器首段等满 10 帧，TTS 那边已经有递增 chunk 的现成实现。

## 小结

- [x] 七个 stage：预处理、两个编码器、thinker、decode（终点）、talker、code2wav（终点）；变体有 `text`、`speech`、`speech-colocated`。
- [x] 四类动态决定（路由、扇入、流结束、终点）都基于 `should_generate_audio_output`，保证一致。
- [x] thinker 流式推 token 和隐状态；talker 的 partial start 攒够可用 chunk（配置 5、下限 3）就开工，"一个 chunk 开工"的新拓扑默认关闭。
- [x] talker 关掉 RadixCache、分块 prefill、重叠调度；`max_seq_len=32768` 源于一次视频提示词溢出。
- [x] code2wav 每 10 帧解码、25 帧左上下文，首段等满 10 帧；首段可配置是 #1148 的方向。
