# 配置与部署：声明式拓扑、按消费者分组与进程规划

<p class="lead">omni 的配置有一条硬规矩：<b>拓扑只在模型的 <code>config.py</code> 里定义</b>——有哪些 stage、怎么连、谁是入口谁是终点，配置文件和命令行只能改这些 stage 的参数，不能增删 stage。在这条规矩之下，同一条流水线可以有完全不同的部署：Qwen3-Omni 默认每个 stage 一个进程、thinker 和 talker 各占一张卡；单卡部署时五个 GPU stage 挤在一张卡上，声码器干脆跑进 talker 的进程里。这一章读配置的结构（按"谁来读"分成三组）、一套"路径语言"（YAML 和命令行是同一种写法的两种拼法）、配置的来源追踪，以及进程和显存是怎么规划出来的——所有实验都不需要 GPU。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 能不能用 YAML 文件给一个模型多加一个 stage？
    2. `stages.thinker.engine.mem_fraction_static` 和 `stages.thinker.factory.max_seq_len` 分别由谁读取？
    3. YAML 里写了 `max_batch_size: 16`，命令行又写了 `--vocoder.factory.max_batch_size 32`，最后是多少？怎么不启动服务就知道？
    4. 两个 stage 的 `process` 写成同一个名字，意味着什么？有什么代价？
    5. Qwen3-Omni 单卡部署（colocated）时，为什么必须给每个 GPU stage 写 `gpu_memory_fraction`？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 不能。拓扑只在 `config.py` 里定义，配置文件和命令行"只覆盖这些 stage 的设置，从不增删"；写一个不存在的 stage 名会直接报错，并列出真实的 stage 名。
    2. `engine.*` 由 SGLang 的 `ServerArgs` 读（只在驱动 SGLang 引擎的 `EngineStageConfig` 上有）；`factory.*` 按参数名传给 stage 的工厂函数。顶层字段（`gpu`、`process`、`tp_size`……）由父进程读，用来做放置和进程规划。
    3. 32：命令行优先于配置文件。用 `sgl-omni config resolve --show diff` 看最终改了什么，`--show provenance` 或 `sgl-omni config explain <路径>` 看每个值来自哪里、覆盖了谁。
    4. 它们在同一个 OS 进程、同一个 asyncio 事件循环里：数据可以直接传 Python 对象（第六章），但共享一个故障域——任何一个 stage 抛异常，整个进程退出；同一张卡上的调度器构造是串行的，冷启动时间从"取最大"变成"求和"。
    5. 一张卡上的多个进程各自分配显存，谁先启动谁先占。colocated 的放置策略要求每个 GPU stage 声明显存比例（或 KV 字节数），这样每个进程只按自己的份额分配，总和不超过 1；不写就在规划阶段报错，而不是启动到一半 OOM。

## 拓扑只在 config.py 里

```text title="docs/developer_reference/config.md @ 921ea2c8 L16-19"
Pipelines are declared with `PipelineConfig` and `StageConfig` in the model's
`config.py`. Stage topology — which stages exist, how they route, where
requests enter — lives here and only here; config files and CLI flags override
settings on these stages but never add or remove them.
```

这条规矩让拓扑在启动之前就是"可见的"：读一个模型的 `config.py`，就知道它有哪些 stage、怎么连；不同部署之间的差别只在参数上。第三、四章的玩具流水线用的就是这套 `PipelineConfig` / `StageConfig`。

## 三组字段，三个消费者

```python title="sglang_omni/config/schema.py @ 921ea2c8 L319-339"
class StageConfig(BaseModel):
    """Single pipeline stage configuration.

    Stage settings are grouped by consumer: fields at the top level are read
    by the parent process (placement, process planning, wiring), ``engine.*``
    by SGLang ServerArgs, and ``factory.*`` by the stage factory's signature.

    Minimal example::

        StageConfig(name="decode", factory_path="...create_decode", terminal=True)

    Fan-in example::

        StageConfig(
            name="aggregate",
            factory_path="...create_aggregate",
            wait_for=["preprocessor", "image_enc", "audio_enc"],
            merge_fn="...merge_for_thinker",
            next="thinker",
        )
    """
```

```text title="docs/developer_reference/config.md @ 921ea2c8 L70-74"
| Group | Consumer | Examples |
| --- | --- | --- |
| stage top level | parent process: placement, process planning, wiring | `gpu`, `tp_size`, `process`, `gpu_memory_fraction` |
| `engine.*` | SGLang `ServerArgs` (only on `EngineStageConfig` stages) | `mem_fraction_static`, `max_running_requests`, `disable_cuda_graph` |
| `factory.*` | the stage factory's signature | `dtype`, `max_seq_len`, `max_concurrency`, `enable_async_decode` |
```

这种"按谁来读分组"的设计，解决的是一个真实的混乱：一个 stage 的参数里，有的给父进程做规划（这张卡放谁），有的给 SGLang（KV 池占多少显存），有的给工厂函数（声码器的 chunk 大小）。分组之后，每组的词汇表归它的消费者所有——`engine.*` 里认识的 key 就是 `ServerArgs` 的字段，`factory.*` 里认识的 key 就是工厂函数的参数名。**不认识的 key 原样透传**，交给消费者去报错：第四章的实验给工厂传 `batch_wait_when_idle`，`FactoryArgs` 里并没有这个字段，它照样到达了 `create_encoder`。

两条通道汇到工厂函数：

````text title="docs/developer_reference/config.md @ 921ea2c8 L235-245"
```text
PipelineConfig.stage_factory_kwargs(name)      # author channel: code wiring
stage.factory.*                                # config channel: by field name
stage.engine.*  ->  server_args_overrides      # config channel: one dict to SGLang
```

Per key, the config channel wins over the author channel;
`server_args_overrides` merges per key the same way. A configured key the
factory does not accept raises at construction. Standard kwargs
(`model_path`, `gpu_id`, `total_gpu_memory_fraction`) are injected only when
the factory signature declares them; `gpu_id` is owned by placement and is
````

`stage_factory_kwargs` 是模型作者在代码里写死的接线（比如 Qwen3-Omni 的 talker 要知道"声码器是不是在我的进程里"），配置通道按 key 覆盖它。`gpu_id`、显存比例这些由放置决定的值，配置里不允许直接写。

## 路径语言：YAML 和命令行是同一种写法

````text title="docs/developer_reference/config.md @ 921ea2c8 L95-140"
## Setting Values: YAML and CLI

There are exactly two user-facing spellings of one path language.

**YAML** — the `stages:` mapping, keyed by stage name:

```yaml
config_cls: MossTTSPipelineConfig
model_path: OpenMOSS-Team/MOSS-TTS

stages:
  tts_engine:
    tp_size: 2
    engine:
      mem_fraction_static: 0.7
  vocoder:
    factory:
      dtype: bfloat16
```

Entries merge by name: a field the file does not write keeps the model's
default. Naming a stage the config class does not define is an error that
lists the real stage names.

**CLI** — dotted flags with the `stages.` prefix implied; the flag starts
from the stage name exactly as the mapping does:

```bash
sgl-omni serve --config omni.yaml \
    --tts_engine.tp_size 2 \
    --tts_engine.engine.mem_fraction_static 0.7 \
    --vocoder.factory.dtype bfloat16 \
    --vocoder.process vocoder
```

CLI text is coerced by the declared field type; free-form group keys fall
back to YAML scalar parsing (`true` → bool, `7` → int). Writing one path
twice at the same precedence is refused, never silently last-one-wins.
The command line outranks the config file; an explicit dotted path outranks
the broadcast flag below.

**Shared values** — the `shared:` selector list writes one value into several
stages at once:

```yaml
shared:
````

试一下。写一个 Qwen3-TTS 的配置文件，再在命令行上覆盖两个值；`config resolve` 只做解析，不加载模型、不需要 GPU：

```bash title="ch7_resolve.sh"
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
  | grep -E 'does not start|stages of this|did you mean' | sed -E 's/^[│ ]+//; s/[│ ]+$//'
```

```text title="输出"
== --show diff
stages.tts_engine.engine.mem_fraction_static: None -> 0.7
stages.vocoder.factory.max_batch_size: None -> 32
== config explain
stages.tts_engine.engine.mem_fraction_static = 0.7
  None  <- model default - value before any patch was applied
  0.6  <- yaml file (tts.yaml)  [superseded]
  0.7  <- cli flag (--mem-fraction-static)  [winner]
== 写错 stage 名
Invalid value: 'vocodr.factory.dtype' does not start from a stage name or a top-level configuration field of Qwen3TTSPipelineConfig
stages of this pipeline: preprocessing, vocoder, tts_engine
did you mean: vocoder
```

三条规则在这里都看得到：

- **优先级**：模型默认值 < 配置文件 < 命令行。`--mem-fraction-static` 是唯一保留的"广播"开关，一次写进所有 SGLang 引擎 stage 的 `engine.mem_fraction_static`；带 stage 名的点分路径又能覆盖广播。
- **来源追踪**：每个值都记着是谁写的、覆盖了谁（`[winner]` / `[superseded]`）。生产环境排查"这个参数到底生效没有"时，`config explain` 比翻日志可靠。
- **早失败**：写错 stage 名在解析阶段就报错，还会给出最接近的真名。同一路径在同一优先级写两次也会被拒绝，而不是"后写的赢"。

## 进程：谁和谁住在一起

每个非 TP 的 stage 都必须显式写 `process`，名字相同的 stage 住进同一个 OS 进程。`run_process` 的文档写清了这意味着什么：

```python title="sglang_omni/pipeline/stage_workers.py @ 921ea2c8 L544-560"
def run_process(
    spec: StageWorkerProcessSpec,
    ready_event: Event,
    log: logging.Logger,
) -> None:
    """Construct and drive all stages owned by one OS process.

    Multi-stage semantics (since the topology PR):
    - All stages in ``spec.stage_specs`` share this OS process and one asyncio
      event loop. ``asyncio.gather`` runs them concurrently; **if any stage
      raises, the whole process exits** and ``MultiProcessPipelineRunner``'s
      ``_monitor_children`` will fail-all in-flight requests on the
      coordinator. There is no per-stage failure isolation inside one process
      group.
    - Scheduler construction is serialized by :func:`gpu_startup_lock` per GPU
      inside :func:`_construct_scheduler` — so when N stages on the same GPU
      live in this process, cold-start time degrades from ``max`` to ``sum``
```

同进程的好处是第六章看到的直传对象、零拷贝；代价有两个：

- **故障域共享**：一个 stage 崩，整个进程退出，Coordinator 让所有在途请求失败；
- **冷启动串行**：同一张卡上的调度器构造被 `gpu_startup_lock` 串行化（两个 stage 同时初始化 CUDA、同时探测显存会互相干扰），N 个 stage 的启动时间是相加的。

TP（张量并行）的 stage 是例外：每个 rank 独占一个进程，rank 0 是 leader，负责和外界通信，其余 rank 是 follower，通过内部队列接收 leader 分发的工作。

## 动手：不用 GPU 规划一次部署

`prepare_pipeline_runtime` 是启动前的规划步骤：解析放置（哪张卡放哪些 stage、显存比例够不够）、编译进程拓扑（哪些 stage 住一起）、分配 IPC 端点。它是纯计算，可以在没有 GPU 的机器上跑：

```python title="ch7_topology.py"
import os
import warnings

warnings.filterwarnings("ignore")
from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.qwen3_omni.config import (
    Qwen3OmniSpeechColocatedPipelineConfig,
    Qwen3OmniSpeechPipelineConfig,
)
from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig
from sglang_omni.pipeline.runtime_config import prepare_pipeline_runtime


def show(title, config):
    print(f"== {title}")
    prep = prepare_pipeline_runtime(config)
    try:
        for group in prep.process_plan.groups:
            where = "CPU " if group.gpu_id is None else f"GPU{group.gpu_id}"
            print(f"  进程 {group.name:14s} {where}  {', '.join(group.stage_names)}")
        for gpu_id, gpu in sorted(prep.placement_plan.gpus.items()):
            missing = f"；未声明比例：{', '.join(gpu.missing_fraction_stage_names)}" if gpu.missing_fraction_stage_names else ""
            print(f"  GPU{gpu_id}：显存比例合计 {gpu.total_gpu_memory_fraction:.2f}{missing}")
    finally:
        prep.runtime_dir.close()


show("Qwen3-TTS", Qwen3TTSPipelineConfig(model_path="Qwen/Qwen3-TTS-12Hz-0.6B-Base"))
show("Qwen3-Omni 语音，默认（两张卡）", Qwen3OmniSpeechPipelineConfig(model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct"))
yaml_path = os.path.join(os.environ["OMNI_TREE"], "examples/configs/qwen3_omni_colocated_h100_bf16.yaml")
show("Qwen3-Omni 语音，单卡（examples/configs 里的 H100 配置）", ConfigManager.from_file(yaml_path).merge_config([]))
try:
    show("Qwen3-Omni 语音，单卡但不写显存比例", Qwen3OmniSpeechColocatedPipelineConfig(model_path="x"))
except ValueError as exc:
    print("  规划失败：", exc)
```

```text title="输出"
== Qwen3-TTS
  进程 pipeline       GPU0  preprocessing, vocoder, tts_engine
  GPU0：显存比例合计 0.00；未声明比例：tts_engine, vocoder
== Qwen3-Omni 语音，默认（两张卡）
  进程 preprocessing  CPU   preprocessing
  进程 image_encoder  GPU0  image_encoder
  进程 audio_encoder  GPU0  audio_encoder
  进程 thinker        GPU0  thinker
  进程 decode         CPU   decode
  进程 talker_ar      GPU1  talker_ar
  进程 code2wav       GPU0  code2wav
  GPU0：显存比例合计 0.02；未声明比例：audio_encoder, image_encoder, thinker
  GPU1：显存比例合计 0.00；未声明比例：talker_ar
== Qwen3-Omni 语音，单卡（examples/configs 里的 H100 配置）
  进程 preprocessing  CPU   preprocessing
  进程 image_encoder  GPU0  image_encoder
  进程 audio_encoder  GPU0  audio_encoder
  进程 thinker        GPU0  thinker
  进程 decode         CPU   decode
  进程 talker_ar      GPU0  talker_ar, code2wav
  GPU0：显存比例合计 0.94
== Qwen3-Omni 语音，单卡但不写显存比例
  规划失败： Qwen colocated speech requires gpu_memory_fraction or engine.kv_cache_bytes for ['audio_encoder', 'image_encoder', 'talker_ar', 'thinker']
```

读这三种部署：

- **Qwen3-TTS** 把三个 stage 全放进一个叫 `pipeline` 的进程：预处理是 CPU 工作，声码器和 TTS 引擎共用 GPU0，三者之间直传对象。小模型这样部署最省事。
- **Qwen3-Omni 默认**是"一个 stage 一个进程"：编码器、thinker、声码器在 GPU0，talker 独占 GPU1。显存比例大多没写——每张卡上的引擎自己按 `mem_fraction_static` 去分。
- **Qwen3-Omni 单卡**：五个 GPU stage 挤在 GPU0，声码器住进了 talker 的进程（进程 `talker_ar` 里有两个 stage）。配置文件给每个 stage 写了显存比例，thinker 拿走 0.78，合计 0.94。

声码器为什么要跑进 talker 的进程？`config.py` 里的注释给了理由：

```python title="sglang_omni/models/qwen3_omni/config.py @ 921ea2c8 L321-324"
# note (ratish): on one card the GPU time-slices between the stage processes,
# so code2wav decodes inside the talker's process on a priority stream instead
# of waiting for its own turn.
COLOCATED_SPEECH_PROCESSES = {**SPEECH_DEFAULT_PROCESSES, "code2wav": "talker_ar"}
```

一张卡上的多个进程在 GPU 上是时间片轮转的；声码器如果是单独的进程，就要等轮到自己的时间片，首包音频会被拖慢。放进 talker 的进程、用一个高优先级的 CUDA 流，就能在 talker 的两步之间插进去。最后一个例子说明，这种部署对显存预算是"强约束"：不写比例，放置策略在规划阶段就拒绝，而不是等到启动后 OOM。

## 副本、MPS 与权重共享

规划之上还有三个和吞吐相关的开关，都写在 `PipelineConfig` 里：

```python title="sglang_omni/config/schema.py @ 921ea2c8 L257-266"
class ProcessConfig(BaseModel):
    """Replica policy for one logical process.

    Keyed by Process Name in ``PipelineConfig.processes``. Member stages come
    from ``StageConfig.process``, so this never repeats them.
    """

    model_config = ConfigDict(extra="forbid")
    num_replicas: int = 1
    replica_devices: list[int] | None = None
```

- **进程副本**：`processes.<进程名>.num_replicas` 让一个逻辑进程起多份（可以用 `replica_devices` 指定放在哪些卡上），Coordinator 按请求轮流绑定到某个副本（第三章 `assign_replica_bindings`）。TTS 这类小模型在一张大卡上起两三份，吞吐几乎线性增长。
- **MPS**（`mps: off | on | auto`）：NVIDIA 的多进程服务，让同一张卡上的多个进程真正并发地共享 SM，而不是时间片轮转。`sglang_omni/mps/` 负责在启动前为每张卡申请一个 MPS 守护进程的"租约"，停止时归还。
- **权重共享**（`weight_share: on`）：同一张卡上的多个副本只加载一份权重：

```python title="sglang_omni/pipeline/weight_share.py @ 921ea2c8 L1-13"
# SPDX-License-Identifier: Apache-2.0
"""Runtime-owned leader/follower assignment for CUDA IPC weight sharing.

Replicas of one logical Process that land on the same physical GPU are a
sharing group: the lowest replica index loads the checkpoint and publishes
CUDA-IPC handles, the others alias them. This module only plans; the mechanism
itself lives in :mod:`sglang_omni.utils.ipc_weights` and is driven by the same
environment contract an external supervisor would use.

Planning happens in the parent, before the coordinator binds and before any
child is spawned, so an unusable configuration fails in milliseconds instead of
after a leader has loaded a full checkpoint.
"""
```

leader 加载检查点并发布 CUDA IPC 句柄，follower 直接映射。注意最后一段：规划在父进程、子进程启动之前完成，配置有问题几毫秒内就失败，而不是等 leader 加载完一整个检查点之后——"尽早失败"是 omni 配置层反复出现的设计原则。

## 练习

**1. 给 Qwen3-TTS 起两份副本。** 在 `ch7_topology.py` 里给 `Qwen3TTSPipelineConfig` 加上 `processes={"pipeline": {"num_replicas": 2}}`，看规划出来的进程组有什么变化。

??? success "参考思路"
    `Qwen3TTSPipelineConfig(model_path=..., processes={"pipeline": ProcessConfig(num_replicas=2)})`（或直接传字典让 pydantic 解析）。规划结果里会出现两份 `pipeline` 进程（名字带副本编号），每份都包含三个 stage；`prep.replica_topology` 记录逻辑 stage 名到实例名的映射，Coordinator 用它把一个请求的所有 stage 绑定到同一个副本。

**2. 找出"拓扑不可改"在哪里被强制执行。** 在 `sglang_omni/config/` 里找到"命令行写了一个不存在的 stage 名"时抛错的代码。

??? success "参考思路"
    `git grep -n 'does not start from a stage name' 921ea2c8 -- sglang_omni/config/`，在 `path.py` 的路径编译里：点分路径的第一段必须是一个已有的 stage 名或顶层字段名，否则报错并用编辑距离给出建议。

**3. 读一个部署配置。** 打开 `examples/configs/qwen3_asr_rtx5090.yaml`，说出每一行属于哪个消费者组、会被谁读到。

??? success "参考思路"
    `config_cls`、`model_path` 是顶层；`stages.asr.factory.dtype` 给工厂函数；`stages.asr.engine.*`（`max_running_requests`、`cuda_graph_max_bs`、`mem_fraction_static`、`enable_torch_compile`）是 SGLang 的 `ServerArgs`。第八章之后你会在 16 GB 的卡上自己调这几项。

!!! interview "怎么讲清楚"
    讲 omni 的配置，先讲规矩：拓扑只在 `config.py` 里，文件和命令行只覆盖参数。再讲结构：字段按消费者分三组（父进程的放置、SGLang 的 `ServerArgs`、工厂函数的参数），认识的字段早校验、不认识的透传给消费者；YAML 和点分命令行是同一种路径语言，优先级是默认 < 文件 < 命令行，每个值都有来源追踪。最后讲部署：`process` 决定谁住一起（直传对象 vs 共享故障域），放置策略在启动前检查显存预算，副本、MPS、权重共享提升吞吐——所有检查都在子进程启动前完成。

## 小结

- [x] 拓扑只在 `config.py` 里定义；配置文件和命令行只能覆盖已有 stage 的参数。
- [x] 字段三组：顶层（父进程：放置、进程）、`engine.*`（SGLang `ServerArgs`）、`factory.*`（工厂函数参数）；不认识的 key 透传。
- [x] 路径语言：YAML 的 `stages:` 映射 = 命令行的 `--<stage>.<组>.<字段>`；默认 < 文件 < 命令行；`config resolve` / `explain` 看结果和来源。
- [x] 同 `process` = 同进程同事件循环：直传对象，但共享故障域、启动串行；TP 的每个 rank 独占进程。
- [x] `prepare_pipeline_runtime` 在启动前规划放置和进程；colocated 必须声明显存比例；副本、MPS、权重共享都在这一层配置。
