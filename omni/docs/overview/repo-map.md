# 仓库地图：目录、代码量与项目节奏

<p class="lead">第一次打开 sglang-omni 的仓库，很容易被吓到：一千多个文件，Python、Rust、Swift 三种语言，25 个模型目录。其实框架本身不大——22 万行 Python 里三分之二是模型代码，真正的运行时（流水线、调度、通信、配置）加起来不到四万行。这一章先用命令把仓库量一遍：每个目录多大、谁依赖谁、模型目录长什么样、文档测试 CI 各在哪；再看项目的节奏，你会知道为什么"盯着 issue 抢活"在这里行不通。最后给一份读代码的路线和工具箱。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. `sglang_omni/` 里哪个子目录最大？框架本身（不算模型）大约多少行？
    2. 一个新模型的代码应该放在哪里，至少有哪几个文件？
    3. 测试代码和产品代码哪个多？
    4. 2026 年 9 月这个仓库合了多少提交、有多少位作者？

??? success "自测参考答案（先自己答，再展开对照）"
    1. `models/`，约 14.8 万行，占三分之二。运行时的核心——`pipeline/`、`scheduling/`、`model_runner/`、`comm/`、`relay/`、`config/`、`proto/`——加起来约 3.9 万行；再加上 `serve/` 和 `client/` 的 API 层约 5.6 万行。
    2. `sglang_omni/models/<模型>/`：`config.py`（PipelineConfig 子类和 stage 列表）、`stages.py`（stage 工厂）、`request_builders.py`（stage 之间的 payload 变换），模型模块放 `components/`。
    3. 差不多一样多：`tests/` 约 22.3 万行，`sglang_omni/` 约 22.3 万行。
    4. 242 个提交、66 位作者（按提交作者名去重）。

## 顶层目录

```bash title="top-level.sh"
cd "$OMNI_TREE"
for d in sglang_omni sglang_omni_router sglang_omni_mlx tests benchmarks docs examples playground scripts Voxt OmniTyper; do
  files=$(find "$d" -type f -not -path '*/__pycache__/*' | wc -l)
  lines=$(find "$d" -type f \( -name '*.py' -o -name '*.rs' -o -name '*.swift' -o -name '*.md' -o -name '*.yaml' -o -name '*.toml' \) -not -path '*/__pycache__/*' -print0 | xargs -0 cat | wc -l)
  printf '%-20s %5d 个文件 %7d 行\n' "$d" "$files" "$lines"
done
```

```text title="输出"
sglang_omni            739 个文件  223343 行
sglang_omni_router      72 个文件   37383 行
sglang_omni_mlx        110 个文件    4698 行
tests                  626 个文件  222907 行
benchmarks             118 个文件   44180 行
docs                    95 个文件   16440 行
examples                66 个文件    3377 行
playground              48 个文件    2630 行
scripts                  8 个文件    1475 行
Voxt                  1010 个文件  212360 行
OmniTyper               42 个文件    6146 行
```

逐个说一下：

| 目录 | 是什么 | 本书在哪讲 |
| --- | --- | --- |
| `sglang_omni/` | 主包：运行时、调度、通信、配置、API、全部模型 | 第三～九章 |
| `sglang_omni_router/` | 多 worker 前面的路由器，`python/` 是旧实现，`rust/` 是新实现 | 第十章 |
| `sglang_omni_mlx/` | Apple Silicon 上用 MLX 跑 Qwen3-ASR 的实现 | 本章末尾 |
| `tests/` | 单测、模型级 CI 测试、测试数据 | 第十一章 |
| `benchmarks/` | 各模型的评测与压测脚本（WER、TTFA、吞吐） | 第八、十二章 |
| `docs/` | 官方文档站（cookbook、开发者参考、设计文档） | 贯穿全书 |
| `examples/` | 部署配置（`examples/configs/*.yaml`）和客户端示例 | 第七章 |
| `Voxt/`、`OmniTyper/` | 两个 macOS 语音输入 App（Swift），调用本机的 omni 服务 | 不讲 |

注意 `tests/` 和 `sglang_omni/` 几乎一样大。这个项目的测试密度非常高，第十一章会看到，一个合格的 PR 往往是"几十行改动 + 上百行测试"。

## sglang_omni/ 里面

```bash title="package.sh"
cd "$OMNI_TREE/sglang_omni"
for d in */; do
  d=${d%/}
  [ "$d" = __pycache__ ] && continue
  n=$(find "$d" -name '*.py' -not -path '*/__pycache__/*' -print0 | xargs -0 cat | wc -l)
  echo "$n $d"
done | sort -rn | awk '{printf "%-14s %7d\n", $2, $1}'
```

```text title="输出"
models          148275
serve            14883
scheduling       13703
pipeline          8248
utils             5743
model_runner      4967
config            4682
relay             3351
vendor            3349
comm              3285
preprocessing     2607
mps               2213
client            1765
profiler          1390
platforms         1243
proto             1135
cli                957
diagnostics        521
http               105
sampling            53
```

![图：sglang_omni/ 的分层](../assets/figures/omni-layers.svg){.aig-svg}

按"一条请求经过的顺序"把它们排成几层，这也是第二部分的章节顺序：

1. **API 层**：`serve/`（OpenAI 兼容接口、语音和转录的各种端点、realtime WebSocket）、`client/`（把 HTTP 请求变成内部请求、把结果拼回来）、`cli/`（`sgl-omni serve`、`sgl-omni config`）。→ 第三章
2. **编排层**：`pipeline/`（Coordinator、Stage、进程管理）、`proto/`（请求和消息类型）。→ 第三、四章
3. **调度层**：`scheduling/`（四类调度器，其中 `OmniScheduler` 嵌着 SGLang）、`model_runner/`（自回归 stage 的前向路径）、`vendor/`（SGLang 补丁）。→ 第四、五章
4. **通信层**：`comm/`（按边选传输、打包）、`relay/`（SHM、CUDA IPC、Mooncake 等后端）。→ 第六章
5. **配置与部署**：`config/`（PipelineConfig、路径语言、来源追踪）、`mps/`（NVIDIA MPS 的租约管理）、`platforms/`（CUDA、ROCm、XPU、NPU、Apple 等硬件差异）。→ 第七章
6. **模型**：`models/`，每个模型一个目录。→ 第八、九章

剩下的 `utils/`、`preprocessing/`（音视频解码、重采样）、`profiler/`、`diagnostics/` 是横向的工具。

## 模型目录的约定

官方开发者文档里写了模型目录的推荐结构：

````text title="docs/developer_reference/main.md @ 921ea2c8 L42-61"
## Model Directory Convention

Model-specific code should stay under `sglang_omni/models/<model>/`.

Recommended layout:

```text
models/<model>/
|-- config.py             # PipelineConfig subclass and StageConfig list
|-- stages.py             # stage factories
|-- routing.py            # optional data-driven routing helpers
|-- request_builders.py   # inter-stage payload transforms
|-- payload_types.py      # typed model-specific payload state
|-- callbacks.py          # feedback callbacks or strategy, when needed
`-- components/           # model modules, processors, vocoders, adapters
```

Only model-local behavior belongs here. The framework-owned layers are still
`Stage`, `Coordinator`, schedulers, model-runner bases, relay, runtime prep, and
runners.
````

看一个真实的例子，Qwen3-TTS：

```bash title="qwen3-tts-files.sh"
git ls-tree -r --name-only "$REF" sglang_omni/models/qwen3_tts/ | sed 's#sglang_omni/models/qwen3_tts/##'
```

```text title="输出"
__init__.py
codec_state_arena.py
compat.py
config.py
engine_builder.py
incremental_codec.py
incremental_codec_cuda_graph.py
model_runner.py
payload_types.py
predictor_kernels.py
prompt_frontend.py
reference_encoder_cuda_graph.py
request_builders.py
sampling_kernels.py
sglang_model.py
speaker_encoder_cuda_graph.py
stages.py
streaming_vocoder.py
```

`config.py`、`stages.py`、`request_builders.py`、`payload_types.py` 是约定里的那几个；`engine_builder.py` 是自回归 stage 的构建器（第五章），`sglang_model.py` 是用 SGLang 并行层写的 talker 模型，`streaming_vocoder.py` 是流式声码器调度器（第八章），其余是 CUDA Graph、kernel 之类的性能优化。

25 个模型目录各声明一个（或几个）HF 架构名，启动时 omni 读 HF `config.json` 里的 `architectures` 字段，按它找到对应的 PipelineConfig：

```bash title="architectures.sh"
git grep -hE '^\s+architecture: ClassVar\[str( \| None)?\] = "' "$REF" -- 'sglang_omni/models/*/config.py' \
  | sed -E 's/.*= "([^"]+)".*/\1/' | sort | paste -sd' ' | fold -s -w 96
```

```text title="输出"
ArkasrForConditionalGeneration AuKForConditionalGeneration AudarTTSForConditionalGeneration 
BailingMM2NativeForConditionalGeneration BailingMMNativeForConditionalGeneration 
DotsTTSForConditionalGeneration FishQwen3OmniForCausalLM FunAsrNanoForConditionalGeneration 
FunCosyVoice3SGLangModel HiggsMultimodalQwen3ForConditionalGeneration LLaDA2MoeModelLM MiniCPMO 
MiniMaxMusic3ForConditionalGeneration MossTTSDelayModel MossTTSLocalModel 
MossTranscribeDiarizeForConditionalGeneration Nemotron3_5AsrForRNNT 
NemotronVoiceChatForCausalLM Qwen3ASRForConditionalGeneration 
Qwen3OmniMoeForConditionalGeneration Qwen3TTSForConditionalGeneration 
VoxtralTTSForConditionalGeneration WhisperForConditionalGeneration Zonos2ForCausalLM
```

## 文档、测试和 CI 在哪

```bash title="docs-tests-ci.sh"
echo "docs/ 的栏目："; git ls-tree -d --name-only "$REF" docs/ | sed 's#docs/#  #'
echo "cookbook 篇数：$(git ls-tree --name-only "$REF" docs/cookbook/ | grep -c '\.md$')"
echo "单测目录数：$(git ls-tree -d --name-only "$REF" tests/unit_test/ | wc -l)"
echo "单测文件数：$(git ls-tree -r --name-only "$REF" tests/unit_test/ | grep -c '/test_.*\.py$')"
echo "CI workflow 数：$(git ls-tree --name-only "$REF" .github/workflows/ | wc -l)"
```

```text title="输出"
docs/ 的栏目：
  _static
  basic_usage
  benchmarks
  cookbook
  design
  developer_reference
  get_started
cookbook 篇数：24
单测目录数：54
单测文件数：511
CI workflow 数：19
```

读文档的顺序建议：`docs/developer_reference/`（架构、流水线、通信、配置，和本书第二部分对应）→ `docs/cookbook/` 里你关心的模型 → `docs/design/`（重构 RFC，讲"为什么这么设计"）。

## 项目节奏

```bash title="pace.sh"
echo "月份      提交   作者"
git log "$REF" --date=format:%Y-%m --format='%ad|%aN' | awk -F'|' '{c[$1]++; if (!seen[$0]++) a[$1]++} END {for (m in c) printf "%s  %5d  %5d\n", m, c[m], a[m]}' | sort
echo "提交总数：$(git rev-list --count "$REF")；标题以 (#PR号) 结尾的：$(git log "$REF" --format=%s | grep -cE '\(#[0-9]+\)$')"
echo "第一个提交：$(git log --reverse --date=short --format='%ad %s' "$REF" | head -1)"
```

```text title="输出"
月份      提交   作者
2026-01     49      6
2026-02     39     11
2026-03     34     12
2026-04     61     17
2026-05    125     25
2026-06    119     30
2026-07    133     34
2026-08    248     54
2026-09    242     66
2026-10    105     35
提交总数：1155；标题以 (#PR号) 结尾的：1106
第一个提交：2026-01-06 Initial commit
```

几个读法：

- 2026-01-06 建仓，前四个月是少数人搭骨架；5 月开始提速，8、9 月每月两百四十多个提交、五六十位作者。
- 几乎每个提交都对应一个 PR（squash 合并）。PR 编号已经到两千六百多，说明有大量 PR 没有合入或被关闭——提 PR 不等于能合。
- 作者数涨得比提交数快：越来越多的人来做"认领一个小问题 → 提一个 PR"。这就是上一次我们看到的现象：标着 `good first issue` 的问题几小时内就被认领。**想在这里做出有分量的贡献，要靠自己读代码找问题**，这正是第十二章的主题。

## 读代码的工具箱

全书用到的几样工具，先列在这里：

```bash title="toolbox.sh" run="no"
# 1. 在固定提交上查东西：不受工作区影响
git grep -n 'class OmniScheduler' 921ea2c8 -- sglang_omni/
git show 921ea2c8:sglang_omni/pipeline/stage/input.py | sed -n '44,60p'

# 2. 一个文件 / 一个函数是怎么变成今天这样的
git log --oneline --follow -- sglang_omni/scheduling/simple_scheduler.py | head
git log -S 'batch_wait_when_idle' --oneline -- sglang_omni/

# 3. 不启动服务、不需要 GPU，看一个模型的流水线最终长什么样
sgl-omni config resolve --model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base --show diff --vocoder.factory.max_batch_size 16

# 4. 打开通信层的追踪日志：每条边第一次选了什么传输方式
SGLANG_OMNI_COMM_TRACE=1 sgl-omni serve ...
```

另外，本书第三章起的实验都依赖一个**不需要 GPU 的 omni 环境**：CPU 版 PyTorch + `--no-deps` 安装的 SGLang（只用它的 Python 部分，不装 CUDA 内核）。安装方法见[首页的"环境"一节](../index.md#环境)。

## 不讲的部分：MLX 与 Swift

`sglang_omni_mlx/` 和各模型目录下的 `mlx/` 子目录，是 Apple Silicon 上的实现：用 MLX 重写了 Qwen3-ASR 等模型的前向，引擎侧借 SGLang 上游的 MLX 后端（`MlxTpModelWorker`）。仓库里没有自写的 Metal kernel，融合算子用的是 MLX 自带的 `mx.fast.scaled_dot_product_attention`、`mx.fast.rope`、`mx.fast.layer_norm`：

```bash title="mlx-fast.sh"
git grep -hoE 'mx\.fast\.[a-z_]+' "$REF" -- sglang_omni sglang_omni_mlx | sort | uniq -c
echo ".metal 文件数：$(git ls-tree -r --name-only "$REF" | grep -c '\.metal$')"
```

```text title="输出"
      1 mx.fast.layer_norm
      3 mx.fast.rope
      5 mx.fast.scaled_dot_product_attention
.metal 文件数：0
```

注意一个容易看错的命名：`sglang_omni/mps/` 是 **NVIDIA MPS**（Multi-Process Service，多个进程共享一张卡），而 `model_runner/audio_torch_mps.py` 里的 MPS 是 **Apple 的 Metal Performance Shaders**（PyTorch 的 `mps` 设备）。

## 练习

**1. 框架有多大？** 写一条命令，算出 `sglang_omni/` 去掉 `models/` 之后的 Python 行数，再算 `models/` 里最大的三个模型目录。

??? success "参考思路"
    `cd "$OMNI_TREE/sglang_omni" && find . -name '*.py' -not -path './models/*' -print0 | xargs -0 cat | wc -l`；模型目录用 `for d in models/*/; do echo "$(find $d -name '*.py' -print0 | xargs -0 cat | wc -l) $d"; done | sort -rn | head -3`。最大的通常是 `qwen3_omni`、`ming_omni`、`minicpm_o` 这类全模态模型。

**2. 一个目录的"热度"。** 统计 2026-09 一个月里，`sglang_omni/` 的每个一级子目录各被多少个提交改过。哪些目录最活跃？这对你选贡献方向有什么提示？

??? success "参考思路"
    `git log 921ea2c8 --since=2026-09-01 --until=2026-10-01 --name-only --format= -- sglang_omni | cut -d/ -f2 | sort | uniq -c | sort -rn`（注意 `--since` 只写日期时的漂移问题，严格统计可以先打印 `%ad` 再按日期前缀过滤）。`models/` 最活跃是自然的；运行时目录里改动多的地方，往往也是 bug 和重构集中的地方。

**3. 找一个模型的入口。** 只给你一个 HF 模型名 `Qwen/Qwen3-ASR-1.7B`，不读文档，怎么找到它对应的 PipelineConfig 和 stage 列表？

??? success "参考思路"
    先看它的 `config.json` 里 `architectures` 是 `Qwen3ASRForConditionalGeneration`，再 `git grep -n '"Qwen3ASRForConditionalGeneration"' 921ea2c8 -- 'sglang_omni/models/*/config.py'` 找到 `models/qwen3_asr/config.py`，读里面的 `stages` 列表；或者直接 `sgl-omni config resolve --model-path Qwen/Qwen3-ASR-1.7B`。

!!! interview "怎么讲清楚"
    讲一个陌生仓库的结构，先讲量级（多少行、几种语言、哪块最大），再讲分层（API → 编排 → 调度 → 通信 → 配置 → 模型），最后讲约定（新模型放哪、至少几个文件）。量级要用命令算出来，而不是估——"三分之二是模型代码、运行时不到四万行"这样的数字比"代码很多"有用得多。

## 小结

- [x] `sglang_omni/` 约 22 万行，模型占三分之二；运行时核心约 3.9 万行，测试和产品代码一样多。
- [x] 分层：API（serve / client / cli）→ 编排（pipeline / proto）→ 调度（scheduling / model_runner / vendor）→ 通信（comm / relay）→ 配置部署（config / mps / platforms）→ 模型。
- [x] 新模型放 `models/<模型>/`：`config.py`、`stages.py`、`request_builders.py`、`components/`；按 HF 架构名注册。
- [x] 项目节奏很快（2026-09：242 个提交、66 位作者），简单的问题会被迅速认领——要靠读代码找切入点。
