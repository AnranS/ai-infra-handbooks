# 2026 年的代码库快照

<p class="lead">基准提交 <code>29f6d408c0</code> 的 <code>srt/</code> 有 1979 个 Python 文件，v0.5.21 的运行时超过 84 万行；2026 年前九个月有 1200 多位作者、1 万多个提交，平均每月 1200 个，是 2024 年的 20 倍。这一章不讲某个功能，而是把今天的代码库当成一张地图：哪些目录是 2025 年下半年之后才有的、每一类目录在解决什么问题、路线图里"兼容与可靠性"的重点怎样体现在提交里，以及一个每两三周发一个版本的项目靠什么不散架。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 2024、2025、2026 三年的提交量和作者数各是多少量级？这说明项目处于什么阶段？
    2. 基准提交里最大的几个目录是什么？哪些是 2025 年下半年之后才出现的？
    3. 2025 年 Q3 路线图把重点从"功能"转向"兼容与可靠性"，在提交里能看到哪些对应？
    4. 一个每两三周发版、上千人贡献的仓库，靠什么机制保持可维护？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 2024 年约 1600 个提交、近 190 位作者；2025 年约 6800 个、近 800 位；2026 年到 10 月初约 1 万 1 千个、1200 多位。从"几个作者的论文代码"到"社区项目"再到"平台级项目"。
    2. `layers/`（400 个文件）、`models/`（287）、`mem_cache/`（156）、`multimodal/` 与 `hardware_backend/`（各 95）、`debug_utils/`（91）、`configs/` 与 `arg_groups/`（各 79）。2025-10 之后新出现的有 `compilation/`、`elastic_ep/`、`checkpoint_engine/`、`dllm/`、`hardware_backend/`、`observability/`、`arg_groups/`、`scheduler_components/`、`kv_canary/`、`weight_cache/`、`beam_search/`、`rust_extensions/`、`kernels/`。
    3. 分配逻辑从调度器拆出（#11313）、piecewise CUDA graph（#10062 及后续）、重叠调度与投机解码的 v2、弹性 EP 与故障 rank（#10423 等）、`debug_utils/` 的张量对比工具、`kv_canary/` 的缓存一致性检查、`observability/` 的整理——都是让系统在更多组合下稳定、可诊断。
    4. 按目录触发的 CI、模块边界（第 10 章的分层）、mixin 与插件化（注意力后端、投机方法注册表、淘汰策略、传输后端）、参数分组（`arg_groups/`）、一致性测试（Python 与 Rust 树的 parity tests）、以及每个版本一个 tag 的节奏。

先看一个六格小剧场，再读正文：

![漫画：一张 2026 年的地图](../assets/comics/codebase-2026.webp){.aig-comic}

## 三年的量级

```bash title="yearly-stats.sh"
REF=${REF:-29f6d408c0}
for y in 2024 2025 2026; do
  printf '%s  提交 %5d  作者 %4d\n' "$y" "$(git rev-list --count --since=$y-01-01 --until=$y-12-31 "$REF")" "$(git shortlog -sn --since=$y-01-01 --until=$y-12-31 "$REF" | wc -l)"
done
echo "累计作者：$(git shortlog -sn "$REF" | wc -l)"
echo "srt/ 的 .py 文件：$(git ls-tree -r --name-only "$REF" -- python/sglang/srt | grep -c '\.py$')；test/ 的 .py 文件：$(git ls-tree -r --name-only "$REF" -- test | grep -c '\.py$')"
echo "2026 年的版本 tag（不含网关）：$(git for-each-ref --sort=creatordate --format='%(creatordate:short) %(refname:short)' refs/tags | grep '^2026' | grep -vc gateway) 个，从 $(git for-each-ref --sort=creatordate --format='%(creatordate:short) %(refname:short)' refs/tags | grep '^2026' | grep -v gateway | head -1) 到 $(git for-each-ref --sort=creatordate --format='%(creatordate:short) %(refname:short)' refs/tags | grep '^2026' | grep -v gateway | tail -1)"
```

```text title="输出"
2024  提交  1607  作者  189
2025  提交  6767  作者  796
2026  提交 10831  作者 1214
累计作者：1877
srt/ 的 .py 文件：1979；test/ 的 .py 文件：2515
2026 年的版本 tag（不含网关）：21 个，从 2026-01-01 v0.5.7 到 2026-09-30 v0.5.21
```

[首页](../index.md)的时间线控件画的就是这条曲线：2024 年每月几十到两三百，2025 年四百到九百，2026 年一千到一千六。测试文件比运行时文件还多，是这个阶段最直观的特征。

## 目录地图

```bash title="srt-dirs-2026.sh"
REF=${REF:-29f6d408c0}
git ls-tree -r --name-only "$REF" -- python/sglang/srt | grep '\.py$' | awk -F/ 'NF > 4 {print $4}' | sort | uniq -c | sort -rn | head -24 | awk '{printf "%4d  %s\n", $1, $2}'
```

```text title="输出"
 400  layers
 287  models
 156  mem_cache
  95  multimodal
  95  hardware_backend
  91  debug_utils
  79  configs
  79  arg_groups
  66  entrypoints
  65  utils
  62  speculative
  58  model_executor
  53  managers
  50  kv_canary
  47  function_call
  46  lora
  36  disaggregation
  30  distributed
  23  model_loader
  15  parser
  15  observability
  14  eplb
  13  compilation
  12  sampling
```

按本书的分期给每个目录标上"出生年份"：

```bash title="new-dirs-2026.sh"
REF=${REF:-29f6d408c0}
for d in compilation elastic_ep checkpoint_engine dllm hardware_backend observability arg_groups kv_canary weight_cache beam_search rust_extensions; do
  printf '%-18s %s\n' "$d" "$(git log --reverse --date=short --format='%ad  %h  %s' "$REF" -- python/sglang/srt/$d | head -1 | cut -c1-78)"
done
printf '%-18s %s\n' "kernels (aot)" "$(git log --reverse --date=short --format='%ad  %h  %s' "$REF" -- python/sglang/kernels | head -1 | cut -c1-78)"
printf '%-18s %s\n' "scheduler_components" "$(git log --reverse --date=short --format='%ad  %h  %s' "$REF" -- python/sglang/srt/managers/scheduler_components | head -1 | cut -c1-78)"
```

```text title="输出"
compilation        2025-10-13  932e263725  Compilation Folder Reset (#11539)
elastic_ep         2025-10-22  904655c5fd  [2/N] Added the core structure of elastic EP and the e
checkpoint_engine  2025-10-24  96a5e4dd79  [Feature] Support loading weights from ckpt engine wor
dllm               2025-11-26  21b0582d4b  [feature] Initial block diffusion language model supp
hardware_backend   2025-12-04  894c0dc57c  [NPU][1/N] NPU basic functions refactor and new models
observability      2026-02-25  3b89302277  Refactor: observability code cleanup (#17862)
arg_groups         2026-05-03  00d620b77d  introduce arg_groups/ with nemotron_h hook (#24328)
kv_canary          2026-05-31  11391b2a1c  Add the KV-canary core: data layer, MHA KV-pool patche
weight_cache       2026-07-25  f9c14e6bd4  [FEAT] Support fast engine recovery through weight cac
beam_search        2026-08-26  ec4bdbfa4a  [Feature] Beam search support (#31626)
rust_extensions    2026-08-16  67e12131df  Build Rust extensions on demand in source checkouts (#
kernels (aot)      2026-07-10  6ed9843b57  [Kernel] Introduce sglang.kernels namespace and migrat
scheduler_components 2026-05-18  062f6f7ae8  Move get_draft_kv_pool to mem_cache.kv_cache_builder (
```

![图：srt 目录按出生时期着色](../assets/figures/sgl-codebase-2026.svg){.aig-svg}

分四类看：

- **2024 年定型的骨架**：`managers/`、`mem_cache/`、`model_executor/`、`layers/`、`models/`、`sampling/`、`constrained/`——[第 10 章](../perf/restructure.md)的产物，今天仍是最大的几个目录。
- **2025 年上半年的规模化**：`speculative/`、`disaggregation/`、`distributed/`、`eplb/`、`lora/`、`entrypoints/`、`function_call/`、`platforms/`。
- **2025 年下半年的可靠性与可观测**：`compilation/`（piecewise CUDA graph 与 torch.compile 后端的归宿，#11539 "Compilation Folder Reset"）、`elastic_ep/`（故障 rank 与弹性扩缩）、`checkpoint_engine/`、`debug_utils/`（对比内部激活张量的工具，#7976）、`hardware_backend/`（各家硬件的适配集中，始于 2025-12 的 NPU 重构）。
- **2026 年的平台化**：`arg_groups/`（几百个启动参数按组整理）、`scheduler_components/`（把近 6000 行的 `scheduler.py` 继续拆分）、`kv_canary/`（KV 缓存的一致性"金丝雀"检查）、`observability/`、`weight_cache/`（引擎快速恢复）、`beam_search/`、`rust_extensions/`（按需编译的 Rust 组件，[第 21 章](gateway.md)）、`kernels/`（sgl-kernel 搬进主包，[第 14 章](../perf/sgl-kernel.md)）。

## 从功能到可靠性

2025 年 Q3 的路线图（issue #7736）把标题写成"兼容与可靠性"，代码里的对应：

```bash title="reliability-commits.sh"
REF=${REF:-29f6d408c0}
for h in b36afed4a7 4ac8e09df0 1801cd199f a40229f6f8 904655c5fd 932e263725 ed0fdbf35b 11391b2a1c 00d620b77d 062f6f7ae8; do
  git log -1 --date=short --format='%ad  %h  %s' $h | cut -c1-96
done
```

```text title="输出"
2025-10-10  b36afed4a7  Separate allocation logic from scheduler (#11313)
2025-10-11  4ac8e09df0  Piecewise CUDA Graph Support & Torch Compile Backend (#10062)
2025-10-23  1801cd199f  support more model in piecewise cuda graph (#11745)
2025-10-15  a40229f6f8  [1/N] Introduce Mooncake Backend and Mooncake EP to Support Elastic EP (
2025-10-22  904655c5fd  [2/N] Added the core structure of elastic EP and the eplb algorithm with
2025-10-13  932e263725  Compilation Folder Reset (#11539)
2025-07-27  ed0fdbf35b  Tool to dump and compare internal activation tensors (#7976)
2026-05-31  11391b2a1c  Add the KV-canary core: data layer, MHA KV-pool patcher, and per-forward
2026-05-03  00d620b77d  introduce arg_groups/ with nemotron_h hook (#24328)
2026-05-18  062f6f7ae8  Move get_draft_kv_pool to mem_cache.kv_cache_builder (#25602)
```

- **把大文件拆小。** #11313 把分配逻辑从调度器拆出；`scheduler_components/`、各类 mixin（输出处理、PP、控制消息）让 `scheduler.py` 停止膨胀。
- **让 CUDA Graph 覆盖更多形状。** piecewise CUDA graph（#10062 起）把前向切成可独立捕获的段，prefill 的非注意力部分也能图化；`compilation/` 成为它和 torch.compile 的共同归宿。
- **让系统能在故障下运行。** 弹性 EP（#10423、#11837 系列）允许 EP 组里有 rank 掉线、也允许运行中扩缩。
- **让问题能被定位。** `debug_utils/` 的张量 dump 与对比、`kv_canary/` 在 KV 池里埋检查值、`observability/` 的指标与追踪整理。
- **让参数可管理。** `arg_groups/` 把 `server_args.py` 里几百个参数按功能分组，每个组自己校验。

这些工作不出现在发布博客的标题里，却是上千人同时改一个仓库的前提。

## 发布节奏与贡献者

v0.5 系列从 2025 年 8 月的 rc0 到 2026 年 9 月 30 日的 v0.5.21，50 个 tag，2026 年里 21 个、大约每两周一个；网关从 2025 年 7 月起另有 12 个 `gateway-v*`。贡献者的构成也变了：2024 年前几位作者占了大半提交（[第九章](../service/v02.md)的 `v02-tag.sh`），2026 年 1200 多位作者里没有谁超过几个百分点。路线图从一个人写的 issue（2024 Q3 的 #634）变成按方向分工的多份 issue（Diffusion 的 #12799 由另一位维护者负责）。

## 设计取舍

- **拆分优先于重写。** 近 6000 行的 `scheduler.py` 没有被推倒重来，而是用 mixin、components、独立模块一块块搬出去——与 2024 年 7 月那次整体重组（[第 10 章](../perf/restructure.md)）不同，大仓库承受不起全局重构。
- **插件化换扩展性。** 注意力后端、投机方法、淘汰策略、传输后端、硬件后端、工具解析器都是注册表 + 实现文件；代价是概念多、新人要先学一遍注册表。
- **测试比代码多。** 2500 多个测试文件、按目录触发的 CI、Python 与 Rust 的一致性测试。
- **仓库里的"说明给机器看"。** `multimodal_gen/` 下有 `.agents` 和 `.claude` 目录——给 AI 编码助手的项目说明，是 2026 年大型开源项目的新特征之一。

## 后来怎么样了

本书的基准止于 2026-10-02。之后的变化可以用[第 26 章](../method/archaeology.md)的脚本自己跑：换一个 `REF`，所有数字会更新。

## 练习

**1. 最活跃的目录。** 用 `git log --since=2026-01-01 --format= --name-only "$REF" | grep '^python/sglang/srt/' | cut -d/ -f4 | sort | uniq -c | sort -rn | head` 找出 2026 年改动最多的 srt 子目录，和文件数最多的目录比较。

??? success "参考思路"
    改动最多的通常是 `layers/`、`models/`、`managers/`、`mem_cache/`；`models/` 文件多但单个文件改动少（新模型一次性加入），`managers/` 文件少但改动频繁。

**2. 一个 mixin 的来历。** 选 `managers/scheduler_pp_mixin.py`，用 `git log --follow` 看它是什么时候从 `scheduler.py` 里拆出来的，拆出前后 `scheduler.py` 的行数各是多少。

??? success "参考思路"
    找到拆分提交后，用 `git show <提交>^:python/sglang/srt/managers/scheduler.py | wc -l` 和 `git show <提交>:... | wc -l` 比较。

**3. 作者结构。** 用 `git shortlog -sn --since=2026-01-01 "$REF" | head -20` 看前 20 位作者占全年提交的比例，和 2024 年比较。

??? success "参考思路"
    2024 年前 3 位作者超过一半；2026 年前 20 位合计也不到一半——项目从核心小组驱动变成社区驱动。

!!! interview "面试怎么答"
    "怎么快速读懂一个几十万行的推理框架？"——用这张地图答：先认 2024 年定型的骨架目录（调度、缓存、执行、层、模型），再按年份看新目录各解决什么（规模化 → 可靠性 → 平台化），最后找注册表（后端、投机方法、淘汰策略）顺藤摸瓜。能说出"测试比代码多、大文件靠 mixin 拆分、参数按组管理"这些工程事实，比背功能列表更能体现判断力。

## 小结

- [x] 三年三个量级：2024 年 1600 个提交 / 190 位作者 → 2025 年 6800 / 800 → 2026 年前九个月 1 万 1 千 / 1200。
- [x] 目录按出生时期分四类：2024 骨架、2025 上半年规模化、2025 下半年可靠性与可观测、2026 平台化。
- [x] 路线图从功能转向兼容与可靠性：拆大文件、扩 CUDA Graph 覆盖、弹性 EP、调试与一致性工具、参数分组。
