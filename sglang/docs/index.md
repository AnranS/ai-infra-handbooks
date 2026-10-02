# SGLang 设计演进

<p class="lead">读一个大项目的源码，最难的不是看懂某个函数，而是弄清"它为什么长成这样"。这本书把 SGLang 仓库 19000 多个提交当作一手史料：从 2023 年的论文和 2024 年 1 月的初始提交开始，按时间顺序讲每一个关键设计是在什么问题下、由哪个提交引入、替换了什么、后来怎么演变。每一章都落到具体的提交号、PR 号和日期，所有命令和数字都能在你自己的克隆上复现。</p>

## 学完能做到

- 看到 SGLang 今天的任何一个目录（`mem_cache/`、`disaggregation/`、`speculative/`、`sgl-router/` ……），能说出它是什么时候、为了解决什么问题出现的，第一版长什么样；
- 用"问题 → 第一版 → 取舍 → 后来的演变"的方式讲清 RadixAttention、零开销调度、PD 分离、大规模 EP、HiCache 这些面试必问的设计，而不是背今天的代码；
- 掌握一套用 git 做源码考古的方法：按月提交量、路径首次出现、关键词首次出现、模块 × 季度热力表、`git log -S`、`--follow`、读 PR 和路线图 issue；
- 对"工业级推理引擎如何从 1 万行长到 80 多万行"有数量上的直觉：哪些阶段在扩功能、哪些阶段在还技术债、哪些决定贯穿始终。

## 学习路线

这本书是[推理系统手册](serving://)的"源码导读 · SGLang"和[手写 mini-sglang](minisgl://)的续篇：前者讲今天的代码怎么读，后者照今天的模块复刻一遍，这本讲这些模块是怎么一步步长出来的。按时间分成六个部分：

<div class="roadmap" markdown>

| 部分 | 章节 | 目标 | 建议用时 |
| --- | --- | --- | --- |
| 起点（2023-10 → 2024-02） | [论文](origins/paper.md) · [初始提交](origins/first-commit.md) · [RadixAttention 第一版](origins/radix-v1.md) · [压缩 FSM 与跳跃解码](origins/fsm-jump.md) · [前端语言](origins/frontend.md) | 读懂作者最初的问题定义和一万行代码里的全部设计 | 2 天 |
| 从研究代码到可用的服务（2024 上半年） | [进程模型重构](service/processes.md) · [服务化](service/api-multimodal.md) · [借力 vLLM](service/borrow-vllm.md) · [v0.2](service/v02.md) | 看一个研究原型怎样变成能部署的东西 | 1 天 |
| 性能工程与零开销调度（2024 下半年） | [目录大重组](perf/restructure.md) · [MLA 与 torch.compile](perf/mla-compile.md) · [重叠调度](perf/overlap.md) · [多卡](perf/multi-gpu.md) · [sgl-kernel](perf/sgl-kernel.md) | 理解"快"是怎么一层层叠出来的 | 2 天 |
| 投机解码、PD 分离与大规模 EP（2025 上半年） | [EAGLE](scale/eagle.md) · [HiCache](scale/hicache.md) · [PD 分离](scale/pd.md) · [大规模 EP](scale/large-ep.md) · [注意力后端](scale/attention-backends.md) | 理解规模化阶段的每个新目录 | 2 天 |
| 从引擎到平台（2025 下半年 → 2026） | [入口层](platform/entrypoints.md) · [Rust 网关](platform/gateway.md) · [RL 闭环](platform/rl.md) · [多模态与 Diffusion](platform/multimodal-diffusion.md) · [2026 快照](platform/codebase-2026.md) | 看懂成熟期的工程重点 | 1 天 |
| 方法与总结 | [十个设计决定](method/principles.md) · [git 工具箱](method/archaeology.md) · [面试怎么讲](method/interview.md) | 把演进讲成故事，并能自己考古 | 半天 |

</div>

先看全书的时间线：按月的提交量、版本和大事件，点一个事件就跳到对应的章节。

<div class="aig-widget" data-widget="sgl-timeline"></div>

## 这本书的做法

**一手史料，固定基准。** 所有统计都在 SGLang 官方仓库的完整克隆上计算，基准固定为 2026-10-02 的 `main` 提交 `29f6d408c0`（书里的脚本用 `$REF` 指代它）。截至这个提交，仓库有 19247 个提交、165 个 tag；页面上的每个数字都来自页面上的那条命令。换一个 `REF`，数字会变，方法不变。

**代码按提交号原样截取。** 书里引用的历史代码都标明了路径、提交和行号，比如 `python/sglang/srt/managers/router/radix_cache.py @ 22085081bb` 的第 113–140 行。核对脚本会用 `git show 提交:路径` 重新截取并逐字比对，所以引用不会因为转述而失真。读的时候建议在本地克隆里同时打开那个版本：

```bash
git clone https://github.com/sgl-project/sglang.git ~/sglang-src
cd ~/sglang-src
git show 22085081bb:python/sglang/srt/managers/router/radix_cache.py | sed -n '113,140p'
```

**官方叙述为证。** 每个阶段的"作者当时怎么想"，尽量引官方材料：论文、LMSYS 博客、路线图 issue、PR 描述和提交信息。推断的地方会写明是推断。

**每章的体例。** 章首自测 → 六格小剧场 → 正文（当时的问题、第一版代码、设计取舍）→ "后来怎么样了" → 练习（都是能用 git 回答的考古题，附参考答案）→ 面试怎么答。

!!! note "版本与命名"
    SGLang 的目录在三年里多次重组。书里按当时的路径写（比如初版的 `srt/managers/router/`），并在每章末尾给出今天的对应位置。提交号一律用前 10 位，PR 号写成 `#1234`，都可以在 GitHub 上直接打开：`https://github.com/sgl-project/sglang/commit/<提交号>`、`https://github.com/sgl-project/sglang/pull/<PR 号>`。

## 代码与运行

全部脚本在仓库的 [`sglang/`](https://github.com/AnranS/ai-infra-handbooks/tree/main/sglang) 目录：

```text
sglang/
├── docs/                正文；每章里的 ```bash title="x.sh" / ```python title="x.py" 块就是本章的考古脚本
├── tools/check_code.py  在本地克隆上重跑所有脚本、重新截取所有引用的代码，与页面逐行比对
└── build/code/          核对时写出的脚本文件
```

核对的方法：

```bash
git clone https://github.com/sgl-project/sglang.git ~/sglang-src      # 完整克隆约 400 MB（git log -S 这类命令要读每个提交，别用 --filter=blob:none）；不要工作区可加 --no-checkout
cd ai-infra-handbooks/sglang
python3 tools/check_code.py                      # 跑全部页面；SGLANG_SRC=/path 指定克隆位置，REF=... 换基准提交
python3 tools/check_code.py docs/origins/radix-v1.md   # 只核对一章
```

脚本只用 `git` 和 Python 标准库，不需要 GPU，也不需要安装 SGLang。
