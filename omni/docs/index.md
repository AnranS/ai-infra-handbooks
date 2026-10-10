# SGLang-Omni 源码导读

<p class="lead">SGLang-Omni 是 SGLang 社区为"能听会说"的模型写的推理服务框架：Qwen3-Omni 这样的全模态模型、各种 TTS 和 ASR 模型，都被拆成多个 stage 组成的流水线，自回归的那几个 stage 把 SGLang 当成库嵌进去。这本书把它从里到外读一遍——请求怎么流过 Coordinator 和 Stage、调度器怎么嫁接 SGLang、stage 之间怎么搬数据、配置和部署怎么规划——再精读三个案例，最后落到"怎么自己找问题、提 MR"。书里的实验大多直接用 omni 真实的运行时在 CPU 上跑，不需要 GPU。</p>

## 学完能做到

- 讲清 SGLang-Omni 和 SGLang 的边界：哪些是 omni 自己的（拓扑、stage、通信、API），哪些是借 SGLang 的（调度、KV、CUDA Graph），借的方式分别是组合、注入还是继承；
- 拿到任何一个模型目录，能画出它的 stage 图、说出每个 stage 用什么调度器、数据在每条边上怎么传、首包延迟由哪几段组成；
- 用 omni 的真实运行时搭玩具流水线做实验：扇出扇入、流式、abort、攒批窗口、传输方式的选择；
- 在没有 GPU 的机器上跑通 omni 的大部分单测、构建并测试 Rust router；
- 按"找切入点 → 确认 → 定范围 → 认领 → 测试 → 描述"的流程，给这个项目提一个有分量的 MR。

## 学习路线

<div class="roadmap" markdown>

| 部分 | 章节 | 目标 | 建议用时 |
| --- | --- | --- | --- |
| 全景 | [定位](overview/position.md) · [仓库地图](overview/repo-map.md) | 划清和 SGLang 的边界，量一遍仓库 | 半天 |
| 一条请求的旅程 | [Coordinator](journey/coordinator.md) · [Stage 与调度器](journey/stage.md) · [嵌入 SGLang](journey/embed-sglang.md) · [进程间通信](journey/comm.md) · [配置与部署](journey/config.md) | 读懂运行时的每一层，并亲手在 CPU 上验证 | 3 天 |
| 案例精读 | [Qwen3-TTS](cases/qwen3-tts.md) · [Qwen3-Omni](cases/qwen3-omni.md) · [Rust router](cases/router.md) | 把机制落到真实模型和真实的前门上 | 2 天 |
| 贡献实战 | [测试、CI 与代码规范](contrib/testing.md) · [从读代码到提 MR](contrib/first-mr.md) | 知道一个 PR 要过哪些关，走完一个完整案例 | 1～2 天 |

</div>

建议按顺序读：第二部分的每一章都依赖前一章建立的概念（第三章的 Coordinator → 第四章的 Stage → 第五章的自回归 stage → 第六章 stage 之间的通信 → 第七章把它们部署起来）。读完第二部分再读案例，案例里会反复回指前面的机制。

这本书是 [SGLang 设计演进](sglang://) 的姊妹篇：那本讲 SGLang 引擎本身是怎么一步步长出来的，这本讲一个**以 SGLang 为库**的上层项目怎么组织。第五章（嵌入 SGLang）里对照的 SGLang 调度器，在那本书的"重叠调度""进程模型"几章有来历。

## 三种环境能做什么

| 环境 | 能做的 | 本书怎么用 |
| --- | --- | --- |
| Linux 开发机（只有 CPU） | 全书所有实验：玩具流水线、配置解析、部署规划、大部分单测、构建和测试 Rust router | 本书的输出都是在这样一台机器（Xeon 8336C）上跑出来的 |
| macOS | 读代码、Rust router 的开发和测试；omni 的 CPU 环境理论上可以装，但**没有验证过** | 第十章、第十二章的 router 工作最适合在这里做 |
| 一张 NVIDIA 消费级卡（例如 16 GB） | 跑真实的小模型：Qwen3-TTS 0.6B / 1.7B、Qwen3-ASR 0.6B / 1.7B；量首包延迟、调显存参数 | 第八章最后一节；Qwen3-Omni（权重约 70 GB）跑不了 |

## 这本书的做法

**固定基准。** 全书基于 sglang-omni 2026-10-10 的 `main` 提交 `921ea2c8`（书里的脚本用 `$REF` 指代它）。截至这个提交仓库有 1155 个提交；页面上的每个数字都来自页面上的那条命令。引用 SGLang 源码的地方基于 SGLang 的 `v0.5.21`——omni 在这个提交上锁定的版本。

**代码按提交号原样截取。** 引用的源码都标明了路径、提交和行号，比如 `sglang_omni/pipeline/stage/input.py @ 921ea2c8` 的第 44–117 行。核对脚本会用 `git show 提交:路径` 重新截取并逐字比对。

**实验跑在真实的运行时上。** omni 的骨架（Coordinator、Stage、ZMQ 控制面、SHM 数据面、配置和规划）不依赖 GPU，也不依赖 SGLang 的 CUDA 内核。本书的环境只装 CPU 版 PyTorch 和 SGLang 的 Python 部分，就能起多进程的真实流水线——stage 的计算换成几行 Python，其余都是 omni 自己的代码。实验里的输出都是实跑的，涉及时间的结论只给"是否超过阈值"这类稳定的判断，不给具体耗时。

**每章的体例。** 章首自测 → 正文（源码 + 实验）→ 练习（附参考思路）→ 怎么讲清楚 → 小结。

## 环境

一键准备（克隆两个仓库、建 Python 环境、装 Rust 工具链）：

```bash title="setup.sh" run="no"
git clone https://github.com/AnranS/ai-infra-handbooks.git
cd ai-infra-handbooks/omni
bash tools/setup_env.sh          # 默认克隆到 ~/sglang-omni-src 和 ~/sglang-src；OMNI_SRC / SGLANG_SRC 可以改
```

它做的事情，手工来是这样：

```bash title="setup-manual.sh" run="no"
uv venv -p 3.12 .venv-omni
uv pip install --python .venv-omni/bin/python --index-url https://download.pytorch.org/whl/cpu \
    torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0
uv pip install --python .venv-omni/bin/python -r requirements-check.txt     # pydantic、pyzmq、transformers 等
uv pip install --python .venv-omni/bin/python --no-deps sglang==0.5.21      # 只要 Python 部分，不装 CUDA 依赖
```

用这个环境运行 omni 的代码时，把基准提交的源码树放进 `PYTHONPATH`（核对脚本会自动导出一份到 `build/tree-921ea2c8/`）。日志里会有几行 `Failed to import mooncake / nixl` 的提示，是因为没有装这两个跨机传输库，不影响本书的实验。

## 代码与校验

```text
omni/
├── docs/                  正文；每章里的 ```bash title="x.sh" / ```python title="x.py" 块就是本章的实验
├── tools/check_code.py    重跑所有实验、重新截取所有引用的源码，与页面逐行比对
├── tools/setup_env.sh     准备环境
├── requirements-check.txt 本书 Python 环境的依赖（锁定版本）
└── build/                 核对时导出的源码树、写出的脚本、cargo 的构建目录
```

```bash title="check.sh" run="no"
python3 tools/check_code.py                          # 全部页面，约 10 分钟（第一次会构建 router）
python3 tools/check_code.py docs/journey/stage.md    # 只核对一章
REF=main python3 tools/check_code.py                 # 换到最新的 main 上看哪些地方变了
```

最后一种用法很有用：项目每个月有两百多个提交，换一个 `REF` 重跑，失败的地方就是这本书和最新代码之间的差异——也常常是值得去读的变化。
