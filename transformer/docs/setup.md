# 项目结构与环境搭建

<p class="lead">这本书的代码就是一个目录里的十几个文件：一个 <code>qwen/</code> 包（一章一个文件）加上每章一个脚本，权重放在 <code>models/</code> 下。这一页把目录结构、环境怎么装、每一章新建和修改了哪些文件、怎么运行和核对一次说清楚；之后每一章开头都会标出那一章动了哪些文件。</p>

## 目录结构

可以直接用仓库里的 `transformer/examples/`（那里的文件是从正文逐字导出的），也可以自己建一个空目录，按章节顺序把文件敲进去。两种方式的目录长一样：

```text
transformer-by-hand/                 ← 仓库里对应 transformer/examples/
├── models/
│   └── Qwen3.5-0.8B/                权重与 tokenizer，hf download 下来的（1.7 GB，不进 git）
├── qwen/                            这本书写的包：一章一个文件，到第 16 章共 17 个
│   ├── weights.py                   （一）读 config.json 和 safetensors
│   ├── …
│   └── moe.py                       （十六）
├── main.py  map.py                  第 0 步
└── ch01_weights.py … ch16_scale.py  每章一个脚本，在这个目录下直接 python chNN_xxx.py
```

全书写完之后 `qwen/` 里是这些文件（由各章的代码块自动算出来；第 10 章改了四个文件）：

@@files all@@

仓库里和这本书有关的还有三样：`transformer/docs/` 是正文（唯一的源，`examples/` 是它的导出）；`transformer/tools/check_code.py` 是校验脚本，按导航顺序把全书接力跑一遍；`transformer/models/` 在仓库里是指向根目录 `models/` 的软链接，一键脚本会帮你建好。

## 环境

需要 Python 3.10 以上、PyTorch（CPU 版就够，2.4 以上）、`safetensors`、`tokenizers`，以及对答案用的 `transformers`（5.0 以上才有 Qwen3.5 的实现）。三步：

```bash
# 1. 虚拟环境（用 uv 或 python -m venv 都行）
python3 -m venv .venv && source .venv/bin/activate
# 2. 依赖：CPU 版 PyTorch 够用；有 NVIDIA 显卡装 CUDA 版，Mac 直接 pip install torch 就带 MPS
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install safetensors tokenizers "transformers>=5.0" huggingface_hub
# 3. 权重 1.7 GB；国内先 export HF_ENDPOINT=https://hf-mirror.com
hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B
# 验证：能 import、权重在
python -c "import torch, transformers, safetensors, tokenizers; print(torch.__version__, transformers.__version__)"
ls models/Qwen3.5-0.8B/config.json
```

| 依赖 | 哪几章用 | 做什么 |
| --- | --- | --- |
| `torch` | 全部 | 张量、矩阵乘、`nn.Module` |
| `safetensors` | 第 1 章起 | 读权重文件（只读目录、按名字取张量） |
| `tokenizers` | 第 9 章起 | 直接读 tokenizer.json，不再经过 transformers |
| `transformers` | 第 0–9、12、16 章 | 参考实现：前九章对 logits，第 12 章对聊天模板，第 16 章对 MoE 层 |
| `huggingface_hub` | 下载 | `hf download` 命令 |

**用仓库里的一键脚本**：Mac 上 `bash env/setup-macos.sh` 会装好 Python 环境、PyTorch 和编译工具链，下载包括 Qwen3.5-0.8B 在内的三个小模型并建好 `transformer/models` 的软链接；之后 `python tools/mac_check.py transformer` 把前十章接力跑一遍当自检。有 NVIDIA 显卡的机器用根目录的 `bash setup-gpu.sh`。

**内存与设备**：fp32 的模型 3 GB；前九章同时加载 HF 的参考实现，峰值约 6 GB（`load_state_dict(assign=True)` 不复制权重，否则是 9 GB）。脚本默认在 CPU 上跑，线程数设为逻辑核的一半——第 15 章量过，32 核的机器开满 32 线程反而慢 60 倍；Mac 不用管。所有代码都能 `.to("cuda")` / `.to("mps")`，第 15 章的脚本会自动选设备。

## 怎么运行、怎么核对

每一章的脚本只依赖前面各章写好的文件，所以按章节顺序在同一个目录里运行就行：

```bash
cd transformer-by-hand            # 或者仓库里的 transformer/examples/
python main.py                    # 第 0 步
python ch01_weights.py            # 第 1 章……以此类推
QWEN_MODEL=/data/models/Qwen3.5-0.8B python ch09_model.py   # 权重不在 models/ 下时用环境变量指
```

脚本都从当前目录往上找 `models/Qwen3.5-0.8B`，所以在 `examples/` 下或者它的任何子目录里跑都能找到仓库根目录的 `models/`。

仓库用户还可以用校验脚本确认页面上的输出和自己机器上的一致：

```bash
python transformer/tools/check_code.py                       # 全书接力跑一遍（约 10 分钟，CPU）
python transformer/tools/check_code.py transformer/docs/cache.md --only   # 只跑第 10 章（前面的章只把文件写进工作目录）
```

它按导航顺序把每一章的代码块写进 `transformer/build/examples/tutorial/`，运行每章的脚本，把标准输出和页面上的「输出」块逐行比对（含耗时的行只比对数字以外的部分）。

## 每一章新建和修改的文件

下表由各章的代码块自动算出来。只有第 10 章改了已有的文件（注意力、线性注意力、解码层、整模型各加一个 `state` 参数），其余各章只新建：

@@files table@@

每一章正文的开头也有同样的信息：那一章之后 `qwen/` 里有哪些文件，哪些是本章新建（`+`）、哪些是本章修改（`~`）。

!!! interview "怎么讲清楚"
    讲"一个项目怎么组织"，说清三件事：源文件在哪、生成物在哪、怎么验证两者一致。这本书的答案是：正文是唯一的源，`examples/` 是导出，`check_code.py` 把正文接力跑一遍并和页面上的输出比对——改了代码而没改输出，构建就会失败。这个"源 → 导出 → 校验"的闭环，和推理框架里"配置 → 生成的 kernel → 数值对齐测试"是同一个思路。

## 小结

- [x] 一个目录：`models/`（权重）、`qwen/`（17 个文件）、每章一个脚本。
- [x] 三步装环境：虚拟环境、PyTorch + 四个库、`hf download`；Mac 用 `env/setup-macos.sh`。
- [x] 按章顺序运行；`check_code.py` 接力核对；每章开头标出新建和修改的文件。
