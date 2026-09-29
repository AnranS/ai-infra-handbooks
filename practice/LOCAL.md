# 本地环境：macOS 与 WSL2 + NVIDIA GPU

题库里的题目按运行环境分三类，题目页和列表上都有标注：

| 标注 | 能在哪里判题 | 说明 |
| --- | --- | --- |
| （无标注） | 浏览器、macOS、WSL2 | Python / numpy 题，以及用 **gpusim**（GPU 模拟器）和 **Triton 模拟器**写的 CUDA、Triton 题 |
| 需要本地 Python | macOS、WSL2 | 用到浏览器里没有的功能，例如线程 |
| 需要 PyTorch | macOS、WSL2 | macOS（Apple Silicon）上自动用 **MPS**，WSL2 上用 **CUDA**，都没有时退回 CPU |
| 需要 NVIDIA GPU | WSL2 | 在真卡上运行和计时 |

另外，部分 CUDA 题带 **CUDA C++ 版本**：在 WSL2 上用 `nvcc` 编译到真卡上运行，报告耗时和带宽；在 macOS 或没有 GPU 的 Linux 上，同一条命令会改用仓库里的 CPU 模拟器编译运行，只检查正确性。

## 一次性安装

先克隆仓库：

```bash
git clone https://github.com/AnranS/ai-infra-handbooks.git
cd ai-infra-handbooks
```

### macOS（Apple Silicon）

```bash
bash practice/env/setup-macos.sh
```

脚本会做这几件事：

1. 检查 Xcode 命令行工具（`clang++`，CUDA C++ 题的 CPU 模拟器要用），没有就提示安装；
2. 安装 [uv](https://docs.astral.sh/uv/)，在 `practice/.venv` 建一个 Python 3.12 虚拟环境；
3. 安装 numpy 和 PyTorch。macOS 上的官方 PyTorch 自带 MPS 后端；macOS 没有 Triton，Triton 题会自动改用模拟器 minitl，写法完全一样；
4. 运行 `judge.py doctor` 自检。

MPS 的注意事项：MPS 不支持 float64，涉及 PyTorch 的题一律用 float32；个别算子 MPS 还没实现时，设置环境变量 `PYTORCH_ENABLE_MPS_FALLBACK=1` 让它回退到 CPU。

### WSL2 + NVIDIA GPU

1. 在 **Windows** 上安装最新的 NVIDIA 显卡驱动，并在 PowerShell 里运行 `wsl --update`。WSL2 直接使用 Windows 的驱动，**不要在 WSL 里安装 Linux 版驱动**。
2. 在 WSL 里（推荐 Ubuntu 22.04 / 24.04）运行：

```bash
INSTALL_CUDA=1 bash practice/env/setup-wsl2.sh
```

脚本会依次：

1. 用 `nvidia-smi` 检查 GPU 和驱动；
2. 没有 `g++` 时安装 build-essential；
3. `INSTALL_CUDA=1` 且没有 `nvcc` 时，从 NVIDIA 的 WSL-Ubuntu 源安装 CUDA Toolkit 12.8。这个源只有工具链、不含驱动；不设 `INSTALL_CUDA` 时只打印安装命令；
4. 建 `practice/.venv`，按驱动版本安装对应 CUDA 版本的 PyTorch（Linux 上会一起装好 Triton）；
5. 运行 `judge.py doctor` 自检。

装好后把 nvcc 加进 PATH：`echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc`。

原生 Linux + NVIDIA GPU 也可以用这个脚本，前提是已经装好驱动。

## 做题

```bash
source practice/.venv/bin/activate
python practice/judge.py doctor          # 看本机能跑哪些题
python practice/judge.py list            # 题目列表（--book cuda 只看某一本）
python practice/judge.py start 12        # 把第 12 题的模板复制到 practice/workspace/，并打印题目
python practice/judge.py test 12         # 判题：跑 practice/workspace/ 里你的代码
python practice/judge.py test 12 --run   # 只跑样例
python practice/judge.py solution 12     # 看参考解答
```

`practice/workspace/` 不进 git，放心改。也可以在网页上写，点「下载」保存成文件，再用 `python practice/judge.py test 12 下载的文件.py` 判题。

CUDA C++ 版本：

```bash
python practice/judge.py start 12 --cuda   # 复制 .cu 模板
python practice/judge.py test 12           # 有 GPU：nvcc 编译 + 真卡运行 + 带宽；没有 GPU：CPU 模拟器
```

Triton 题默认在有 NVIDIA GPU 时用真 Triton，否则用模拟器。设置 `PRACTICE_TRITON=emulate` 可以强制用模拟器，这样能看到模拟器报出的越界和 load/store 统计。

## 常见问题

- **WSL2 里 `torch.cuda.is_available()` 是 False**：先确认 `nvidia-smi` 能看到显卡；再确认装的是 CUDA 版 PyTorch（`python -c "import torch; print(torch.version.cuda)"` 不应该是 None）。
- **nvcc 报 `unsupported GNU version`**：CUDA 12.8 支持到 GCC 14，太新的 GCC 需要装旧版本，并加 `-ccbin g++-13`（可以通过环境变量 `PRACTICE_NVCC` 指向一个包装脚本）。
- **`-arch=native` 不被支持**：说明 nvcc 太旧（低于 11.5），请升级 CUDA Toolkit。
- **macOS 上 CUDA C++ 模拟器编译失败**：模拟器用 `ucontext` 实现线程切换，在 macOS 上已标记为弃用，个别系统版本可能编译不过。这不影响网页和 Python 题；CUDA C++ 题可以只在 WSL2 上做。
- **浏览器里第一次运行很慢**：需要下载约 10 MB 的 Python 运行环境（Pyodide）和 numpy，之后会走浏览器缓存。
