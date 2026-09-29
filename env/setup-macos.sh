#!/usr/bin/env bash
# 在 Mac 上一次装好全部手册的学习环境（Apple Silicon 与 Intel 都可以）。
# 用法：在仓库根目录运行
#   bash env/setup-macos.sh            # 基础环境 + Qwen2.5-0.5B、Qwen3-0.6B、Gemma 3 270M 三个小模型（约 3 GB）
#   bash env/setup-macos.sh --with-vl  # 另外下载多模态一章用的 Qwen2.5-VL-3B（约 7.5 GB）
# 装完运行自检：.venv/bin/python tools/mac_check.py
set -euo pipefail
cd "$(dirname "$0")/.."
WITH_VL=0
[[ "${1:-}" == "--with-vl" ]] && WITH_VL=1

if [[ "$(uname -s)" != "Darwin" && -z "${FORCE_SETUP:-}" ]]; then
  echo "这个脚本用于 macOS；Linux / WSL2 请参考各手册的 README 和 practice/env/setup-wsl2.sh" >&2
  exit 1
fi

# 1. Xcode 命令行工具：clang++、git（C++ 手册、CUDA 模拟器、PyTorch 扩展都要用）
if [[ "$(uname -s)" == "Darwin" ]] && ! xcode-select -p >/dev/null 2>&1; then
  echo "==> 安装 Xcode 命令行工具（弹窗确认，装完后重新运行本脚本）"
  xcode-select --install || true
  exit 1
fi

# 2. Homebrew 的 cmake、ninja、llvm：C++ 手册的工程章节要 cmake；较新的 LLVM 自带较新的 libc++（std::jthread）
if command -v brew >/dev/null 2>&1; then
  echo "==> brew install cmake ninja llvm"
  brew install cmake ninja llvm || echo "（brew 安装失败，可以稍后手动安装；只影响 C++ 手册的部分例子）"
else
  echo "提示：没有找到 Homebrew（https://brew.sh）。C++ 手册的 CMake 章节和 std::jthread 的例子需要 cmake 与较新的 LLVM。"
fi

# 3. uv：Python 与虚拟环境管理
if ! command -v uv >/dev/null 2>&1; then
  echo "==> 安装 uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# 4. 一个共用的虚拟环境 .venv（Python 3.12）：大模型原理、推理系统、分布式训练、CUDA 手册的 PyTorch 章节、mini-sglang、练习题、大作业
echo "==> 创建 .venv（Python 3.12）并安装依赖"
uv venv .venv --python 3.12 --allow-existing
uv pip install --python .venv/bin/python \
  torch torchvision numpy transformers safetensors tokenizers pillow \
  pytest pybind11 ninja setuptools \
  msgpack pyzmq fastapi uvicorn prompt_toolkit psutil httpx openai \
  modelscope huggingface_hub

# 5. Python 手册的检查环境（它的例子按 Python 3.14 验证）
echo "==> 创建 python/.venv-check（Python 3.14）"
uv venv python/.venv-check --python 3.14 --allow-existing
uv pip install --python python/.venv-check/bin/python pytest mypy

# 6. 各手册的检查脚本默认找的解释器路径：都指向 .venv
link() {  # link <路径> <目标>：路径不存在（或已经是软链接）时才创建
  if [[ -L "$1" || ! -e "$1" ]]; then ln -sfn "$2" "$1"; else echo "保留已有的 $1"; fi
}
link llm/.venv-llm ../.venv
link serving/.venv-llm ../.venv
link cpp/.venv-py ../.venv
link practice/.venv ../.venv

# 7. 模型：放在仓库根目录的 models/，三本书各自的 models 指向它（ModelScope 在国内快，失败时换 Hugging Face）
mkdir -p models
download() {  # download <仓库名> <目录名>
  if [[ -f "models/$2/config.json" ]]; then echo "已存在 models/$2"; return; fi
  echo "==> 下载 $1"
  .venv/bin/python - "$1" "models/$2" <<'PY' || .venv/bin/python -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], local_dir=sys.argv[2])" "$1" "models/$2"
import sys
from modelscope import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2])
PY
}
download Qwen/Qwen2.5-0.5B-Instruct Qwen2.5-0.5B-Instruct
download Qwen/Qwen3-0.6B Qwen3-0.6B
download google/gemma-3-270m gemma-3-270m    # 推理系统手册「新模型接入与精度对齐」一章用
[[ $WITH_VL == 1 ]] && download Qwen/Qwen2.5-VL-3B-Instruct Qwen2.5-VL-3B-Instruct
link llm/models ../models
link serving/models ../models
link minisgl/models ../models

cat <<'MSG'

完成。常用命令：
  .venv/bin/python tools/mac_check.py          # 自检：每本书跑几个代表性的例子（约 10 分钟）
  source .venv/bin/activate                     # 之后在这个环境里跑各章的代码
  python practice/judge.py doctor               # 练习题的环境检查
MSG
