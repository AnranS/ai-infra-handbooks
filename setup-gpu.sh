#!/usr/bin/env bash
# 在一台有 NVIDIA 显卡的机器（Linux 或 WSL2）上一次装好跑手册代码需要的环境。
# 用法：在仓库根目录运行
#   bash setup-gpu.sh                 # 环境 + 三个小模型（约 4 GB）
#   SKIP_MODELS=1 bash setup-gpu.sh   # 只装环境，不下模型
# 装完运行自检：.venv-gpu/bin/python gpu_check.py
#
# Mac 上用 env/setup-macos.sh（没有 CUDA，相应的检查会跳过）。
set -euo pipefail
cd "$(dirname "$0")"

echo "==> 1/6 检查显卡与驱动"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "没有找到 nvidia-smi：这台机器上没有 NVIDIA 驱动。" >&2
  echo "只想跑 CPU 的部分（正文的绝大多数例子都能跑）就用 env/setup-macos.sh，或者自己建一个装 CPU 版 torch 的虚拟环境。" >&2
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo "==> 2/6 检查 nvcc（CUDA 手册的 .cu 示例要用它编译）"
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | tail -2
else
  echo "没有找到 nvcc。CUDA 手册的示例需要 CUDA Toolkit："
  echo "  Ubuntu: sudo apt install nvidia-cuda-toolkit     （版本较旧，够用）"
  echo "  或装官方的 CUDA Toolkit：https://developer.nvidia.com/cuda-downloads"
  echo "（没有 nvcc 也能继续：PyTorch 的部分不需要它）"
fi

echo "==> 3/6 uv：Python 与虚拟环境管理"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> 4/6 创建 .venv-gpu（Python 3.12）并安装 CUDA 版 PyTorch"
uv venv .venv-gpu --python 3.12 --allow-existing
uv pip install --python .venv-gpu/bin/python \
  torch torchvision numpy transformers safetensors tokenizers pillow \
  pytest pybind11 ninja setuptools \
  msgpack pyzmq fastapi uvicorn prompt_toolkit psutil httpx openai \
  modelscope huggingface_hub triton
.venv-gpu/bin/python - <<'PY'
import torch
print(f"torch {torch.__version__}；CUDA 可用：{torch.cuda.is_available()}"
      + (f"；{torch.cuda.get_device_name(0)}，{torch.cuda.device_count()} 张卡" if torch.cuda.is_available() else ""))
PY

echo "==> 5/6 各手册的检查脚本默认找的解释器路径指向 .venv-gpu"
link() {  # link <路径> <目标>：路径不存在（或已经是软链接）时才创建
  if [[ -L "$1" || ! -e "$1" ]]; then ln -sfn "$2" "$1"; else echo "保留已有的 $1"; fi
}
link llm/.venv-llm ../.venv-gpu
link serving/.venv-llm ../.venv-gpu
link minisgl/.venv-llm ../.venv-gpu
link cpp/.venv-py ../.venv-gpu
link practice/.venv ../.venv-gpu

echo "==> 6/6 模型：放在仓库根目录的 models/"
if [[ -n "${SKIP_MODELS:-}" ]]; then
  echo "跳过（SKIP_MODELS=1）"
else
  mkdir -p models
  download() {  # download <仓库名> <目录名>
    if [[ -f "models/$2/config.json" ]]; then echo "已存在 models/$2"; return; fi
    echo "==> 下载 $1"
    .venv-gpu/bin/python -c "import sys; from modelscope import snapshot_download; snapshot_download(sys.argv[1], local_dir=sys.argv[2])" "$1" "models/$2" \
      || .venv-gpu/bin/python -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], local_dir=sys.argv[2])" "$1" "models/$2"
  }
  download Qwen/Qwen3-0.6B Qwen3-0.6B                # 大模型原理、推理系统、mini-sglang 的例子模型
  download Qwen/Qwen3.5-0.8B Qwen3.5-0.8B            # 推理系统手册的线性注意力、多模态两章，大作业四
  download google/gemma-3-270m gemma-3-270m          # 推理系统手册「新模型接入与精度对齐」一章用
fi
link llm/models ../models                            # 各书都从自己的目录找 models/
link serving/models ../models
link minisgl/models ../models

cat <<'EOF'

装好了。接下来：

  .venv-gpu/bin/python gpu_check.py          # 自检：把需要真卡的例子跑一遍（约 15 分钟）
  .venv-gpu/bin/python gpu_check.py bench    # 只跑基准测试（CUDA 手册那一节用的）

各书正文里的代码也以真实文件的形式放在 <书>/examples/ 下，可以直接读和跑。
EOF
