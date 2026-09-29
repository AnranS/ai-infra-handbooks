#!/usr/bin/env bash
# WSL2（或原生 Linux）+ NVIDIA GPU 练习环境：检查驱动 → CUDA Toolkit（nvcc）→ uv + PyTorch（CUDA）+ Triton。
# 用法：在仓库根目录运行  bash practice/env/setup-wsl2.sh
#   INSTALL_CUDA=1 bash practice/env/setup-wsl2.sh    # 没有 nvcc 时自动安装 CUDA Toolkit（需要 sudo）
#   TORCH_CUDA=cu126 bash practice/env/setup-wsl2.sh  # 指定 PyTorch 的 CUDA 版本（默认按驱动自动选）
set -euo pipefail
cd "$(dirname "$0")/../.."

if grep -qi microsoft /proc/version 2>/dev/null; then
  echo "==> 检测到 WSL2"
else
  echo "==> 不是 WSL2：按原生 Linux + NVIDIA GPU 处理"
fi

# 1. 驱动：WSL2 用的是 Windows 上的 NVIDIA 驱动，Linux 里不要再装驱动
if ! command -v nvidia-smi >/dev/null 2>&1; then
  cat >&2 <<'MSG'
找不到 nvidia-smi。
- WSL2：在 Windows 上安装最新的 NVIDIA 显卡驱动（Game Ready / Studio 均可），然后在 PowerShell 里执行 wsl --update，
  重启 WSL（wsl --shutdown）。不要在 WSL 里安装 Linux 版驱动。
- 原生 Linux：先安装 NVIDIA 驱动。
MSG
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
DRIVER_CUDA=$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | grep -o '[0-9.]*$' || echo 0)
echo "驱动支持的最高 CUDA 版本：$DRIVER_CUDA"

# 2. 编译器与 CUDA Toolkit（nvcc）
if ! command -v g++ >/dev/null 2>&1; then
  echo "==> 安装 g++（nvcc 的主机编译器）"
  sudo apt-get update && sudo apt-get install -y build-essential
fi
NVCC=$(command -v nvcc || true)
[[ -z "$NVCC" && -x /usr/local/cuda/bin/nvcc ]] && NVCC=/usr/local/cuda/bin/nvcc
if [[ -z "$NVCC" ]]; then
  CMDS='wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb
sudo apt-get update && sudo apt-get install -y cuda-toolkit-12-8'
  if grep -qi microsoft /proc/version 2>/dev/null; then :; else
    CMDS=${CMDS//wsl-ubuntu/ubuntu2204}
  fi
  if [[ "${INSTALL_CUDA:-0}" == 1 ]]; then
    echo "==> 安装 CUDA Toolkit 12.8（WSL-Ubuntu 源，只含工具链、不含驱动）"
    bash -c "$CMDS"
    NVCC=/usr/local/cuda/bin/nvcc
  else
    echo "没有找到 nvcc。CUDA C++ 题需要它，安装命令（或用 INSTALL_CUDA=1 重新运行本脚本）："
    echo "$CMDS"
  fi
fi
if [[ -n "$NVCC" ]]; then
  "$NVCC" --version | tail -1
  case ":$PATH:" in *":$(dirname "$NVCC"):"*) ;; *)
    echo "提示：把 nvcc 加进 PATH：echo 'export PATH=$(dirname "$NVCC"):\$PATH' >> ~/.bashrc" ;;
  esac
fi

# 3. uv
if ! command -v uv >/dev/null 2>&1; then
  echo "==> 安装 uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# 4. PyTorch（CUDA 版，Linux 上会一起装好 Triton）
if [[ -z "${TORCH_CUDA:-}" ]]; then
  if awk "BEGIN{exit !($DRIVER_CUDA >= 12.8)}"; then TORCH_CUDA=cu128; else TORCH_CUDA=cu126; fi
fi
echo "==> 创建 practice/.venv（Python 3.12），PyTorch 使用 $TORCH_CUDA"
uv venv practice/.venv --python 3.12 --allow-existing
uv pip install --python practice/.venv/bin/python numpy
uv pip install --python practice/.venv/bin/python torch --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"

echo "==> 环境自检"
PATH="$(dirname "${NVCC:-/usr/local/cuda/bin/nvcc}"):$PATH" practice/.venv/bin/python practice/judge.py doctor
cat <<'MSG'

完成。以后先激活环境：  source practice/.venv/bin/activate
然后：                  python practice/judge.py start 1   # 复制第 1 题的模板
                        python practice/judge.py test 1    # 判题
MSG
