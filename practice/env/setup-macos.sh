#!/usr/bin/env bash
# macOS 练习环境：uv + Python 3.12 + numpy + PyTorch（Apple Silicon 上用 MPS）。
# 用法：在仓库根目录运行  bash practice/env/setup-macos.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "这个脚本用于 macOS；WSL2 / Linux 请用 practice/env/setup-wsl2.sh" >&2
  exit 1
fi
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "提示：这是 Intel Mac，没有 MPS，PyTorch 题会跑在 CPU 上。"
fi

# 1. 命令行工具（clang++）：CUDA C++ 题在 Mac 上用 CPU 模拟器编译运行
if ! xcode-select -p >/dev/null 2>&1; then
  echo "==> 安装 Xcode 命令行工具（弹窗确认后，重新运行本脚本）"
  xcode-select --install || true
  exit 1
fi

# 2. uv（Python 包管理器）
if ! command -v uv >/dev/null 2>&1; then
  echo "==> 安装 uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# 3. 虚拟环境与依赖（macOS 上 PyTorch 官方 wheel 自带 MPS 支持；没有 Triton，Triton 题自动用模拟器）
echo "==> 创建 practice/.venv（Python 3.12）"
uv venv practice/.venv --python 3.12 --allow-existing
uv pip install --python practice/.venv/bin/python numpy torch

echo "==> 环境自检"
practice/.venv/bin/python practice/judge.py doctor
cat <<'MSG'

完成。以后先激活环境：  source practice/.venv/bin/activate
然后：                  python practice/judge.py start 1   # 复制第 1 题的模板
                        python practice/judge.py test 1    # 判题
MSG
