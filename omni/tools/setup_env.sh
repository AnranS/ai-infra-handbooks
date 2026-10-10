#!/usr/bin/env bash
# 准备《SGLang-Omni 源码导读》的环境：不需要 GPU。
#   1. 完整克隆 sglang-omni（书里的 git 命令和引用都基于它）；引用 SGLang 源码的几处还需要 SGLang 的克隆
#   2. omni/.venv-omni：CPU 版 PyTorch + 只装 Python 部分的 SGLang（--no-deps，不装 CUDA 内核）+ 跑实验需要的依赖
#   3. Rust 工具链（第十章构建、测试 router 用；版本由仓库里的 rust-toolchain.toml 决定）
# 在 Linux x86-64 上验证过；macOS 上没有验证。
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
OMNI_SRC="${OMNI_SRC:-$HOME/sglang-omni-src}"
SGLANG_SRC="${SGLANG_SRC:-$HOME/sglang-src}"
VENV="$HERE/.venv-omni"

[ -d "$OMNI_SRC/.git" ] || git clone https://github.com/sgl-project/sglang-omni.git "$OMNI_SRC"
[ -d "$SGLANG_SRC/.git" ] || git clone https://github.com/sgl-project/sglang.git "$SGLANG_SRC"
git -C "$OMNI_SRC" cat-file -e '921ea2c8^{commit}' 2>/dev/null || git -C "$OMNI_SRC" fetch origin
git -C "$SGLANG_SRC" cat-file -e 'v0.5.21^{commit}' 2>/dev/null || git -C "$SGLANG_SRC" fetch --tags origin

command -v uv >/dev/null || { echo "需要 uv：https://docs.astral.sh/uv/"; exit 1; }
if [ ! -x "$VENV/bin/python" ]; then
  uv venv -q -p 3.12 "$VENV"
fi
PY="$VENV/bin/python"
uv pip install -q --python "$PY" --index-url https://download.pytorch.org/whl/cpu \
  "torch==2.13.0" "torchvision==0.28.0" "torchaudio==2.11.0"
uv pip install -q --python "$PY" -r "$HERE/requirements-check.txt"
uv pip install -q --python "$PY" --no-deps "sglang==0.5.21"

if ! command -v cargo >/dev/null && [ ! -x "$HOME/.cargo/bin/cargo" ]; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none --no-modify-path
fi

echo "完成。核对全书：python3 $HERE/tools/check_code.py（约 10 分钟，第一次会构建 router）"
