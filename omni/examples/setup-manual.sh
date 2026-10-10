uv venv -p 3.12 .venv-omni
uv pip install --python .venv-omni/bin/python --index-url https://download.pytorch.org/whl/cpu \
    torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0
uv pip install --python .venv-omni/bin/python -r requirements-check.txt     # pydantic、pyzmq、transformers 等
uv pip install --python .venv-omni/bin/python --no-deps sglang==0.5.21      # 只要 Python 部分，不装 CUDA 依赖
