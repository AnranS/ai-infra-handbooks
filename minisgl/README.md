# 手写 mini-sglang

A Chinese-language MkDocs Material book that rebuilds [mini-sglang](https://github.com/sgl-project/mini-sglang) (pinned at commit `9a91cfa`) from scratch, one module per chapter, with the same package layout and interfaces as upstream plus CPU reference implementations so every step can be verified without a GPU.

## Layout

- `python/minisgl/`: the implementation (mirrors upstream `python/minisgl/`); `utils/device.py`, `attention/torch_backend.py`, `moe/torch_backend.py`, `kernel/torch_ops.py` and `engine/graph.py:EmulatedGraph` are the CPU additions
- `tests/`: one pytest file per chapter; `tests/fakes/` holds same-interface PyTorch fakes of FlashInfer and `sgl_kernel.flash_attn`; `tests/cuda/` holds a self-checking CUDA program for the custom kernels
- `examples/`: the scripts whose output appears in the book (`tools/check.py` writes it to `docs/_outputs/`)
- `hooks/include_code.py`: MkDocs hook that expands `@@code path[:Symbol]@@`, `@@output name@@` and `@@upstream path[:Symbol]@@` (links with line numbers into the pinned upstream commit)
- `docs/`: the book

## Commands

```bash
pip install -e ".[dev]"                        # torch (CPU is fine), transformers, pyzmq, fastapi, ...
python tools/check.py                          # examples -> docs/_outputs, nvcc + CPU-emulator kernel checks, pytest
PYTHONPATH=python:tests pytest -q tests        # the test suite only (about 2 minutes on CPU)
python -m minisgl --model models/Qwen3-0.6B    # run the server
```

`tools/check.py` expects `models/Qwen3-0.6B` and `models/Qwen2.5-0.5B-Instruct`, an upstream checkout at `~/src-reading/mini-sglang` (or `$UPSTREAM`) on commit `9a91cfa`, and the CUDA handbook's nvcc toolchains (`~/cuda-handbook/.toolkit*`). The generated `docs/_outputs/` is committed so the site builds without any of these.
