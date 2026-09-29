# 手写 mini-sglang

A Chinese-language MkDocs Material book that rebuilds [mini-sglang](https://github.com/sgl-project/mini-sglang) (pinned at commit `9a91cfa`) from scratch, one module per chapter, with the same package layout and interfaces as upstream plus CPU reference implementations so every step can be verified without a GPU.

## Layout

- `python/minisgl/`: the implementation (mirrors upstream `python/minisgl/`); `utils/device.py`, `attention/torch_backend.py`, `moe/torch_backend.py`, `kernel/torch_ops.py` and `engine/graph.py:EmulatedGraph` are the CPU additions
- `tests/`: one pytest file per chapter; `tests/fakes/` holds same-interface PyTorch fakes of FlashInfer and `sgl_kernel.flash_attn`; `tests/cuda/` holds a self-checking CUDA program for the custom kernels
- `examples/`: the scripts whose output appears in the book (`tools/check.py` writes it to `docs/_outputs/`)
- `hooks/include_code.py`: MkDocs hook that expands `@@code path[:Symbol]@@`, `@@output name@@`, `@@upstream path[:Symbol]@@` (links with line numbers into the pinned upstream commit), `@@diagram name caption@@` (inline SVG from `docs/assets/diagrams/`) and `@@video name caption@@` (`docs/assets/videos/<name>.mp4` + `.jpg` poster)
- `tools/diagrams.py`: generates the 15 theme-aware SVG architecture diagrams
- `tools/videos/`: the 6 narrated teaching videos — one canvas scene per video (`<name>.js`, narration text in `SEGMENTS`) on a shared engine (`engine.js`); `render.py` synthesizes the narration with edge-tts, times each scene segment to its audio, renders frames with headless Chromium and encodes with ffmpeg
- `docs/`: the book

## Commands

```bash
pip install -e ".[dev]"                        # torch (CPU is fine), transformers, pyzmq, fastapi, ...
python tools/check.py                          # examples -> docs/_outputs, nvcc + CPU-emulator kernel checks, pytest
PYTHONPATH=python:tests pytest -q tests        # the test suite only (about 2 minutes on CPU)
python -m minisgl --model models/Qwen3-0.6B    # run the server
```

```bash
PYTHONPATH=python python tools/diagrams.py     # regenerate docs/assets/diagrams/*.svg
cd tools/videos && python render.py lifecycle batching radix overlap tp cudagraph --workers 8
                                               # needs edge-tts, playwright (+ chromium), imageio-ffmpeg and network access for TTS
python render.py radix --preview 30 60         # stills at given seconds -> build/radix/preview/
```

`tools/check.py` expects `models/Qwen3-0.6B` (the Qwen2 config test reads `tests/configs/`), an upstream checkout at `~/src-reading/mini-sglang` (or `$UPSTREAM`) on commit `9a91cfa`, and the CUDA handbook's nvcc toolchains (`~/cuda-handbook/.toolkit*`). The generated `docs/_outputs/` is committed so the site builds without any of these.
