# 手写 mini-sglang

A bilingual MkDocs Material book (Chinese source, English edition under `docs-en/`) that rebuilds [mini-sglang](https://github.com/sgl-project/mini-sglang) (pinned at commit `9a91cfa`) from scratch. It starts from `examples/ch00_tiny_engine.py`, a 126-line engine that imports nothing from the package, and then replaces one naive part per chapter in four stages, keeping the same package layout and interfaces as upstream plus CPU reference implementations so every step runs and can be verified without a GPU.

## Layout

- `python/minisgl/`: the implementation (mirrors upstream `python/minisgl/`); `utils/device.py`, `attention/torch_backend.py`, `moe/torch_backend.py`, `kernel/torch_ops.py` and `engine/graph.py:EmulatedGraph` are the CPU additions
- `tests/`: one pytest file per chapter; `tests/fakes/` holds same-interface PyTorch fakes of FlashInfer and `sgl_kernel.flash_attn`; `tests/cuda/` holds a self-checking CUDA program for the custom kernels
- `examples/`: the scripts whose output appears in the book (`tools/check.py` writes it to `docs/_outputs/`)
- `steps.json`: which files each chapter creates, modifies or brings in early (`tools/steps.py derive` computes it by actually running each chapter's main against only the files so far; `check` re-verifies)
- `hooks/include_code.py`: MkDocs hook that expands `@@tree@@` (the chapter's file tree from `steps.json`), `@@code path[:Symbol]@@`, `@@output name@@`, `@@upstream path[:Symbol]@@` (links with line numbers into the pinned upstream commit), `@@diagram name caption@@` (inline SVG from `docs/assets/diagrams/`) and `@@video name caption@@` (`docs/assets/videos/<name>.mp4` + `.jpg` poster)
- `tools/diagrams.py`: generates the 15 theme-aware SVG architecture diagrams; `--lang en` writes the English versions to `docs-en/assets/diagrams/`, translating every label through the shared `../i18n/en/figures.json`
- `i18n-en-code.json`: the English text of every comment and docstring in the included source. The hook applies it only to the English build, so the code, the string literals and the line structure stay identical; `tools/code_i18n.py` lists what is still missing, and a missing entry fails `mkdocs build --strict`
- `i18n-en-strings.json`: the English text of the labels the example scripts print. `tools/outputs_en.py` applies it, runs the examples and writes `docs/_outputs_en/`, so the English pages show output produced by the very code they display. Chinese kept as data (the tokenizer demo's sample text) is deliberately absent from the table
- `tools/videos/`: the 6 narrated teaching videos — one canvas scene per video (`<name>.js`, narration text in `SEGMENTS`) on a shared engine (`engine.js`); `render.py` synthesizes the narration with edge-tts, times each scene segment to its audio, renders frames with headless Chromium and encodes with ffmpeg
- `docs/`: the book

## Commands

```bash
pip install -e ".[dev]"                        # torch (CPU is fine), transformers, pyzmq, fastapi, ...
python tools/check.py                          # examples -> docs/_outputs, nvcc + CPU-emulator kernel checks, pytest
python tools/outputs_en.py                     # the same examples with English labels -> docs/_outputs_en
PYTHONPATH=python:tests pytest -q tests        # the test suite only (about 2 minutes on CPU)
python tools/steps.py check                    # rebuild the package chapter by chapter from steps.json and run each chapter's main
python -m minisgl --model models/Qwen3-0.6B    # run the server
```

```bash
PYTHONPATH=python python tools/diagrams.py     # regenerate docs/assets/diagrams/*.svg
cd tools/videos && python render.py lifecycle batching radix overlap tp cudagraph --workers 8
                                               # needs edge-tts, playwright (+ chromium), imageio-ffmpeg and network access for TTS
python render.py radix --preview 30 60         # stills at given seconds -> build/radix/preview/
```

`tools/check.py` expects `models/Qwen3-0.6B` (the Qwen2 config test reads `tests/configs/`), an upstream checkout at `~/src-reading/mini-sglang` (or `$UPSTREAM`) on commit `9a91cfa`, and the CUDA handbook's nvcc toolchains (`~/cuda-handbook/.toolkit*`). The generated `docs/_outputs/` and `docs/_outputs_en/` are committed so the site builds without any of these. Change an example and both have to be regenerated.
