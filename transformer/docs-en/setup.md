# Project layout and setup

<p class="lead">The code of this book is a dozen or so files in one directory: a <code>qwen/</code> package (one file per chapter) plus one script per chapter, with the weights under <code>models/</code>. This page covers the directory layout, how to install the environment, which files each chapter creates and changes, and how to run and verify everything in one place; after this, every chapter starts by marking the files it touches.</p>

## Directory layout {#目录结构}

You can use `transformer/` from the repository directly (the files under `examples/` are exported verbatim from the text), or start from an empty directory, copy `pyproject.toml` into it, and type the files in chapter by chapter:

<!-- i18n:diagram 0883f50c9b -->
```text
transformer/                         ← this book in the repository; use the same layout if you start from scratch
├── pyproject.toml  uv.lock          the uv project: the dependency list and the locked versions, installed by one uv sync
├── .venv/                           the virtual environment uv sync creates (not in git)
├── models/
│   └── Qwen3.5-0.8B/                the weights and tokenizer from hf download (1.7 GB, not in git)
├── examples/                        the code from the text (exported verbatim from the pages):
│   ├── qwen/                        the package this book writes: one file per chapter, 17 by chapter 16
│   │   ├── weights.py               1. reads config.json and the safetensors file
│   │   ├── …
│   │   └── moe.py                   16.
│   ├── main.py  map.py              step 0
│   └── ch01_weights.py … ch16_scale.py   one script per chapter: uv run python examples/chNN_xxx.py
├── docs/  docs-en/                  the text (the single source)
└── tools/check_code.py              the checker: runs the whole book as a relay in navigation order
```

Once the book is finished, `qwen/` holds these files (computed automatically from the chapters' code blocks; chapter 10 changes four files):

@@files all@@

The text (`docs/`) is the single source and `examples/` is its export; the build checks that the two agree. In the repository `models/` can be a symlink to the root `models/` (the Mac setup script creates it), or you can download the weights straight into it.

## Environment {#环境}

The project is managed with [uv](https://docs.astral.sh/uv/): the dependencies are listed in `pyproject.toml`, and one `uv sync` creates the virtual environment and installs the pinned versions (on Linux, torch comes from PyTorch's CPU index automatically; on macOS the PyPI wheel includes MPS). After that, every script runs through `uv run`, with no environment to activate by hand.

```bash
# 1. install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. in the repository's transformer/ directory, create the environment and install the dependencies (starting from an empty directory: copy pyproject.toml there and uv sync)
cd transformer
uv sync
# 3. the weights, 1.7 GB; in China, export HF_ENDPOINT=https://hf-mirror.com first
uv run hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B
# verify: the imports work and the weights are there
uv run python -c "import torch, transformers, safetensors, tokenizers; print(torch.__version__, transformers.__version__)"
ls models/Qwen3.5-0.8B/config.json
```

The full `pyproject.toml` follows. With an NVIDIA GPU, point the index in `[tool.uv.sources]` at a CUDA build (for example `https://download.pytorch.org/whl/cu128`) and `uv sync` again:

```toml
[project]
name = "transformer-by-hand"
version = "0.1.0"
description = "《手写 Transformer》的代码：从 main.py 开始、一章一个文件写出 Qwen3.5 的混合架构"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.4",               # the CPU build is enough: on Linux it comes from the pytorch-cpu index below, on macOS the PyPI wheel includes MPS
    "safetensors>=0.4",         # reads the weight file
    "tokenizers>=0.20",         # from chapter 2 on, reads tokenizer.json directly
    "transformers>=5.0",        # used only in chapters 0, 9 and 16: the black box at the finish line, the whole model's logits, the MoE layer (Qwen3.5 arrived in 5.0)
    "huggingface_hub>=0.30",    # the hf download command
]

[tool.uv]
package = false                 # just a directory of scripts: no packaging, no installing itself

[tool.uv.sources]
torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]   # with an NVIDIA GPU, switch to a CUDA index (see the setup page)

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

| Dependency | Used in | For |
| --- | --- | --- |
| `torch` | everywhere | tensors, matrix multiplication, `nn.Module` |
| `safetensors` | from chapter 1 | reading the weight file (index only, or tensors by name) |
| `tokenizers` | from chapter 2 | reading tokenizer.json directly, without transformers |
| `transformers` | chapters 0, 9, 16 | step 0 runs the finish line as a black box, chapter 9 checks the whole model's logits, chapter 16 checks the MoE layer |
| `huggingface_hub` | the download | the `hf download` command |

**The repository's one-click scripts**: on a Mac, `bash env/setup-macos.sh` installs the Python environment shared by all the handbooks, PyTorch and the compiler toolchain, downloads three small models including Qwen3.5-0.8B, and creates the `transformer/models` symlink (this book can use that shared environment too: `PYTHON=.venv/bin/python python transformer/tools/check_code.py`); afterwards `python tools/mac_check.py transformer` runs the first ten chapters as a relay for a self-check. On a machine with an NVIDIA GPU use `bash setup-gpu.sh` in the repository root.

**Memory and devices**: the fp32 model is 3 GB; step 0 and chapter 9 also load the HF reference, peaking at about 6 GB (`load_state_dict(assign=True)` does not copy the weights, otherwise it would be 9 GB). The scripts run on the CPU by default with the thread count set to half the logical cores — chapter 15 measured that a 32-core machine with all 32 threads is 60× slower; on a Mac there is nothing to tune. All the code can be moved with `.to("cuda")` / `.to("mps")`, and chapter 15's script picks the device automatically.

## Running and verifying {#怎么运行怎么核对}

Every chapter's script depends only on the files written in the earlier chapters, so just run them in chapter order in the same directory:

```bash
cd transformer                    # in this book's directory, run every chapter's script with uv run
uv run python examples/main.py    # step 0
uv run python examples/ch01_weights.py          # chapter 1, and so on
QWEN_MODEL=/data/models/Qwen3.5-0.8B uv run python examples/ch09_model.py   # point to the weights with the environment variable when they are not under models/
```

The scripts look for `models/Qwen3.5-0.8B` upwards from the current directory, so running from `transformer/` or any of its subdirectories finds it.

The checker uses the same environment and confirms that the outputs on the pages match your own machine:

```bash
uv run python tools/check_code.py                       # run the whole book as a relay (about 10 minutes on a CPU)
uv run python tools/check_code.py docs/cache.md --only  # run only chapter 10 (the earlier chapters only write their files into the work directory)
```

It writes every chapter's code blocks into `build/examples/tutorial/` in navigation order, runs each chapter's script, and compares the standard output line by line with the "output" block on the page (for lines with timings, everything but the numbers); the interpreter defaults to `transformer/.venv` (created by uv sync) and can be overridden with the `PYTHON` environment variable.

## The files each chapter creates and changes {#每一章新建和修改的文件}

The table below is computed from the chapters' code blocks. Only chapter 10 changes existing files (attention, linear attention, the decoder layer and the whole model each gain a `state` argument); every other chapter only creates:

@@files table@@

The same information opens every chapter: which files `qwen/` holds after that chapter, which are new in it (`+`) and which it changes (`~`).

!!! interview "How to explain it"
    To explain "how a project is organized", cover three things: where the sources are, where the generated artifacts are, and how the two are kept consistent. This book's answer: the text is the single source, `examples/` is an export, and `check_code.py` runs the text as a relay and compares with the outputs on the page — change the code without the output and the build fails. This "source → export → verification" loop is the same idea as "config → generated kernel → numerical alignment test" in an inference framework.

## Summary {#小结}

- [x] One directory: `pyproject.toml` (a uv project), `models/` (the weights), `examples/qwen/` (17 files), one script per chapter.
- [x] Three steps to install: install uv, `uv sync`, `uv run hf download`; everything runs through `uv run` from then on. On a Mac the shared environment from `env/setup-macos.sh` works too.
- [x] Run in chapter order; `check_code.py` verifies as a relay; every chapter opens by marking the files it creates and changes.
