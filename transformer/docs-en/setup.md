# Project layout and setup

<p class="lead">The code of this book is a dozen or so files in one directory: a <code>qwen/</code> package (one file per chapter) plus one script per chapter, with the weights under <code>models/</code>. This page covers the directory layout, how to install the environment, which files each chapter creates and changes, and how to run and verify everything in one place; after this, every chapter starts by marking the files it touches.</p>

## Directory layout {#目录结构}

You can use `transformer/examples/` from the repository directly (its files are exported verbatim from the text), or start from an empty directory and type the files in chapter by chapter. Either way the directory looks the same:

<!-- i18n:diagram 8bfe4bca13 -->
```text
transformer-by-hand/                 ← transformer/examples/ in the repository
├── models/
│   └── Qwen3.5-0.8B/                the weights and tokenizer from hf download (1.7 GB, not in git)
├── qwen/                            the package this book writes: one file per chapter, 17 by chapter 16
│   ├── weights.py                   1. reads config.json and the safetensors file
│   ├── …
│   └── moe.py                       16.
├── main.py  map.py                  step 0
└── ch01_weights.py … ch16_scale.py  one script per chapter; run python chNN_xxx.py in this directory
```

Once the book is finished, `qwen/` holds these files (computed automatically from the chapters' code blocks; chapter 10 changes four files):

@@files all@@

Three more things in the repository belong to this book: `transformer/docs/` is the text (the single source; `examples/` is its export); `transformer/tools/check_code.py` is the checker, which runs the whole book as a relay in navigation order; and `transformer/models/` is a symlink to the repository's root `models/`, which the one-click setup script creates for you.

## Environment {#环境}

You need Python 3.10 or later, PyTorch (the CPU build is enough, 2.4 or later), `safetensors`, `tokenizers`, and `transformers` for the comparisons (5.0 or later, which is where the Qwen3.5 implementation lives). Three steps:

```bash
# 1. a virtual environment (uv or python -m venv, either works)
python3 -m venv .venv && source .venv/bin/activate
# 2. dependencies: the CPU build of PyTorch is enough; install the CUDA build on an NVIDIA machine, and on a Mac a plain pip install torch comes with MPS
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install safetensors tokenizers "transformers>=5.0" huggingface_hub
# 3. the weights, 1.7 GB; in China, export HF_ENDPOINT=https://hf-mirror.com first
hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B
# verify: the imports work and the weights are there
python -c "import torch, transformers, safetensors, tokenizers; print(torch.__version__, transformers.__version__)"
ls models/Qwen3.5-0.8B/config.json
```

| Dependency | Used in | For |
| --- | --- | --- |
| `torch` | everywhere | tensors, matrix multiplication, `nn.Module` |
| `safetensors` | from chapter 1 | reading the weight file (index only, or tensors by name) |
| `tokenizers` | from chapter 9 | reading tokenizer.json directly, without transformers |
| `transformers` | chapters 0–9, 12, 16 | the reference: logits in the first nine chapters, the chat template in chapter 12, the MoE layer in chapter 16 |
| `huggingface_hub` | the download | the `hf download` command |

**The one-click scripts in the repository**: on a Mac, `bash env/setup-macos.sh` installs the Python environment, PyTorch and the compiler toolchain, downloads three small models including Qwen3.5-0.8B, and creates the `transformer/models` symlink; afterwards `python tools/mac_check.py transformer` runs the first ten chapters as a relay for a self-check. On a machine with an NVIDIA GPU use `bash setup-gpu.sh` in the repository root.

**Memory and devices**: the fp32 model is 3 GB; the first nine chapters also load the HF reference, peaking at about 6 GB (`load_state_dict(assign=True)` does not copy the weights, otherwise it would be 9 GB). The scripts run on the CPU by default with the thread count set to half the logical cores — chapter 15 measured that a 32-core machine with all 32 threads is 60× slower; on a Mac there is nothing to tune. All the code can be moved with `.to("cuda")` / `.to("mps")`, and chapter 15's script picks the device automatically.

## Running and verifying {#怎么运行怎么核对}

Every chapter's script depends only on the files written in the earlier chapters, so just run them in chapter order in the same directory:

```bash
cd transformer-by-hand            # or transformer/examples/ in the repository
python main.py                    # step 0
python ch01_weights.py            # chapter 1, and so on
QWEN_MODEL=/data/models/Qwen3.5-0.8B python ch09_model.py   # point to the weights with the environment variable when they are not under models/
```

The scripts look for `models/Qwen3.5-0.8B` upwards from the current directory, so running from `examples/` or any of its subdirectories finds the repository's root `models/`.

Repository users can also use the checker to confirm that the outputs on the pages match their own machine:

```bash
python transformer/tools/check_code.py                       # run the whole book as a relay (about 10 minutes on a CPU)
python transformer/tools/check_code.py transformer/docs/cache.md --only   # run only chapter 10 (the earlier chapters only write their files into the work directory)
```

It writes every chapter's code blocks into `transformer/build/examples/tutorial/` in navigation order, runs each chapter's script, and compares the standard output line by line with the "output" block on the page (for lines with timings, everything but the numbers).

## The files each chapter creates and changes {#每一章新建和修改的文件}

The table below is computed from the chapters' code blocks. Only chapter 10 changes existing files (attention, linear attention, the decoder layer and the whole model each gain a `state` argument); every other chapter only creates:

@@files table@@

The same information opens every chapter: which files `qwen/` holds after that chapter, which are new in it (`+`) and which it changes (`~`).

!!! interview "How to explain it"
    To explain "how a project is organized", cover three things: where the sources are, where the generated artifacts are, and how the two are kept consistent. This book's answer: the text is the single source, `examples/` is an export, and `check_code.py` runs the text as a relay and compares with the outputs on the page — change the code without the output and the build fails. This "source → export → verification" loop is the same idea as "config → generated kernel → numerical alignment test" in an inference framework.

## Summary {#小结}

- [x] One directory: `models/` (the weights), `qwen/` (17 files), one script per chapter.
- [x] Three steps to install: a virtual environment, PyTorch plus four libraries, `hf download`; on a Mac, `env/setup-macos.sh`.
- [x] Run in chapter order; `check_code.py` verifies as a relay; every chapter opens by marking the files it creates and changes.
