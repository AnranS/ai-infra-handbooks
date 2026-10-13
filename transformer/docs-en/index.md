# Transformer by Hand

<p class="lead">Start from one <code>main.py</code> and, one file per chapter, <strong>write the newest Qwen architecture by hand</strong>: zero-centered RMSNorm, SwiGLU, RoPE that rotates only a quarter of each head, GQA attention with an output gate, Gated DeltaNet linear attention (recurrent and chunked), the KV cache and the memory table, sampling, the chat template, batching, MTP speculative decoding, MoE. Every step runs. In the first eight chapters, every part we write comes with a small experiment that shows what it does — who is close to whom in embedding space, what the attention matrix looks like, why the memory table does not grow with length; chapter 9 assembles the whole model and, for the only time in the book, checks its logits against Hugging Face (differences around 1e-5); every chapter after that checks itself with "stateful == stateless", "batch == single" or "hand-written template == official template". The example model is <strong>Qwen3.5-0.8B</strong>: the only model of the 3.5 / 3.6 / 3.8 generation that runs comfortably on a CPU or a Mac, while the code we write applies unchanged to the 27B and the MoE versions.</p>

## Who this book is for {#这本书适合谁}

- You have read *LLM Internals* and know what attention is, but have never written a real model starting from its weight file;
- You want to understand how this generation of open models differs from the Llama-style architecture: linear attention, gating, QK-Norm, M-RoPE, MTP — what are they, exactly;
- You are heading for inference systems and want to measure for yourself what the KV cache, the memory table, batching and speculative decoding each save.

After reading and typing along, you should be able to:

- Look at config.json and the safetensors index and say what every tensor is and why it has that shape;
- Write Qwen3.5's forward pass without the reference implementation, and know how to verify it;
- Explain the delta rule of Gated DeltaNet, why it can be parallelized in chunks and why its state does not grow with length;
- Turn a model that "only computes logits" into a small engine that streams a chat, batches requests and decodes speculatively.

## How it unfolds {#怎么展开}

![Figure: the route through this book: step 0, three stages, how each stage is verified and what you hold at its end](assets/figures/roadmap.svg){.aig-svg}

<div class="roadmap" markdown>

| Stage | Chapters | What you do | How it is verified |
| --- | --- | --- | --- |
| [Step 0](main.md) | main.py | HF as a black box, your own decode loop; open the weight file and draw the map of the book | token for token the same as `model.generate` |
| Stage 1 · Writing the parts one by one | [1](weights.md) [2](embedding.md) [3](norm.md) [4](mlp.md) [5](rope.md) [6](attention.md) [7](deltanet.md) [8](chunk.md) [9](model.md) | one file per chapter, writing every part from the weight file up | one small experiment per chapter showing what the part does; chapter 9 assembles the whole model and checks its logits against HF once, and transformers is never imported again |
| Stage 2 · From a model to an engine | [10](cache.md) [11](sample.md) [12](chat.md) [13](batch.md) [14](mtp.md) | KV cache and memory table, sampling, the chat template and thinking mode, batching, MTP speculative decoding | stateful == stateless; hand-written template == official template; batch == single; speculative == greedy |
| Stage 3 · Faster and bigger | [15](speed.md) [16](scale.md) | threads, chunking, bf16, devices; the parameter budget of the whole family, a hand-written MoE | real timings; the MoE matched to transformers with random weights |

</div>

The code is a relay: chapter N's script uses every file written in the chapters before it. The package ends up as 17 files, about 800 lines including comments; the file tree is in [step 0](main.md#后面怎么走).

## How to run it {#怎么跑}

The directory layout, the environment and the files each chapter touches are on [Project layout and setup](setup.md); the shortest version is three commands (they need [uv](https://docs.astral.sh/uv/); the dependencies are the CPU build of PyTorch, `safetensors`, `tokenizers`, and `transformers` 5.x for step 0 and chapters 9 and 16):

```bash
cd transformer && uv sync                                           # the project is managed with uv: create the environment and install the dependencies from pyproject.toml
uv run hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B   # in China, export HF_ENDPOINT=https://hf-mirror.com first
uv run python examples/main.py                                      # step 0; after that, one script per chapter
```

The code lives under `transformer/examples/` in the repository (the `qwen/` package plus one `chNN_*.py` per chapter) and runs from the `transformer/` directory with `uv run python examples/chNN_xxx.py`. The scripts look for `models/Qwen3.5-0.8B` upwards from the current directory; the environment variable `QWEN_MODEL` can point elsewhere.

- **Memory**: one fp32 copy of the model is 3 GB; step 0 and chapter 9 also load the HF reference, about 6 GB in total.
- **CPU**: the scripts set the thread count to half the logical cores (chapter 15 explains why); a 32-core machine decodes at about 18 tok/s, and an Apple Silicon Mac runs it on the CPU as well.
- **GPU**: everything can be moved with `.to("cuda")` / `.to("mps")`; chapter 15's script does so automatically.

## Where to go next {#接下来往哪走}

| If you want to | Go to |
| --- | --- |
| Turn this small engine into a real inference service: scheduling, the radix cache, CUDA Graphs, tensor parallelism | [mini-sglang from Scratch](minisgl://) |
| How these operators are written fast on a GPU | [Advanced CUDA](cuda://) |
| The ideas behind attention, RoPE and MoE and their variants | [LLM Internals](llm://) |
| Train a model like this from scratch | [Train a Small Model](scratch://) |
| The whole engineering picture of inference serving: the architecture and scheduling of vLLM / SGLang | [Inference Systems](serving://) |

The chapter-by-chapter route through all the handbooks is on the [roadmap](root://roadmap/); the job-hunting schedule is the [sprint plan](root://plan/).

## How it is verified {#怎么验证的}

- Every chapter's script really runs on a CPU, and the output on the page matches the run line by line (`transformer/tools/check_code.py` re-runs the whole book as a relay in navigation order);
- Chapter 9's whole model is compared with `Qwen3_5ForCausalLM` from Hugging Face transformers 5.17 on logits: the maximum difference is around 1e-5 (the book's only comparison with HF; a match means every part before it is right); the chunked linear attention matches the recurrent one element by element; the hand-written chat template matches the official Jinja template token by token; the MoE layer matches transformers with random weights to 0.
- Numbers with time in them (tok/s, ms) are real timings on a 32-core Xeon and will differ on another machine.
