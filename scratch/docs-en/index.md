# Train a small model from scratch

<p class="lead">This short book starts from an empty directory and trains a small language model on your own computer that writes in the style of <em>Romance of the Three Kingdoms</em>: preparing the corpus, training a tokenizer, writing the model and the training loop, evaluating and generating, then training it faster and larger; then instruction tuning so that it answers in the conversation format, a pass through post-training with LoRA, DPO and distillation, and an export into a format inference frameworks load directly; and finally moving to one consumer card to see what a night of training buys — a text model and an image generator. Every piece of code has really been run, and the main line takes about 6 minutes on a CPU. Real training has several orders of magnitude more data and compute; the procedure is the same.</p>

## Quick start {#快速开始}

Get it running first, then come back and read why each step is the way it is. A CPU is all you need, and the five scripts on the main line take about **6 minutes** in total.

```bash
# ① Environment: CPU PyTorch (2.4 or newer) and tokenizers
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install tokenizers

# ② Make an empty directory and copy in the code blocks with file names (each page downloads as a notebook too)
mkdir mylm && cd mylm

# ③ Run them in order; each script leaves its output for the next
python prepare.py        # (1) corpus → BPE tokenizer → token sequence
python train.py          # (2) train a 1.8M GPT that writes in the style of the novel
python chat_data.py      # (4) build instruction data out of the same corpus
python sft.py            # (4) instruction tuning; it starts answering in the conversation format
python export.py         # (5) export into LLaMA's weight layout

# ④ Say something to it
python talk.py
```

This is what it looks like once trained — three real questions and answers from a model with 1.8M parameters that has only ever read one novel:

```text
你：这句话是谁说的：「吾自有计。」
模型：操

你：接下来写：玄德大喜，遂引兵而进。
模型：次日，人报魏兵溃散。

你：「操大怒，拔剑欲斩之。」的下一句是什么？
模型：孔明
```

Whether the answers are right is another matter, but it really is answering in the conversation format and knows where to stop. A real model is the same pipeline with the data and the compute scaled up by a few orders of magnitude.

Every other script is an **experiment**: measuring time, checking an equivalence, comparing a few approaches. They produce nothing the main line needs, so run the ones you want, or just read the output on the page.

<div class="roadmap" markdown>

| Script | Ch. | What it does | Time | Produces |
| --- | --- | --- | --- | --- |
| `prepare.py` | [1](data.md) | download the corpus, train the BPE tokenizer, split train/validation | 4 s | `tokenizer.json`, `tokens.pt` |
| `model.py` | [2](model.md) | the model definition (imported by every later chapter) | — | — |
| `sizes.py` | [2](model.md) | the arithmetic behind vocabulary, depth/width and KV heads | 2 s | — |
| `train.py` | [2](model.md) | the full training loop, checkpointing and sampling | 2.5 min | `ckpt.pt` |
| `resume.py` | [2](model.md) | check that resuming matches an uninterrupted run exactly | 6 s | — |
| `sampling.py` | [2](model.md) | what temperature and top-k do | 2 s | — |
| `speed.py` | [3](scale.md) | where a step's time goes, and MFU | 20 s | — |
| `grad_accum.py` | [3](scale.md) | gradient accumulation equals a large batch | 2 s | — |
| `ddp_train.py` | [3](scale.md) | DDP in two processes, matching a single process | 8 s | — |
| `scaling.py` | [3](scale.md) | four model sizes, 400 steps each | 2.6 min | — |
| `budget.py` | [3](scale.md) | time and cost extrapolated to GPUs and large models | instant | — |
| `chat_data.py` | [4](sft.md) | build three kinds of instruction data from the corpus | under 1 s | `sft_*.jsonl` |
| `chat.py` | [4](sft.md) | the chat template and the loss mask (imported later) | 1 s | — |
| `sft.py` | [4](sft.md) | instruction tuning with the loss on the replies only | 3.2 min | `sft.pt`, `tokenizer_chat.json` |
| `lora.py` | [5](align.md) | change the answer format by training 1.34% of the parameters | 1 min | — |
| `dpo.py` | [5](align.md) | optimize the policy directly on preference pairs | 1.7 min | — |
| `distill.py` | [5](align.md) | distil the 1.8M teacher's distribution into a 0.62M student | 3.3 min | — |
| `export.py` | [5](align.md) | export into `LlamaForCausalLM`'s weight layout | 2 s | `minisanguo/` |

</div>

The times were measured on an Intel Xeon Platinum 8336C with `OMP_NUM_THREADS=2` and only give the order of magnitude (the whole book runs in 15 minutes); the scripts that need a GPU (`train_gpu.py` and the last chapter) are not in the table.

## Who this book is for {#这本书适合谁}

- You have read how a Transformer is built and can write PyTorch, but have never trained a language model yourself.
- You want to be able to say "I trained a model from scratch" and explain what each step does and where the time goes.
- You have only a laptop or one consumer card, and want to know what that much compute can do.

By the end, working through it, you should be able to:

- Train a BPE tokenizer from a plain text file and explain how the vocabulary size is traded off.
- Write a small pre-norm GPT and a complete training loop (the learning-rate schedule, gradient clipping, validation, checkpoints, resuming, generation).
- Measure where a step's time goes and how much of the compute is used, and then use gradient accumulation, `torch.compile`, mixed precision and DDP to train it faster and larger.
- Write a chat template, put the loss on the assistant's replies only, and fine-tune the base model into a small assistant that answers in the right format.
- Write LoRA, DPO and white-box distillation from scratch, explain what each one changes and what it costs, and export the model for an inference framework.
- Work out the memory budget for a 16 GB card: how large a model fits and how far a night of training gets.

## The six chapters {#六章的路线}

<div class="roadmap" markdown>

| Chapter | What you do | What you can do afterwards | Suggested time |
| --- | --- | --- | --- |
| [1. The corpus and the tokenizer](data.md) | prepare the corpus, train a BPE tokenizer, split the training and validation sets | choose a vocabulary size and know why pre-tokenization matters | half a day |
| [2. The model and the training loop](model.md) | write a small GPT and a complete training loop, save checkpoints, resume, sample | train your first model that writes classical Chinese | 1 day |
| [3. Faster and bigger](scale.md) | measure the time and the utilization, add gradient accumulation, compilation and multi-card DDP, then scale the model up | know where training is slow and what to count when scaling | 1 day |
| [4. From continuation to conversation](sft.md) | build instruction data, write a chat template, and fine-tune with the loss on the assistant's replies only | turn the base model into a small assistant that answers in the right format | 1 day |
| [5. Cheaper tuning and better alignment](align.md) | write LoRA, DPO and white-box distillation from scratch, then export to LLaMA's weight layout | explain what each post-training step does, and hand the model to an inference framework | 1 day |
| [What one consumer card can train](one-gpu.md) | move to a real 16 GB card: the memory budget, a text model overnight, a small image generator | have a quantitative sense of what one card can do | 1 day |

</div>

The six chapters' code is a **relay**: each chapter leaves its corpus, tokenizer and checkpoints for the next, so work through them in order in one directory.

## Where to go next {#接下来往哪走}

| If you want to | Go to |
| --- | --- |
| Write the BPE, the model and AdamW yourself, with no libraries, and pass a grading script | [the LLM handbook's assignment](llm://training/assignment/) |
| What to do when one card is not enough: DDP, ZeRO, tensor and pipeline parallelism | [the distributed training handbook](train://) |
| Why each part of the model is designed the way it is | [LLM internals](llm://) |
| How a trained model is served efficiently | [inference systems](serving://), [writing mini-sglang](minisgl://) |

The chapter-by-chapter route through every handbook is in the [learning roadmap](root://roadmap/), and the job-hunting schedule is in the [17-week plan](root://plan/).

## How it is verified {#怎么验证的}

- Every script with a file name really runs on **CPU PyTorch**, and the output on the page matches the run line by line; the six chapters run as a relay in one working directory, about 15 minutes end to end.
- The chapter that needs a GPU (the last one) is only syntax-checked, and the numbers on the page state the card they were measured on (an RTX 5070 Ti, 16 GB).

```bash
# CPU PyTorch (2.4 or newer) and tokenizers
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install tokenizers
```
