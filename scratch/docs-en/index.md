# Train a small model from scratch

<p class="lead">This short book starts from an empty directory and trains a small language model on your own computer that writes in the style of <em>Romance of the Three Kingdoms</em>: preparing the corpus, training a tokenizer, writing the model and the training loop, evaluating and generating, then training it faster and larger, and finally moving to one consumer card to see what a night of training buys — a text model and an image generator. Every piece of code has really been run, and one training run takes about 3 minutes on a CPU. Real pretraining has several orders of magnitude more data and compute; the procedure is the same.</p>

## Who this book is for {#这本书适合谁}

- You have read how a Transformer is built and can write PyTorch, but have never trained a language model yourself.
- You want to be able to say "I trained a model from scratch" and explain what each step does and where the time goes.
- You have only a laptop or one consumer card, and want to know what that much compute can do.

By the end, working through it, you should be able to:

- Train a BPE tokenizer from a plain text file and explain how the vocabulary size is traded off.
- Write a small pre-norm GPT and a complete training loop (the learning-rate schedule, gradient clipping, validation, checkpoints, resuming, generation).
- Measure where a step's time goes and how much of the compute is used, and then use gradient accumulation, `torch.compile`, mixed precision and DDP to train it faster and larger.
- Work out the memory budget for a 16 GB card: how large a model fits and how far a night of training gets.

## The four chapters {#四章的路线}

<div class="roadmap" markdown>

| Chapter | What you do | What you can do afterwards | Suggested time |
| --- | --- | --- | --- |
| [1. The corpus and the tokenizer](data.md) | prepare the corpus, train a BPE tokenizer, split the training and validation sets | choose a vocabulary size and know why pre-tokenization matters | half a day |
| [2. The model and the training loop](model.md) | write a small GPT and a complete training loop, save checkpoints, resume, sample | train your first model that writes classical Chinese | 1 day |
| [3. Faster and bigger](scale.md) | measure the time and the utilization, add gradient accumulation, compilation and multi-card DDP, then scale the model up | know where training is slow and what to count when scaling | 1 day |
| [What one consumer card can train](one-gpu.md) | move to a real 16 GB card: the memory budget, a text model overnight, a small image generator | have a quantitative sense of what one card can do | 1 day |

</div>

The four chapters' code is a **relay**: each chapter leaves its corpus, tokenizer and checkpoints for the next, so work through them in order in one directory.

## Where to go next {#接下来往哪走}

| If you want to | Go to |
| --- | --- |
| Write the BPE, the model and AdamW yourself, with no libraries, and pass a grading script | [the LLM handbook's assignment](llm://training/assignment/) |
| What to do when one card is not enough: DDP, ZeRO, tensor and pipeline parallelism | [the distributed training handbook](train://) |
| Why each part of the model is designed the way it is | [LLM internals](llm://) |
| How a trained model is served efficiently | [inference systems](serving://), [writing mini-sglang](minisgl://) |

The chapter-by-chapter route through every handbook is in the [learning roadmap](root://roadmap/), and the job-hunting schedule is in the [17-week plan](root://plan/).

## How it is verified {#怎么验证的}

- Every script with a file name really runs on **CPU PyTorch**, and the output on the page matches the run line by line; the four chapters run as a relay in one working directory.
- The chapter that needs a GPU (the fourth) is only syntax-checked, and the numbers on the page state the card they were measured on (an RTX 5070 Ti, 16 GB).

```bash
# CPU PyTorch (2.4 or newer) and tokenizers
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install tokenizers
```
