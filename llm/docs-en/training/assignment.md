# Capstone: train a small language model from scratch

<p class="lead">After reading this book, the best test is to train a language model from scratch with your own hands: write the tokenizer, the model, the optimizer and the training loop yourself, and get the validation loss below the target within a time limit. Following CS336's approach, this capstone gives only interfaces, tests and a grading script, no skeleton: you write every line of code, and wherever you get stuck is a chapter worth rereading.</p>

The code and instructions are in [`assignments/a1-lm/`](https://github.com/AnranS/ai-infra-handbooks/tree/main/assignments/a1-lm) in the repository.

!!! tip "To follow along once first"
    The [Train a Small Model](scratch://) book walks through the same pipeline end to end with off-the-shelf libraries (`tokenizers`, `torch.optim.AdamW`): the corpus, the tokenizer, a small GPT, the training loop, resuming, sampling, then multiple GPUs and scale estimates. Read it first for the big picture, then come back and implement each part yourself as this assignment requires.

## What to build {#要做什么}

| File | Contents | Chapter | Checks |
| --- | --- | --- | --- |
| `lm/bpe.py` | byte-level BPE: training, encoding, decoding; how merges and ties are handled is written at the top of the file | [Tokenization](../basics/tokenization.md) | 5 tests |
| `lm/model.py` | a pre-norm Transformer: RMSNorm, causal multi-head attention with RoPE, SwiGLU | [Anatomy of the Transformer](../transformer/build-llm.md) | 4 tests (shapes, parameter count, causality) |
| `lm/optim.py` | your own AdamW (matching PyTorch elementwise), linear warmup + cosine schedule, gradient clipping | [Pretraining](pretraining.md) | 3 tests |
| `train.py` (create it yourself) | the training loop: tokenize, sample random windows, forward and backward, save checkpoints | — | grading script |

Rules: you may use PyTorch tensor operations, autograd, `nn.Linear` and `nn.Embedding`; you may not use `torch.optim.AdamW`, `nn.MultiheadAttention`, `nn.Transformer*`, `clip_grad_norm_` or ready-made tokenizer libraries.

## Steps and the target {#步骤与达标线}

```bash
cd assignments/a1-lm
python data.py                       # generate about 4 MB of training text (deterministic "short stories" with long-range dependencies on character names)
python -m pytest -q tests            # all 12 tests pass
python train.py                      # write it yourself; it saves out/model.pt
python evaluate.py out/model.pt      # validation cross-entropy per byte ≤ 0.10 nats/byte
```

- **Scored "per byte"**: the total loss divided by the number of bytes in the validation set, independent of vocabulary size, so different tokenization schemes compare fairly;
- **Anti-cheating**: the grading script first checks that the tokenizer restores the validation set losslessly and that the model is causal (changing later tokens does not affect earlier outputs);
- **Time limit**: under 10 minutes on a CPU. The reference implementation (vocabulary 512, 4 layers, d_model 128, context 256, 1500 steps) takes about 2.5 minutes with 16 threads and reaches 0.089 nats/byte.

## Afterwards {#做完之后}

- Measure training throughput and model FLOPs utilization, and compare with the formulas in the [estimation](../inference/estimation.md) chapter;
- Ablations: remove RoPE, switch to post-norm, change the vocabulary size, and see how each affects loss and speed;
- Hand the model to the [distributed training capstone](train://practice/assignment/) for multi-process training, or plug it into an inference engine.
