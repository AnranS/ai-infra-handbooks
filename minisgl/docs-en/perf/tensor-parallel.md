# Tensor parallelism

<p class="lead">When a model will not fit on one GPU, or one GPU's bandwidth cannot reach the latency you want, each layer's weights have to be split across several GPUs: that is tensor parallelism (TP). All those <code>get_tp_info().size</code> calls in the linear layers, the embeddings and the KV pool of earlier chapters finally come into play here. We communicate over gloo on a CPU, and TP=2 and TP=4 produce output identical to single-GPU Hugging Face.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. How many all-reduces does one decoder layer need under tensor parallelism? Where are they?
    2. Which dimension do column parallel and row parallel split? Why is the MLP "column then row"?
    3. Qwen3-0.6B has 8 KV heads. How are they divided at TP=16?
    4. Vocabulary-parallel embeddings use an all-reduce and the output layer an all-gather. Why the difference?

??? success "Answers (try it yourself first, then expand)"
    1. Two: one after attention's `o_proj` (row parallel) and one after the MLP's `down_proj` (row parallel).
    2. Column parallel splits along the output dimension (each rank computes part of the output); row parallel splits along the input dimension (each rank gets a partial sum that needs an all-reduce). Column then row: what column parallel outputs is exactly the slice of input row parallel needs, and the activation in between is elementwise and needs no communication, so the whole MLP only needs the one all-reduce at the end.
    3. 8 KV heads do not go round 16 ranks: each KV head is replicated onto 2 ranks, so each rank holds 1 KV head (along with its query heads).
    4. The embedding splits the vocabulary: each rank looks up only the tokens in its own slice and outputs zeros for the rest, and an all-reduce sums them into the complete embedding. The output layer, also split by vocabulary, computes logits for part of the vocabulary on each rank, and they have to be all-gathered into the complete distribution before sampling.

**Files to revisit**: `layers/linear.py`, `layers/embedding.py`, `shard_tensor` in `models/weight.py`, `distributed/`, and `_init_communication` in `engine/engine.py`.

@@tree@@

**This step's main**: `examples/ch16_tp.py` — it uses only the files above; `python tools/steps.py check` rebuilds this tree chapter by chapter and runs it.

@@video tp Animation: how tensor parallelism splits a layer and how often it communicates (about 1.5 minutes, Chinese narration and subtitles)@@

## Megatron-style sharding {#megatron-式切分}

A decoder layer has two "matmul, elementwise, matmul" structures: attention (`qkv_proj`, attention, `o_proj`) and the MLP (`gate_up_proj`, SiLU × up, `down_proj`). Megatron-LM's way of splitting them (the idea is in the [tensor-parallelism chapter of the Inference Systems handbook](serving://distributed/tensor-parallel/)):

- **split the first matmul along the output dimension** (column parallel): each rank computes some of the output channels. Attention splits **by head**: each rank owns a few heads, and heads are independent of each other, so attention itself needs no communication at all. The MLP splits along the intermediate dimension, and SiLU × up is elementwise, which needs no communication either.
- **split the second matmul along the input dimension** (row parallel): each rank multiplies its slice of the intermediate result by its slice of the weight and gets a full-shaped **partial sum**, and one **all-reduce** adds them at the end.

So each layer has two all-reduces, after `o_proj` and after `down_proj`, and everything else is local.

@@diagram tp-sharding the data flow of one decoder layer under tensor parallelism@@

@@code python/minisgl/layers/linear.py:LinearQKVMerged@@

@@code python/minisgl/layers/linear.py:LinearRowParallel@@

A merged `qkv_proj` needs care when sharding: cutting the merged matrix straight down the middle would give all of q to rank 0 and k and v to rank 1. Instead q, k and v are split separately and concatenated on each rank. That is exactly what the streaming loader does, `shard_tensor` first and merge second (chapter 3).

## Replicating KV heads when there are not enough {#kv-头不够分时复制}

@@code python/minisgl/models/weight.py:shard_tensor@@

A GQA model has fewer KV heads than q heads. Qwen3-0.6B has 16 q heads and 8 KV heads: at TP=2 each rank gets 8 q heads and 4 KV heads; at TP=16 each rank gets 1 q head, but there are only 8 KV heads, so every two ranks share one KV head (`div_even(..., allow_replicate=True)` returns 1) and each keeps its own copy. That wastes a little memory but keeps attention free of communication.

## Vocabulary parallelism {#词表并行}

The embedding's vocabulary is split by row, one slice per rank; a lookup outputs zeros for tokens outside the slice, and an **all-reduce** sums them, since each token finds its real vector on exactly one rank. The output layer is split by vocabulary too, with each rank computing the logits for its slice; sampling needs the complete distribution, so an **all-gather** stitches the slices together (`ParallelLMHead.forward` in chapter 2). The first adds the ranks' results, the second concatenates them.

## Running it {#运行}

@@code examples/ch16_tp.py@@

@@output ch16_tp@@

- Weight shapes: at TP=2, q_proj, k_proj and gate_proj halve along the output dimension, o_proj and down_proj halve along the input dimension, the vocabulary halves, and the norms are not split; at TP=16 k_proj still has 128 rows, one complete KV head, replicated.
- Both ranks' `qkv_proj` has (8 + 2 × 4) × 128 = 2048 rows, and each rank's KV pool holds only 4 KV heads.
- One forward pass does 57 all-reduces (28 layers × 2, plus 1 in the embedding) and 1 all-gather (the output layer), exactly as the analysis says.
- The logits differ from single-GPU by about 1e-5 at most: an all-reduce changes the order of floating-point addition, so the results are no longer bit-identical, but greedy decoding picks exactly the same tokens (this chapter's tests verify TP=2 and TP=4 at the service level).

## How communication is implemented {#通信的实现}

@@code python/minisgl/distributed/impl.py:TorchDistributedImpl@@

`DistributedCommunicator` picks an implementation from a plugin list, defaulting to `torch.distributed`: NCCL on a GPU, gloo on a CPU. gloo does not support `all_gather_into_tensor`, so on a CPU we use `all_gather` writing into slices of the output tensor.

The process group is created as the engine initializes:

@@code python/minisgl/engine/engine.py:Engine._init_communication@@

On a GPU a separate gloo group is created for control information on the CPU (the broadcast count of chapter 14, the memory sync at startup).

!!! diff "Difference from upstream: PyNCCL"
    By default (`use_pynccl=True`) upstream does not use `torch.distributed` for GPU communication but calls NCCL directly through tvm-ffi (`kernel/csrc/src/pynccl.cu`), with a preallocated communication buffer. That costs less overhead and gives more control over how it interacts with CUDA Graph. We implement only `torch.distributed` and leave PyNCCL as an exercise (it needs a GPU).

!!! upstream "The official implementation"
    - the linear layers: @@upstream layers/linear.py:LinearQKVMerged@@, @@upstream layers/linear.py:LinearRowParallel@@
    - sharding: @@upstream models/weight.py:_shard_tensor@@
    - communication: @@upstream distributed/impl.py:DistributedCommunicator@@
    - the process group: @@upstream engine/engine.py:Engine._init_communication@@

## Tests {#测试}

@@code tests/test_ch16_tp.py:test_tensor_parallel_matches_hf@@

Each TP configuration brings up a complete service (several scheduler processes, the tokenizer, the API server), and the greedy output for 3 prompts matches single-GPU Hugging Face word for word. That also verifies chapter 14's multi-rank message synchronization.

!!! interview "How to explain it"
    On tensor parallelism: the Megatron split shards attention by head (qkv column parallel, o_proj row parallel) and the MLP column parallel first (gate/up along the intermediate dimension) then row parallel (down), so each decoder layer needs only two all-reduces, one after attention and one after the MLP. A merged qkv has to be split per q, k and v before being concatenated, or the ranks get the wrong heads; when there are fewer KV heads than ranks (Qwen3-0.6B's 8 KV heads at TP=16) each KV head is replicated onto two ranks. Vocabulary parallelism: the embedding looks up only its own slice and zeroes the rest before an all-reduce, while the output layer computes its slice's logits and all-gathers them. Each rank differs from single-GPU only by the tiny error from the order of floating-point summation.

## Exercises {#练习}

1. Compute how much data the two all-reduces per layer move for Llama-3.1-70B (80 layers, hidden 8192, bf16) at TP=8 with a decode batch of 64. At 450 GB/s of NVLink bandwidth one way, roughly what fraction of the time is communication?
2. If the TP size does not divide the q head count (Qwen2.5-0.5B has 14 q heads at TP=4), where does it fail? How would you support it?
3. Implement a PyNCCL plugin: load `libnccl.so` with `ctypes` and implement `all_reduce`; compare it against `torch.distributed` on a GPU.

??? success "Answers"
    1. Each all-reduce moves `[64, 8192]` bf16 values, 1 MiB; a ring all-reduce has each rank send about 2 × (8−1)/8 × 1 MiB ≈ 1.75 MiB, roughly 4 µs of bandwidth time, plus a few microseconds of fixed latency per all-reduce. With 80 layers × 2 ≈ 160 of them, that is about 1 to 2 ms in total. At small batch sizes the fixed latency dominates, which is why the TP size should not get too large during decode.
    2. The `div_even(num_qo_heads, tp_size)` assertion in `LinearQKVMerged` fails. You can pad the head count (adding empty heads) or support only divisible TP sizes; production engines usually just require divisibility.
    3. Follow the official `kernel/pynccl.py` and SGLang's `pynccl_wrapper.py`: rank 0 creates an ID with `ncclGetUniqueId` and broadcasts it over the CPU process group; each rank calls `ncclCommInitRank`; and `all_reduce` calls `ncclAllReduce` on the current CUDA stream.

## Summary {#小结}

- [x] The Megatron split: column parallel first (by head or by intermediate dimension), row parallel second, two all-reduces per layer.
- [x] A merged qkv is split per projection and then concatenated; KV heads are replicated across ranks when there are not enough.
- [x] Vocabulary parallelism: the embedding all-reduces a sum, the output layer all-gathers a concatenation.
- [x] Each rank's output differs from single-GPU only by the tiny error from the order of floating-point summation, and greedy decoding agrees.
