# MoE and fused MoE

<p class="lead">An MoE model like Qwen3-30B-A3B has hundreds of experts per layer and activates only a few per token. The hard part is that the tokens in a batch go to different experts, and computing expert by expert would launch hundreds of small matmuls. Fused MoE sorts all the "(token, expert) pairs" by expert and pads each expert's run to a multiple of the block size, so a whole MoE layer needs only two kernel launches. This chapter implements the MoE layer, a reference backend and a fused MoE written in Triton, and verifies it on a CPU with the Triton interpreter.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. How is MoE routing computed? What does `norm_topk_prob` mean?
    2. After sorting the (token, expert) pairs by expert, why is each expert's run padded to a multiple of `BLOCK_M`?
    3. Under tensor parallelism, how are the expert weights sharded? What communication is needed?
    4. How do you run a Triton kernel on a machine with no GPU?

??? success "Answers (try it yourself first, then expand)"
    1. The router computes a score per expert, and after a softmax the top-k experts and their weights are taken; `norm_topk_prob` means renormalizing those k weights (dividing by their sum) so they add up to 1.
    2. In the fused MoE kernel each output tile (`BLOCK_M` rows) can belong to only one expert and multiplies by that expert's weight, so padding each expert's run to a multiple of `BLOCK_M` keeps a tile from straddling two experts.
    3. Every expert's weights split along the intermediate dimension: gate / up by column, down by row, so each rank holds part of every expert. The router is not split (every rank computes the full routing). One all-reduce at the end of the layer adds the partial sums.
    4. Use Triton's interpreter mode (`TRITON_INTERPRET=1`): the kernel runs tile by tile on the CPU with numpy and the same semantics, which verifies correctness (performance means nothing there).

**Files you will write**: `layers/moe.py`, `moe/base.py`, `moe/torch_backend.py`, `moe/fused.py`, `moe/__init__.py`, and `MoEMLP` in `models/utils.py`.

## The MoE layer {#moe-层}

@@code python/minisgl/models/utils.py:MoEMLP@@

`gate` is an unsharded linear layer (a full copy per rank) that scores every token against every expert. `MoELayer` holds all the experts' packed weights and leaves the computation to the MoE backend in the Context:

@@code python/minisgl/layers/moe.py:MoELayer@@

`gate_up_proj` has shape `[experts, 2 × intermediate, hidden]`, with each expert's gate and up concatenated (just like a dense MLP). Under tensor parallelism every expert's intermediate dimension is split, exactly as in a dense MLP: column parallel first, row parallel second, one all-reduce at the end. The weight loader reads the separately stored `experts.0.gate_proj`, `experts.1.gate_proj` and so on from the checkpoint and packs them into a three-dimensional tensor (chapter 3).

## Routing {#路由}

@@code python/minisgl/moe/base.py:select_experts@@

Take the top k experts after a softmax; with `norm_topk_prob=True` (Qwen3-MoE) those k weights are renormalized to sum to 1.

## The reference backend {#参考后端}

@@code python/minisgl/moe/torch_backend.py:TorchMoeBackend@@

Grouped by expert: for each selected expert, pick out the tokens routed to it, run one SwiGLU MLP, multiply by the routing weight, and accumulate back onto the tokens with `index_add_`. Straightforward, but as many rounds of small matmuls as there are experts.

## Fused MoE {#fused-moe}

@@diagram fused-moe sort by expert, pad, then two fused GEMMs@@

@@code python/minisgl/moe/fused.py:moe_align_block_size@@

The first step sorts all the (token, expert) pairs by expert and pads each expert's run to a multiple of `BLOCK_M`, filling the padding with a placeholder (equal to the pair count). Every `BLOCK_M` consecutive pairs then belong to one expert, so a thread block can handle the matmul of those `BLOCK_M` tokens against **one expert's** weights.

@@code python/minisgl/moe/fused.py:fused_moe_kernel@@

Each program instance owns one `[BLOCK_M, BLOCK_N]` tile of the output: it reads that tile's expert id, gathers `BLOCK_M` input rows by the sorted indices (in the first GEMM a token corresponds to `TOP_K` pairs, so the row is `pair index // TOP_K`), and does a tiled matmul against that expert's weights, with the placeholder rows masked out. The second GEMM multiplies by the routing weight along the way.

@@code python/minisgl/moe/fused.py:FusedMoeBackend.forward@@

A whole MoE layer: sort (in PyTorch), the first fused GEMM (`w1`), SiLU × up, the second fused GEMM (`w2`, times the routing weight), then sum the k results per token.

## Running Triton on a CPU {#在-cpu-上运行-triton}

Triton has an interpreter mode (`TRITON_INTERPRET=1`): the kernel is interpreted program instance by program instance on the CPU with NumPy, with the same semantics as on a GPU, only very slow. It must be set before `triton` is imported.

@@code examples/ch20_moe.py@@

@@output ch20_moe@@

5 tokens choosing 2 experts each make 10 pairs. After sorting, expert 1's two pairs (the 3rd and 4th) form the first tile with two placeholders added; expert 3 has only one pair and fills a tile as well, for 6 tiles in all. A bigger block wastes more on padding; a smaller one makes the matmul less efficient. The fused MoE on 37 tokens differs from the reference by about float32 rounding.

## End to end {#端到端}

A small two-layer, 8-expert Qwen3-MoE model built from random weights is compared against the Hugging Face implementation (the reference backend is the default on a CPU):

@@code tests/test_ch20_moe.py:test_tiny_qwen3_moe_matches_hf@@

!!! upstream "The official implementation"
    - @@upstream layers/moe.py:MoELayer@@, @@upstream models/qwen3_moe.py@@
    - fused MoE: @@upstream moe/fused.py@@ and `kernel/triton/fused_moe.py`. The routing (`topk_softmax`) and the sorting and padding (`moe_align_block_size`) use CUDA kernels from sgl_kernel, and the block size is one of two choices by shape: small tiles (16 × 32 × 64) when the token count does not exceed the expert count, large ones (64 × 64 × 32) otherwise, which is exactly what exercise 1 is about

!!! diff "Differences from upstream"
    Our Triton kernel is a simplified version: the block sizes are fixed (`BLOCK_M=16`, `BLOCK_N=32`, `BLOCK_K=32`) rather than chosen by shape, there is no `GROUP_SIZE_M` grouping (which improves L2 hit rate) as in the official kernel, and the sorting and padding use PyTorch rather than a CUDA kernel. The `torch` reference backend is ours and is the default on a CPU.

## Tests {#测试}

@@code tests/test_ch20_moe.py:test_fused_triton_moe_in_interpreter@@

`tests/test_ch20_moe.py` also verifies the reference backend against the most literal "per token, per expert" loop, and Llama 3's long-context RoPE from chapter 2 (again a small random-weight model compared with Hugging Face).

!!! interview "How to explain it"
    On MoE: routing is softmax, then top-k, then an optional renormalization (`norm_topk_prob` makes the k selected weights sum to 1). Fused MoE sorts the (token, expert) pairs by expert and pads each expert's run to a multiple of `BLOCK_M`, so every output tile belongs to one expert and a single kernel computes every expert's GEMM (two launches per layer, gate_up and down) instead of one launch per expert. Under tensor parallelism each expert is split along the intermediate dimension, like a dense MLP, with one all-reduce; expert parallelism splits by expert instead and needs an all-to-all. Without a GPU, Triton's interpreter mode runs the kernel on a CPU with the same semantics to verify correctness.

## Exercises {#练习}

1. With `BLOCK_M=64`, 128 experts, and only 32 tokens per decode step (each choosing 8 experts), how many pairs are there after padding? What fraction is wasted? What does that mean for decode?
2. Expert parallelism (EP) puts different experts on different ranks, so every token has to travel to its expert's rank. How does its communication pattern differ from this chapter's tensor-parallel split? (See the [expert-parallelism chapter of the Inference Systems handbook](serving://distributed/expert-parallel/).)
3. Fuse SiLU × up into the first GEMM's output: the kernel has to compute both the gate and the up tile. How would you arrange `offs_n` so that one program instance gets the matching gate and up columns?

??? success "Answers"
    1. 256 pairs spread over at most 128 experts, so most experts have only 1 to 3 pairs and each pads to 64: at worst about 128 × 64 = 8192 rows for 256 useful ones, around 97% wasted. The fused MoE kernel is very inefficient during decode, which is why decode often uses smaller tiles or kernels designed for small batches (a GEMV per expert, for instance).
    2. Tensor parallelism gives every rank part of every expert and communicates with one all-reduce (whose volume is proportional to tokens × hidden); expert parallelism gives every rank all of some experts and needs an all-to-all to send tokens to their experts' ranks and gather the results back, with a volume that depends on the routing distribution and can be badly unbalanced.
    3. Have the program instance own a stretch `[n, n + BLOCK_N)` of the intermediate dimension, load the gate columns `[n, n+BLOCK_N)` and the up columns `[I + n, I + n + BLOCK_N)` together, do two `tl.dot`s, and write out `[BLOCK_M, BLOCK_N]` after `silu(gate) * up`.

## Summary {#小结}

- [x] The MoE layer: an unsharded router plus expert weights packed into a three-dimensional tensor; under tensor parallelism each expert splits along the intermediate dimension with one all-reduce.
- [x] Routing: softmax, top-k, optional renormalization.
- [x] Fused MoE: sort by expert and pad to the block size so every output tile belongs to one expert, giving two kernel launches per layer.
- [x] The Triton interpreter runs the kernel on a CPU with the same semantics, which is enough to verify correctness.
