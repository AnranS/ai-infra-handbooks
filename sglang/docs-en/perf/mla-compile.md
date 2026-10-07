# MLA and torch.compile: v0.3's two sources

<p class="lead">The v0.3 blog post of 4 September 2024 is titled "DeepSeek MLA 7x faster, torch.compile 1.5x faster, LLaVA-OneVision with several images and videos". The first two are this chapter's subject: MLA is the first time the memory pool and the kernels were redone for a new attention structure, and the start of SGLang's binding to the DeepSeek family; torch.compile adds a layer of compilation on top of CUDA graphs to push the latency down at small batches. Both begin with one commit in early August and a switch that already existed.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. How does MLA's KV cache differ from multi-head attention's? Which three places did SGLang change for it?
    2. What is weight absorption? Why use it during decode and not necessarily during prefill?
    3. What does torch.compile compile in SGLang, and what does it not? How does it relate to CUDA graphs?
    4. Under what conditions do the v0.3 blog's 3 to 7 times and 1.5 times hold?

??? success "Answers for the self-test (answer first, then open this)"
    1. MLA caches one compressed latent vector per token per layer (of `kv_lora_rank` dimensions) plus a short stretch of RoPE dimensions (`qk_rope_head_dim`), rather than a K and a V for every KV head. The changes: the memory pool gained an `MLATokenToKVPool` (one vector of `kv_lora_rank + qk_rope_head_dim` per slot); the attention kernels support q and k differing in dimension (the `kv_lora_rank` parameter of `extend_attention` and `token_attention`); and `DeepseekV2AttentionMLA` in the model file computes attention directly in the latent space with the absorbed weights.
    2. Absorb `W_UK` (the matrix that decompresses K from the latent vector) into the query side: `q · (W_UK c)ᵀ = (q W_UKᵀ) · cᵀ`, so the attention scores can be computed on the compressed latent vectors without decompressing every historical token's K; likewise `W_UV` moves to the output side. During decode there is one query per step scoring against the whole history, and decompressing the history's K costs far more than transforming one query, so absorption pays; during prefill there are as many queries as keys, the decompression is relatively cheap, and absorption enlarges each head's q dimension and makes the matrix multiply more expensive — so each path has its range.
    3. It compiles the ordinary operators inside a Transformer layer — the linear layers, the normalisations, the activations (`torch.compile(model.forward, mode="max-autotune-no-cudagraphs")`) — while attention and sampling keep FlashInfer's kernels; the compilation happens before the CUDA graph is captured, so the compiled forward pass is what gets captured. It is therefore a stacking of "compilation reduces the kernel count and fuses operators" with "replaying a graph reduces the launch overhead", and it is only on for batches within `--torch-compile-max-bs`.
    4. 3 to 7 times: DeepSeek-V2 (Lite at TP=1, the large model at TP=8) on H100s, the ShareGPT dataset, BF16 and FP8, against vLLM's throughput. 1.5 times: the latency at small batches (1 to 32), against SGLang itself without compilation.

A six-panel strip first, then the chapter:

<!-- comic ../assets/comics/mla-compile.webp is in Chinese; put it back once the English version exists -->

## MLA: starting from a 439-line commit {#mla从一个-439-行的提交开始}

```bash title="mla-commits.sh"
REF=${REF:-29f6d408c0}
git log --reverse --date=short --format='%ad  %h  %s' "$REF" | awk '$1 >= "2024-07-20" && $1 <= "2024-09-10"' | grep -iE 'mla|deepseek' | cut -c1-96
```

```text title="output"
2024-07-21  eedc12e12e  Support Deepseek MoE Model (#689)
2024-07-26  679ebcbbdc  Deepseek v2 support (#693)
2024-08-05  e1eae1fd15  Support MLA for DeepSeek-V2 with Triton - step 1 (#905)
2024-08-13  65915f9f3e  fix: temporary solution for DeepSeek V2 H100 layout conversion issue (#1
2024-08-19  df191254ab  Optimize MLA/GQA/MQA Triton decoding (#1138)
2024-08-30  f414352ae6  Transpose mla weight offline (#1261)
2024-09-01  54772f784a  feat: fix fp8 for MLA and support bmm fp8 for DeepSeek V2 (#1285)
```

#693 of 26 July made DeepSeek-V2 run, but by the path of treating MLA as ordinary multi-head attention: decompress the full K and V from the latent vector and store them in the ordinary KV pool. Ke Bao's #905 "Support MLA for DeepSeek-V2 with Triton - step 1" of 5 August is the real MLA implementation, changing 10 files and 439 lines:

| File | The change |
| --- | --- |
| `mem_cache/memory_pool.py` | a new `MLATokenToKVPool`: one vector of `kv_lora_rank + qk_rope_head_dim` dimensions per slot per layer, instead of a K and a V |
| `layers/extend_attention.py`, `token_attention.py` | the Triton kernels support q having a larger dimension than v (`kv_lora_rank`) |
| `models/deepseek_v2.py` | a new class `DeepseekV2AttentionMLA`, with weight absorption |
| `server_args.py` | a `--disable-mla` switch (the MLA path on by default) |

Two pools stand side by side in v0.3.0's memory pool:

```python title="python/sglang/srt/mem_cache/memory_pool.py @ v0.3.0 L56-62,146-150,204-216"
class BaseTokenToKVPool(ABC):
    """A memory pool that maps a token to its kv cache locations"""

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
...
class MHATokenToKVPool(BaseTokenToKVPool):

    def __init__(
        self,
        size: int,
...
class MLATokenToKVPool(BaseTokenToKVPool):

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
    ):
        super().__init__(size, dtype)

        self.kv_lora_rank = kv_lora_rank
```

`BaseTokenToKVPool` abstracts "read and write each layer's KV by slot", `MHATokenToKVPool` is the first version's double K/V buffer, and `MLATokenToKVPool` stores one latent vector per slot. [Chapter three](../origins/radix-v1.md)'s radix tree is entirely unaffected: it deals in slot indices and not in what a slot holds — the first time the design of "the tree and the pool at two levels" shows its worth.

The weight absorption happens once when the model loads:

```python title="python/sglang/srt/models/deepseek_v2.py @ v0.3.0 L735-745" linenums="735"
            for layer_id in range(self.config.num_hidden_layers):
                self_attn = self.model.layers[layer_id].self_attn
                w_kc, w_vc = self_attn.kv_b_proj.weight.unflatten(
                    0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
                ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
                self_attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
                self_attn.w_vc = w_vc.contiguous().transpose(1, 2)
                if hasattr(self_attn.kv_b_proj, "weight_scale"):
                    self_attn.w_scale = self_attn.kv_b_proj.weight_scale
                del self_attn.kv_b_proj

```

`kv_b_proj`'s weights are split into `w_kc` (K's decompression matrix) and `w_vc` (V's), and decode uses them directly in a batched matrix multiply:

```python title="python/sglang/srt/models/deepseek_v2.py @ v0.3.0 L447-460,474-488"
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        if self.w_kc.dtype == torch.float8_e4m3fn:
            q_nope_val, q_nope_scale = input_to_float8(
                q_nope.transpose(0, 1), torch.float8_e4m3fn
            )
            q_nope_out = bmm_fp8(
                q_nope_val, self.w_kc, q_nope_scale, self.w_scale, torch.bfloat16
            )
        else:
            q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)
        q_input[..., : self.kv_lora_rank] = q_nope_out.transpose(0, 1)

        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
...
        if self.w_vc.dtype == torch.float8_e4m3fn:
            attn_output_val, attn_output_scale = input_to_float8(
                attn_output.transpose(0, 1), torch.float8_e4m3fn
            )
            attn_bmm_output = bmm_fp8(
                attn_output_val,
                self.w_vc,
                attn_output_scale,
                self.w_scale,
                torch.bfloat16,
            )
        else:
            attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), self.w_vc)
        attn_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        output, _ = self.o_proj(attn_output)
```

`q_nope` is multiplied by `w_kc` into the latent space first (`torch.bmm`, one small matrix per head), attention is computed on the latent vectors, and the output is multiplied by `w_vc` back into the value space. Under FP8 it uses `bmm_fp8` — added by #1285 on 1 September, and the source of the blog's "FP8 batched MatMul". #1138 "Optimize MLA/GQA/MQA Triton decoding" of 19 August is the source of the blog's "grouped decoding kernels": during decode one KV head is shared by several query heads (after absorption MLA amounts to MQA), and the kernel computes the heads of one group in a single block, improving the KV's reuse. The four together (absorption, the grouped decode kernel, the FP8 bmm and the FP8 KV cache) make the blog's 3 to 7 times.

![Figure: MLA's two paths, decompressing the history's K or absorbing the decompression matrix into the query](../assets/figures/sgl-mla-absorb.svg){.aig-svg}

## torch.compile: a switch that already existed {#torchcompile早已存在的开关}

`--enable-torch-compile` was already in v0.2.0, hidden in `CudaGraphRunner`'s `use_torch_compile` parameter ([chapter nine](../service/v02.md)'s `compile_bs = [1, 2, 4, 8, 16, 24, 32]`). v0.3.0's implementation:

```python title="python/sglang/srt/model_executor/cuda_graph_runner.py @ v0.3.0 L58-76" linenums="58"
def patch_model(
    model: torch.nn.Module, enable_compile: bool, tp_group: "GroupCoordinator"
):
    backup_ca_comm = None

    try:
        if enable_compile:
            _to_torch(model)
            monkey_patch_vllm_all_gather()
            backup_ca_comm = tp_group.ca_comm
            tp_group.ca_comm = None
            yield torch.compile(model.forward, mode="max-autotune-no-cudagraphs")
        else:
            yield model.forward
    finally:
        if enable_compile:
            _to_torch(model, reverse=True)
            monkey_patch_vllm_all_gather(reverse=True)
            tp_group.ca_comm = backup_ca_comm
```

`patch_model` is a context manager: with compilation on it first calls `_to_torch(model)` — temporarily replacing the model's custom operators (vLLM's `RMSNorm`, `SiluAndMul` and other CUDA kernels) with pure PyTorch implementations so that `torch.compile` can see and fuse them; it turns off the custom all-reduce (whose communication the compiler cannot handle); and then `torch.compile(model.forward, mode="max-autotune-no-cudagraphs")`. The `no-cudagraphs` is the key: the compiler's own CUDA graphs are off, and SGLang's `CudaGraphRunner` captures the compiled forward pass from outside. On exit the operators are put back. Compilation happens only for the batch sizes in `compile_bs` (later `--torch-compile-max-bs`), because compiling one batch shape takes tens of seconds to minutes, and at a large batch the launch overhead does not matter anyway.

#993 of 8 August moved the compilation configuration into `cuda_graph_runner.py`, #1223 of 26 August added a CI throughput test for compilation, and #1306 of 2 September fixed the sampler's bugs under CUDA graphs and compilation — the blog's 1.5 times (batches of 1 to 32, Llama-3.1-8B) was polished into shape over that month. The blog also mentions being faster than gpt-fast at a batch of 1 while keeping continuous batching and RadixAttention.

```bash title="compile-commits.sh"
REF=${REF:-29f6d408c0}
git log --reverse --date=short --format='%ad  %h  %s' "$REF" | awk '$1 >= "2024-07-01" && $1 <= "2024-09-10"' | grep -iE 'torch.?compile|compile' | cut -c1-96
```

```text title="output"
2024-08-08  9f662501a3  Move torch.compile configs into cuda_graph_runner.py (#993)
2024-08-13  0076f11541  fix: use devel for Triton's compiler requirements (#1074)
2024-08-26  c61a1b6f97  Torch compile CI throughput test (#1223)
2024-09-02  a5a134f39f  Fix bugs in sampler with CUDA graph / torch.compile (#1306)
```

## Design trade-offs {#设计取舍}

- **MLA became a second pool rather than a change to the tree.** Separating "what a slot holds" from "how a slot is referenced" lets MLA reuse the prefix cache, the CUDA graphs and the scheduler at no cost. The later SWA pool, hybrid attention pool and NSA index pool all took this route.
- **The absorbing path and the decompressing path coexist.** The division between `forward_absorb` (decode) and `forward_normal` (prefill) is already there in v0.3.0, and later became a dynamic choice by the batch's extend length (the inference-systems handbook's [MLA chapter](serving://moe/mla/) covers both paths' arithmetic).
- **Compile only the ordinary operators.** Attention and sampling go to dedicated kernels while the compiler fuses the linear layers, the normalisations and the activations — each tool doing what it is good at. The price is the trick of temporarily swapping operators in `_to_torch`, and the monkey patches on vLLM's custom operators (cleaned up along with the dependency's removal).
- **Only on for small batches.** The compilation time is proportional to the number of batch shapes, and a large batch does not lack parallelism.

## What happened afterwards {#后来怎么样了}

- MLA: FlashInfer's MLA kernel was connected in 2024-12; FlashMLA, cutlass MLA (`sgl-kernel/csrc/attention/cutlass_mla_kernel.cu`) and the TRT-LLM MLA backend arrived in 2025-03; `--disable-mla`, `--enable-flashinfer-mla` and the other switches were deprecated in 2025-04 and unified into the choice of attention backend; DeepSeek-V3 and R1's EP deployment is [chapter 18](../scale/large-ep.md).
- torch.compile: 2025 added a `compilation/` directory and piecewise CUDA graphs (#11490) — cutting the forward pass into segments captured separately so that prefill's non-attention parts can be graphed too; `torch_compile_max_bs` is still the switch.
- `MLATokenToKVPool` is still in `mem_cache/memory_pool.py` today, with a dozen or so other pools beside it.

## Exercises {#练习}

**1. What absorption costs.** With DeepSeek-V2's configuration (`num_attention_heads=128`, `qk_nope_head_dim=128`, `kv_lora_rank=512`), work out how many multiply-accumulates one query takes on the absorbing path during decode, and how many the decompressing path takes over a history of 4096 tokens.

??? success "Answer"
    Absorbing: each head multiplies its 128-dimensional `q_nope` by `w_kc` (128x512) into 512 dimensions, so 128 heads is 128 x 128 x 512 ≈ 8.4M multiply-accumulates, independent of the history's length. Decompressing: each historical token's 512-dimensional latent vector is multiplied by `W_UK` back into 128 heads x 128 dimensions, so 4096 x 512 x 128 x 128 ≈ 34G. Four orders of magnitude apart, which is why decode must absorb.

**2. The two pools' interface.** Read v0.3.0's `BaseTokenToKVPool`, list the methods a subclass has to implement, and explain why `RadixAttention` does not need to know the pool's type.

??? success "A way to approach it"
    `get_key_buffer`, `get_value_buffer`, `get_kv_buffer`, `set_kv_buffer` and the rest read and write by layer and slot; `RadixAttention` only hands the `forward_batch` to the attention backend, which reads and writes according to the pool's type.

**3. What gets compiled.** Read v0.3.0's `_to_torch`, list the operator types temporarily replaced, and explain why.

??? success "A way to approach it"
    vLLM's custom CUDA operators `RMSNorm`, `SiluAndMul`, `GeluAndMul` and the like are swapped for their `forward_native` (pure PyTorch); the compiler can only fuse PyTorch operators it can trace, and a custom CUDA operator is a black box to it.

!!! interview "How to answer in an interview"
    "Why does MLA save KV, and how is it computed at inference?" — Start with the cache: one latent vector plus a short RoPE stretch per token, rather than a K and a V per head. Then absorption: move the decompression matrices to the query and the output sides so that decode computes attention in the latent space, independent of the history's length. Then the engineering: SGLang added a second memory pool and kernels supporting different q and v dimensions for it, while the radix tree and the scheduler were untouched. Mentioning #905 of 2024-08 and the several MLA backends it grew into shows you know it was built step by step.

## Summary {#小结}

- [x] #905 (2024-08-05): `MLATokenToKVPool`, Triton kernels supporting `kv_lora_rank` and a `DeepseekV2AttentionMLA` with weight absorption, plus the grouped decode kernel and the FP8 bmm, make up v0.3's 3 to 7 times.
- [x] The two-level design of the tree and the pool lets MLA reuse the prefix cache and the scheduling at no cost.
- [x] torch.compile went from a switch in v0.2 to a selling point in v0.3: compile the ordinary operators, leave attention and sampling to kernels, capture outside the CUDA graph, and turn it on for small batches only.
