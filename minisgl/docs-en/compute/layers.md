# The op layer: BaseOP and the layers

<p class="lead">mini-sglang does not use <code>torch.nn.Module</code>. It writes its own 99-line <code>BaseOP</code> instead. This chapter works out why, then implements every layer a decoder is made of: linear layers, the embedding and output layers, RMSNorm, RoPE, the activation and the attention layer. The layers themselves are thin, and the real computing goes to the ops in <code>minisgl.kernel</code>, which have PyTorch reference implementations and switch to FlashInfer or custom kernels on a GPU.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Which parts of `nn.Module` does an inference engine need? Which does it not?
    2. Why is the model built on the `meta` device first? What problem does that create for RoPE's cos/sin table?
    3. What are the two tensors returned by the "fused residual add plus RMSNorm"?
    4. Why does the LM head only compute each request's last position during prefill?

??? success "Answers (try it yourself first, then expand)"
    1. Needed: organizing parameters hierarchically, and collecting and loading weights by name. Not needed: autograd, train / eval mode, hooks, the registration machinery behind `parameters()`, device moves through `to()`. Inference only runs forward, and the weights are loaded once and never change.
    2. Building on the meta device only allocates metadata, so it takes no memory and runs no initialization, and the real weights replace it afterwards, which makes building a model instant. But RoPE's cos / sin table is not a weight, and computing it on the meta device yields no data, so it has to be computed separately on a real device (one table shared by every layer).
    3. One is the normalized output that goes into the next sub-layer, the other is the new residual after the addition, kept for the next residual add.
    4. Only the logits at each request's last position are used to sample the next token; the logits elsewhere are useless. The LM head is one of the largest matmuls, because the vocabulary is big, so computing only the last position saves most of the compute and memory.

**Files you will write**: `layers/base.py`, `layers/linear.py`, `layers/embedding.py`, `layers/norm.py`, `layers/rotary.py`, `layers/activation.py`, `layers/attention.py`, plus the ops in `kernel/torch_ops.py` and `kernel/__init__.py`.

## Why not nn.Module {#为什么不用-nnmodule}

An inference engine uses `nn.Module` for exactly one thing: **collecting and loading weights by name**, where the names must match the keys in the checkpoint (`model.layers.0.self_attn.o_proj.weight`). Everything else in `nn.Module`, from `Parameter` and autograd to hooks, `train()`/`eval()` and the extra overhead of `__call__`, goes unused. `BaseOP` implements collecting and loading by name through a Python object's `__dict__` and has nothing else:

@@code python/minisgl/layers/base.py:BaseOP@@

There are only three rules:

1. a `Tensor` attribute that does not start with an underscore is a weight, and its name is the attribute path;
2. a `BaseOP` attribute that does not start with an underscore is a sub-module, handled recursively;
3. attributes starting with an underscore (RoPE's `_cos_sin_cache`, a linear layer's `_comm`) take no part.

`load_state_dict` does not copy data into existing tensors; it **replaces the attribute with the tensor from the checkpoint** (`setattr`). That fits "build the model on the meta device first": tensors created on `torch.device("meta")` have a shape but no memory, so building a 70B model is instant and costs nothing, and loading swaps in the real tensors. If any key in `state_dict` is left unclaimed at the end, it raises, because a leftover key usually means the model structure is wrong.

`OPList` names its sub-modules `0`, `1`, `2` and so on, matching `layers.0` and `layers.1` in the checkpoint; `StateLessOP` is for layers with no weights (RoPE, the attention layer itself), whose `state_dict` is always empty.

@@code python/minisgl/layers/base.py:OPList@@

Building one decoder layer on the meta device with the Qwen3-0.6B config, and looking at the weight names it collects:

@@code examples/ch02_layers.py@@

@@output ch02_layers@@

The names match the Hugging Face checkpoint except in two places: `q_proj`, `k_proj` and `v_proj` are merged into `qkv_proj`, and `gate_proj` and `up_proj` into `gate_up_proj`. Merging turns three matmuls into one and reads the input only once. The weight loader of the next chapter does the merging as it reads.

Note that `q_norm` is an attribute of both `RopeAttn` and the `AttentionLayer` inside it, yet appears only once: `AttentionLayer` is a `StateLessOP` and collects no weights, so it merely references the one `RopeAttn` holds.

## Ops: reference implementations and dispatch {#算子参考实现与分派}

Layers only organize parameters; the computing goes to `minisgl.kernel`. Every op has a PyTorch reference implementation whose order of operations deliberately matches Hugging Face, so that in float32 they agree bit for bit:

@@code python/minisgl/kernel/torch_ops.py:rmsnorm@@

@@code python/minisgl/kernel/torch_ops.py:fused_add_rmsnorm@@

`kernel/__init__.py` does the dispatch: FlashInfer for a CUDA tensor whose dtype is fp16 or bf16 when FlashInfer is installed, the reference implementation otherwise. The dtype condition is easy to miss: FlashInfer's kernels are compiled for half precision only, and fp32 raises `failed to dispatch data type` down in C++ — and fp32 is exactly what this book uses to verify correctness.

@@code python/minisgl/kernel/__init__.py:rmsnorm@@

!!! diff "Difference from upstream: op dispatch"
    The official layers do `from flashinfer import rmsnorm` and friends directly, so they only run on a GPU. We have every layer call its ops through `minisgl.kernel`, with the reference implementation on a CPU. The last two lines of the example above show that the reference RMSNorm and RoPE differ from Hugging Face by exactly 0.

## Linear layers {#线性层}

@@code python/minisgl/layers/linear.py:_LinearTPImpl@@

`_LinearTPImpl` records both the full and this rank's input and output dimensions, and `forward` is one `F.linear`. Four subclasses cover the four ways to shard under tensor parallelism:

| Class | Used for | How TP shards it |
| --- | --- | --- |
| `LinearReplicated` | the MoE router | not sharded, a full copy per rank |
| `LinearColParallelMerged` | `gate_up_proj` | split along the output dimension, several projections merged |
| `LinearQKVMerged` | `qkv_proj` | split by attention head, with KV heads replicated when there are not enough |
| `LinearRowParallel` / `LinearOProj` | `down_proj`, `o_proj` | split along the input dimension, result all-reduced |

At TP=1 they all degenerate into ordinary linear layers. The sharding details and the communication are in the [tensor parallelism](../perf/tensor-parallel.md) chapter; for now just remember that the `get_tp_info().size` appearing in the constructors is 1 on a single GPU.

## Embedding and output layers {#词嵌入与输出层}

@@code python/minisgl/layers/embedding.py:VocabParallelEmbedding@@

The lookup goes to the `indexing` op. Under tensor parallelism each rank keeps only a slice of the vocabulary, `vocab_range = (start, length)`, fills zeros for tokens it does not hold, and all-reduces at the end, since every token finds its non-zero vector on exactly one rank.

@@code python/minisgl/layers/embedding.py:ParallelLMHead.forward@@

The output layer has one important optimization: **during prefill it takes only the hidden state at each request's last position** to compute logits. For a 1000-token prompt, prefill only needs the next-token distribution at position 1000; computing every position would waste a thousandfold on the LM head's compute and memory (`[1000, 1024] × [1024, 151936]`). The index of the last position comes from `get_last_indices` in the attention metadata (chapter 5). During decode each request has only one position anyway, so there is nothing to pick.

Qwen3-0.6B ties its embeddings (`tie_word_embeddings`), so `ParallelLMHead` uses `embed_tokens`'s weight directly and its own `state_dict` is empty; if the checkpoint still carries `lm_head.weight` during loading, as Qwen3-0.6B does, it is dropped.

## RMSNorm and the residual {#rmsnorm-与残差}

@@code python/minisgl/layers/norm.py:RMSNormFused@@

In a decoder layer the residual add is always followed immediately by a normalization: `residual = x + residual; x = norm(residual)`. `RMSNormFused` merges the two into one op that returns both the normalized result and the new residual; on a GPU it is a single FlashInfer kernel, saving one read and write of the hidden state. In the first layer, where there is no residual yet, `x` is the residual. So a whole decoder layer reads like this:

@@code python/minisgl/models/decoder.py:DecoderLayer.forward@@

## RoPE {#rope}

@@code python/minisgl/layers/rotary.py:RotaryEmbedding@@

`cos_sin_cache` is computed once for every position at construction time, and the forward pass looks it up by `positions`. The op rotates in the NeoX style, pairing the two halves, which matches the Hugging Face implementation for the Llama and Qwen families:

@@code python/minisgl/kernel/torch_ops.py:apply_rope_inplace@@

`get_rope` caches with `functools.cache`: every layer has the same RoPE parameters and shares one table. One more detail: the model is built on `torch.device("meta")`, while the cos/sin table is real data and cannot live there. So when `get_rope` sees that the current default device is meta, it uses the real device named earlier through `set_rope_device`:

@@code python/minisgl/layers/rotary.py:get_rope@@

Forget to call `set_rope_device` before building on meta and you get `RuntimeError: Call set_rope_device() before building a model on meta device`.

Llama 3.1's long-context extension (`rope_type: llama3`) post-processes the frequencies: high-frequency components are left alone, low-frequency ones are divided by `factor`, and the middle transitions smoothly. Its correctness is verified in chapter 20 against Hugging Face using a small Llama 3 model built from random weights.

!!! diff "Difference from upstream: YaRN"
    The official `_get_rope` also supports `yarn`, but it only changes the frequencies and never multiplies by the attention scaling factor from the Hugging Face implementation (`0.1 * ln(factor) + 1`). We implement only `default` and `llama3` and leave YaRN as an exercise.

## The attention layer {#注意力层}

@@code python/minisgl/layers/attention.py:AttentionLayer.forward@@

The attention layer holds no weights either: it splits the merged `qkv` into three, applies an RMSNorm per head to q and k for Qwen3 (QK-Norm), adds RoPE, and hands q, k and v to the **attention backend** in the global context. Writing the KV cache and computing attention are the backend's job, implemented in chapter 5.

!!! upstream "The official implementation"
    - `BaseOP`: @@upstream layers/base.py:BaseOP@@
    - the output layer taking only the last position: @@upstream layers/embedding.py:ParallelLMHead.forward@@
    - RoPE's meta-device handling: @@upstream layers/rotary.py:get_rope@@
    - the attention layer: @@upstream layers/attention.py:AttentionLayer.forward@@

    Upstream applies QK-Norm and RoPE in place on the views from `qkv.split(...)`, since FlashInfer's ops accept strided inputs. Our reference ops make a `.contiguous()` copy first, which is more direct to read at the price of one extra copy.

## Tests {#测试}

`tests/test_ch02_layers.py` checks `BaseOP`'s naming rules, the error on leftover keys, RMSNorm and RoPE matching Hugging Face, and the vocabulary-parallel lookup:

@@code tests/test_ch02_layers.py:test_vocab_parallel_indexing_masks_other_shards@@

!!! interview "How to explain it"
    On the op layer: an inference engine needs only a small part of `nn.Module`, namely collecting and loading weights by name and organizing the hierarchy; it needs no autograd, no hooks and no training state, which is why mini-sglang uses the lightweight `BaseOP`, where layers only organize parameters and the computing goes to the kernels (a reference implementation that matches HF numerically on a CPU, FlashInfer on a GPU). The model is built as an empty shell on the `meta` device, which allocates no memory and takes no time, and loading swaps in the real weights; RoPE's cos/sin table, however, is computed rather than loaded and must live on a real device. The fused residual add plus RMSNorm returns the normalized output and the new residual. During prefill the LM head computes only each request's last position, which saves an enormous amount of useless vocabulary projection.

## Exercises {#练习}

1. Add a `num_params()` method to `BaseOP` that returns the total number of weight elements. Use it to compute Qwen3-0.6B's parameter count (careful not to double-count the tied embeddings).
2. Remove the "prefill takes only the last position" logic from `ParallelLMHead.forward` and prefill a 1000-token request. How large is the LM head's output tensor in float32?
3. Implement YaRN: add a branch to `_get_rope` and multiply cos/sin by the attention scaling factor. Verify it against a small model built with `transformers`'s `Qwen3Config(rope_scaling={"rope_type": "yarn", ...})`.

??? success "Answers"
    1. `sum(t.numel() for t in self.state_dict().values())`. With tied embeddings `ParallelLMHead.state_dict()` is empty, so nothing is double-counted: Qwen3-0.6B comes to about 596 million.
    2. `[1000, 151936]` float32 values, about 580 MiB; taking only the last position makes it 0.6 MB.
    3. Follow Hugging Face's `_compute_yarn_parameters`: the frequencies interpolate between `inv_freq / factor` and `inv_freq` along a `ramp`, and `attention_factor = 0.1 * ln(factor) + 1` multiplies the cos/sin table as it is built.

## Summary {#小结}

- [x] `BaseOP` collects and loads weights by attribute path through `__dict__`; attributes starting with an underscore are not weights; loading replaces rather than copies, which is what makes meta-device construction instant.
- [x] Layers only organize parameters and hand the computing to `minisgl.kernel`: a reference implementation that matches HF numerically on a CPU, FlashInfer on a GPU.
- [x] During prefill the output layer computes logits only at each request's last position.
- [x] The RoPE table must live on a real device, and every layer shares one table.
- [x] The attention layer only splits, applies QK-Norm and RoPE, and leaves the cache writes and the attention itself to the attention backend.
