# Model and weight loading

<p class="lead">With the layers in place, this chapter assembles them into a complete model: pull the fields the engine cares about out of Hugging Face's <code>config.json</code>, create the model by architecture name, then use a streaming loader that reads the weights out of safetensors while merging and sharding them on the way in. Four architectures are supported: Llama, Qwen2, Qwen3 and Qwen3-MoE.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. How do the decoder layers of Llama, Qwen2 and Qwen3 differ?
    2. Why does the weight loader stream? What is wrong with one `torch.load` of the whole checkpoint?
    3. `q_proj`, `k_proj` and `v_proj` are stored separately in the checkpoint. How are they merged into `qkv_proj` while loading? Does the order matter?
    4. Where does RoPE's `rope_theta` live in a transformers 4.x config, and where in a 5.x one?

??? success "Answers (try it yourself first, then expand)"
    1. The structure is the same and only three switches differ: Qwen2's q, k and v projections have biases; Qwen3 applies QK-Norm to q and k before RoPE (and has no qkv bias); Qwen3-MoE replaces the MLP with an MoE.
    2. One `torch.load` reads the whole checkpoint into memory, and the peak can be several times the model (the raw weights plus the sharded and merged copies); streaming shards and merges as it reads, so the peak stays small.
    3. Once q, k and v have each been read, they are sharded (under tensor parallelism) and then concatenated along the output dimension into one `qkv_proj` weight. The order matters: it must match the order in which the model splits `qkv_proj`'s output back into q, k and v, or they end up swapped.
    4. transformers 4.x puts it at the top level of the config (`rope_theta`); 5.x moves it into a `rope_parameters` dictionary. `ModelConfig` has to accept both.

**Files you will write**: `models/config.py`, `models/base.py`, `models/utils.py`, `models/decoder.py`, `models/register.py`, `models/weight.py`.

## ModelConfig: only the fields we need {#modelconfig只取需要的字段}

A Hugging Face config class has dozens of fields, and different models use different names and defaults. `ModelConfig` collapses the fields the engine cares about into one frozen dataclass:

@@code python/minisgl/models/config.py:ModelConfig@@

A few easy traps:

- **`head_dim`**: Qwen3's `head_dim` (128) is not `hidden_size / num_attention_heads` (1024 / 16 = 64), so the explicit field has to win;
- **RoPE parameters**: transformers 5 merged `rope_theta` and `rope_scaling` into a `rope_parameters` dictionary (`{"rope_theta": 1000000, "rope_type": "default"}`), while 4.x keeps them as two separate fields. `from_hf` accepts both;
- **Qwen2's biases**: Qwen2's q/k/v projections have biases, but its config has no `attention_bias` field, so it has to be decided from `model_type`.

@@code examples/ch03_models.py@@

@@output ch03_models@@

## The model structure {#模型结构}

The decoder layers of the four architectures are nearly identical, differing only in three switches:

| Architecture | q/k/v bias | QK-Norm | MLP |
| --- | --- | --- | --- |
| Llama | depends on `attention_bias` (usually none) | no | SwiGLU |
| Qwen2 | yes | no | SwiGLU |
| Qwen3 | no | yes | SwiGLU |
| Qwen3-MoE | no | yes | MoE (chapter 20) |

So we use one generic `decoder.py`:

@@code python/minisgl/models/decoder.py:CausalLM@@

@@code python/minisgl/models/decoder.py:Qwen3ForCausalLM@@

The sub-layers live in `models/utils.py`: `RopeAttn` (qkv_proj, attention layer, o_proj), `GatedMLP` (gate_up_proj, silu × up, down_proj) and `MoEMLP`.

@@code python/minisgl/models/utils.py:GatedMLP@@

`CausalLM.forward()` takes no arguments, reads `input_ids` from the global context, and returns `[batch_size, vocab]` logits, one row per request, which is what comes out of the LM head taking only the last position during prefill.

!!! diff "Difference from upstream: the model files"
    Upstream writes one file per architecture (`llama.py`, `qwen2.py`, `qwen3.py`, `qwen3_moe.py`, `mistral.py`, each 80-odd nearly identical lines) and lazy-loads them through a registry keyed on `architectures[0]`. We merge them into one `decoder.py` plus three switches, and do not implement Mistral.

## Streaming weight loading {#流式权重加载}

@@code python/minisgl/models/weight.py:load_weight@@

The loader is a generator that yields one fully processed `(name, tensor)` at a time:

1. **Read one at a time**: `safetensors.safe_open` pulls a single tensor from the file on demand, so the whole checkpoint never has to be in memory. A 70B bf16 model is 140 GB, and reading it at once would blow out CPU memory; streaming adds only one tensor plus a small merge buffer at the peak.
2. **Shard for TP**: `shard_tensor` takes this rank's slice (covered in chapter 16). Sharding before merging means the merged tensor is naturally this rank's `[q_local; k_local; v_local]`.
3. **Merge projections**: `q_proj` goes into a buffer when it is read, and once the same layer's `k_proj` and `v_proj` have also arrived they are concatenated **in the fixed q, k, v order** and yielded. That order must match how the attention layer splits them (`qkv.split([q_dim, kv_dim, kv_dim])`); the order in the file is alphabetical, which is why `down_proj` appears before the merged `gate_up_proj` in the output above, and has nothing to do with the concatenation order.
4. **Pack the experts**: an MoE model stores each expert separately (`experts.0.gate_proj`, `experts.1.gate_proj` and so on), and once all of them have been read `torch.stack` turns them into one `[E, ...]` three-dimensional tensor.

The two assertions at the end make sure no merge group was left half-read.

## Into the model {#装进模型}

The engine (chapter 6) loads in three lines:

```python
with torch.device("meta"), torch_dtype(config.dtype):   # an empty shell: shapes only, no memory
    self.model = create_model(config.model_config)
self.model.load_state_dict({k: v.to(self.dtype) for k, v in load_weight(path, device)})
```

`torch_dtype` temporarily changes the default dtype so that the `torch.empty(...)` calls during construction already use the target precision, which is what lets the dtype assertion in `load_state_dict` pass. `load_state_dict` replaces the meta tensors with the real ones from the checkpoint; Qwen3-0.6B's checkpoint carries an extra `lm_head.weight` (equal to `embed_tokens.weight` with tied embeddings) that `ParallelLMHead.load_state_dict` drops, while every other key must match exactly or it raises.

!!! upstream "The official implementation"
    - config: @@upstream models/config.py:ModelConfig.from_hf@@ (upstream pins `transformers<=4.57.3`; this book is verified on transformers 5.17)
    - the Qwen3 structure: @@upstream models/qwen3.py:Qwen3DecoderLayer@@
    - streaming loading: @@upstream models/weight.py:load_weight@@
    - sharding: @@upstream models/weight.py:_shard_tensor@@

## Tests {#测试}

@@code tests/test_ch03_models.py:test_weight_loader_produces_exactly_the_model_keys@@

This test makes sure the set of keys the loader produces is exactly the model's `state_dict` (apart from the dropped `lm_head.weight`), with matching shapes. `test_model_config_from_hf` checks the key fields of the Qwen3 and Qwen2.5 configs (Qwen2.5 only needs its `config.json`, kept in `tests/configs/`, so no model download is required).

!!! interview "Answering in an interview"
    On model loading: Llama, Qwen2, Qwen3 and Qwen3-MoE share one decoder structure and differ only in three switches, whether qkv carries a bias (Qwen2), whether there is QK-Norm (Qwen3), and whether the MLP is an MoE. The weights have to be loaded as a stream: reading the whole checkpoint at once doubles the peak memory, whereas streaming shards for tensor parallelism as it reads, concatenates `q_proj` / `k_proj` / `v_proj` into one `qkv_proj` in the same order the forward pass splits them, merges gate and up, and packs the MoE experts into a three-dimensional tensor. The config has to accept both the old and the new layout (transformers 5.x moved `rope_theta` into the RoPE parameters), and `head_dim` is not always `hidden / heads`.

## Exercises {#练习}

1. Write a function on top of `ModelConfig` that computes how many bytes of KV cache one token takes (the next chapter needs it). What is it for Qwen3-0.6B in bf16?
2. Add a `tqdm` progress bar to the loader, shown only on TP rank 0.
3. If a checkpoint put `q_proj` and `k_proj` in two different safetensors files, would our loader still merge them correctly? Why?

??? success "Answers"
    1. `2 * num_layers * num_kv_heads * head_dim * 2 bytes` = 2 × 28 × 8 × 128 × 2 = 114688 bytes, about 112 KB.
    2. `for file in tqdm(files, disable=not get_tp_info().is_primary())`, which is what upstream does.
    3. Yes. The `merge_buf` persists across files, so a group merges as soon as all its projections have been read; only a group still unfinished after every file has been read raises.

## Summary {#小结}

- [x] `ModelConfig` collapses the various HF configs into the fields the engine needs; watch out for `head_dim`, the old and new homes of the RoPE parameters, and Qwen2's biases.
- [x] Llama, Qwen2, Qwen3 and Qwen3-MoE share one decoder structure, differing in the qkv bias, QK-Norm and MoE switches.
- [x] The streaming loader shards and merges as it reads: a small memory peak, q/k/v and gate/up merged into one matmul each, and MoE experts packed into a three-dimensional tensor.
- [x] The model is built as an empty shell on the meta device in the target dtype, then `load_state_dict` swaps in the real weights.
