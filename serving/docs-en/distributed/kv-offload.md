# Tiered KV caching and offloading

<p class="lead">GPU memory holds only so much KV Cache, which caps the prefix cache's hit rate: a user in a multi-turn chat comes back after a few minutes, or an agent calls the same long context again and again, and by then those prefixes were evicted long ago and must be recomputed. Yet a server has terabytes of CPU memory and tens of terabytes of SSD. Tiered KV caching stores KV evicted from the GPU in cheaper, larger storage and reads it back on a hit. This chapter first works out how much faster "reading back" is than "recomputing", then adds a CPU cache tier to the mini engine and checks the effect on multi-turn conversations.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. Which is faster, reading KV back from CPU memory or prefilling again? By how much? How does it depend on the model's architecture?
    2. In a tiered cache, when is KV written to the next tier? When is it read back?
    3. What mechanisms do vLLM and SGLang use to support KV offloading?

??? success "Answers (try first, then expand to compare)"
    1. Reading back is much faster: for Qwen2.5-7B, recomputing takes about 31 μs per token, reading back from CPU memory about 1.2 μs (about 27 times faster), and from a local NVMe SSD about 9.6 μs. It depends on architecture: MLA's KV is small and its compute per token large, so reading back wins by more (DeepSeek-V3: 152 μs to recompute, 1.4 μs to read back from memory).
    2. It is written to the next tier when evicted from the GPU cache (CPU memory, and below that SSD or distributed storage); it is read back when a new request's prefix misses on the GPU but hits in the next tier.
    3. vLLM uses the KV connector interface (native CPU offloading, or LMCache), sharing one interface with PD disaggregation; SGLang implements tiered caching with HiCache.

## Read back or recompute {#读回来还是重算}

Recomputing one token's KV takes a full forward pass (about 2 × parameter count floating-point operations); reading one token's KV back only moves "KV size per token" bytes. Estimate:

```python
gpu_flops, mfu = 989e12, 0.5                            # H100 BF16 peak and prefill utilization
tiers = {"CPU 内存（PCIe 5.0，约 50 GB/s）": 50e9, "本地 NVMe SSD（约 6 GB/s）": 6e9}
models = {"Qwen2.5-7B（GQA）": (7.6e9, 56 * 1024), "LLaMA-3-70B（GQA，8 卡 TP）": (70.6e9 / 8, 320 * 1024 / 8),
          "DeepSeek-V3（MLA，激活 37B，按单卡算力折算）": (37.6e9, 70272)}
for name, (params, kv_bytes) in models.items():
    recompute_us = 2 * params / (gpu_flops * mfu) * 1e6
    loads = "，".join(f"{tier.split('（')[0]}读回 {kv_bytes / bw * 1e6:.2f} μs" for tier, bw in tiers.items())
    print(f"{name}：每 token 重算 {recompute_us:.1f} μs；{loads}")
```

```text title="output"
Qwen2.5-7B（GQA）：每 token 重算 30.7 μs；CPU 内存读回 1.15 μs，本地 NVMe SSD读回 9.56 μs
LLaMA-3-70B（GQA，8 卡 TP）：每 token 重算 35.7 μs；CPU 内存读回 0.82 μs，本地 NVMe SSD读回 6.83 μs
DeepSeek-V3（MLA，激活 37B，按单卡算力折算）：每 token 重算 152.1 μs；CPU 内存读回 1.41 μs，本地 NVMe SSD读回 11.71 μs
```

(This ignores attention's own compute, which grows with the context, so recomputing really costs even more.) Even reading back from SSD is several to a dozen-odd times faster than recomputing; from CPU memory it is tens to over a hundred times faster. MLA models, with small KV per token and heavy compute, benefit most from reading back. So whenever it hits, a tiered cache almost always pays off; the keys are **capacity** and **not letting reads block compute**.

## The tiers {#分层结构}

<!-- i18n:diagram 27b227cdff -->
```text
L1  GPU memory (tens of GB)            ← KV in use + the hottest prefix cache
 │   written to the next tier on eviction (or earlier: write-through)
L2  CPU memory (hundreds of GB to TB)  ← prefix cache shared within the machine
 │
L3  SSD / distributed storage (TB–PB)  ← shared across machines and instances (Mooncake Store, 3FS, LMCache, etc.)
```

![Figure: tiers of the KV Cache: the lower, the larger the capacity and the lower the bandwidth](../assets/figures/kv-tiers.svg){.aig-svg}

Design points:

- **When to write down**: write-back (write only on eviction, this chapter's approach) writes the least; write-through (write a copy asynchronously once computed) needs no waiting on eviction and makes sharing with other instances easier;
- **When to read up**: start reading as soon as a request arrives and a hit is found, overlapping with queueing and with other requests' compute (prefetching); load layer by layer, so compute can start once the first layer is read;
- **Granularity and indexing**: reuse the prefix cache's block hashes as global keys, so every tier and every instance can find the same KV with the same key. That vLLM's block hashes can be used for this directly is one of the strengths of its design (see [prefix caching](../engine/prefix-cache.md#两种思路)).

## Adding a CPU cache tier to the mini engine {#给迷你引擎加一层-cpu-缓存}

Add a tier on top of the previous chapter's `PrefixCachingBlockPool`: before a block on the GPU is reused (evicted), store its contents in CPU memory by hash; on lookup, if the GPU misses but the CPU hits, allocate a GPU block, copy the contents back, and treat it as a hit:

```python title="tiered.py"
"""tiered.py —— 两级前缀缓存：GPU 上的块被驱逐时，把内容存到 CPU 内存；之后命中时再拷回来，而不是重算。"""

from collections import OrderedDict

from prefix_cache import PrefixCachingBlockPool


class TieredPrefixCachingBlockPool(PrefixCachingBlockPool):
    def __init__(self, num_blocks: int, block_size: int, host_capacity_blocks: int):
        super().__init__(num_blocks, block_size)
        self.kv = None                                   # bound after the engine creates the KV tensors
        self.host: OrderedDict[int, list] = OrderedDict()  # block hash -> per-layer (K, V), in LRU order
        self.host_capacity = host_capacity_blocks
        self.num_offloaded = self.num_loaded = 0

    def allocate(self, n: int):
        if n > len(self.free_queue):
            return None
        for b in list(self.free_queue)[:n]:              # a block about to be reused: if it carries a hash, offload it to CPU first
            h = self.block_hash[b]
            if h is not None and h not in self.host:
                self.host[h] = [(self.kv.k[l][b].clone(), self.kv.v[l][b].clone()) for l in range(len(self.kv.k))]
                self.num_offloaded += 1
                if len(self.host) > self.host_capacity:
                    self.host.popitem(last=False)
        return super().allocate(n)

    def lookup(self, token_ids: list[int]) -> list[int]:
        plan = []                                        # each hit block: (hash, GPU block id, or None if it is on CPU)
        for h in self.block_hashes(token_ids):
            b = self.hash_to_block.get(h)
            if b is None and h not in self.host:
                break
            plan.append((h, b))
        # first pull GPU hits that sit in the free queue out of it, so allocating new blocks for CPU hits below cannot reuse them
        protected = [b for _, b in plan if b is not None and self.ref_cnt[b] == 0]
        for b in protected:
            del self.free_queue[b]
        need = sum(b is None for _, b in plan)
        new = self.allocate(need) if need else []
        if new is None:                                  # no room: keep only the contiguous run of GPU hits
            new = []
            plan = plan[:next((i for i, (_, b) in enumerate(plan) if b is None), len(plan))]
        loaded = iter(new)
        hit = []
        for h, b in plan:
            if b is None:                                # CPU hit: copy back into the newly allocated GPU block
                b = next(loaded)
                for l, (k, v) in enumerate(self.host[h]):
                    self.kv.k[l][b].copy_(k)
                    self.kv.v[l][b].copy_(v)
                self.host.move_to_end(h)
                self.block_hash[b], self.hash_to_block[h] = h, b
                self.ref_cnt[b] = 0                      # register it as "free but cached"; the scheduler's later touch will reference it
                self.free_queue[b] = None
                self.num_loaded += 1
            hit.append(b)
        for b in protected:
            self.free_queue[b] = None
        self.num_queries += len(token_ids)
        self.num_hits += len(hit) * self.block_size
        return hit
```

One detail deserves attention: copying a block back from CPU needs a new GPU block, and allocating may reuse a block from the free queue. If the reused block happens to be **a block this lookup already hit on the GPU** (its reference count is still 0), the prefix just found gets overwritten. So the lookup must first pull those blocks out of the free queue to protect them, and put them back after copying. The first version of the implementation missed this, and its output did not match the reference; this kind of bug only shows up when comparing with the reference token by token.

The experiment: open a conversation about each of 6 cities, 3 turns each; each turn processes all conversations together, and the next turn's prompt is the full history. The GPU has only 24 KV blocks, nowhere near enough for 6 conversations, so the previous turn's prefixes are evicted before the next turn arrives:

```python
import torch
from transformers import AutoTokenizer
from mini_llm import Transformer
from nano_engine import LLMEngine, SamplingParams
from prefix_cache import PrefixCachingBlockPool
from tiered import TieredPrefixCachingBlockPool

torch.set_num_threads(16)
path = "models/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(path)
model = Transformer.from_pretrained(path)
cities = ["杭州", "成都", "西安", "广州", "南京", "武汉"]
followups = ["它有什么特色美食？", "适合几月去旅游？"]

def multi_turn(pool):
    engine = LLMEngine(model, eos_token_id=tok.eos_token_id, pool=pool, num_blocks=24)
    if isinstance(pool, TieredPrefixCachingBlockPool):
        pool.kv = engine.kv
    histories = [[{"role": "user", "content": f"用两句话介绍{c}。"}] for c in cities]
    outputs = []
    for turn in range(3):
        prompts = [tok(tok.apply_chat_template(h, tokenize=False, add_generation_prompt=True, enable_thinking=False)).input_ids
                   for h in histories]
        replies = engine.generate(prompts, SamplingParams(max_tokens=20))
        outputs.append(replies)
        for h, r in zip(histories, replies):
            h.append({"role": "assistant", "content": tok.decode(r, skip_special_tokens=True)})
            if turn < 2:
                h.append({"role": "user", "content": followups[turn]})
    return outputs, sum(s["num_tokens"] for s in engine.step_log)

reference, computed = multi_turn(None)
print(f"不缓存：            共计算 {computed} 个 token")
gpu_only, computed = multi_turn(PrefixCachingBlockPool(24, 16))
print(f"只有 GPU 前缀缓存：  共计算 {computed} 个 token，输出一致：{gpu_only == reference}")
pool = TieredPrefixCachingBlockPool(24, 16, host_capacity_blocks=200)
tiered, computed = multi_turn(pool)
print(f"GPU + CPU 两级缓存：共计算 {computed} 个 token，输出一致：{tiered == reference}，"
      f"卸载 {pool.num_offloaded} 个块，读回 {pool.num_loaded} 个块")
assert gpu_only == reference and tiered == reference
```

```text title="output"
不缓存：            共计算 1512 个 token
只有 GPU 前缀缓存：  共计算 1400 个 token，输出一致：True
GPU + CPU 两级缓存：共计算 1049 个 token，输出一致：True，卸载 47 个块，读回 22 个块
```

With too little GPU capacity, the prefix cache saves only 7% of the compute; with the CPU tier added, history pushed out of the GPU can all be read back, the tokens to compute drop by about 30%, and the output is exactly the same.

The saving is smaller than one might expect for another reason, unrelated to cache capacity: when Qwen3's chat template renders past turns, it removes the empty `<think>\n\n</think>\n\n` in front of the assistant's answer. The sequence actually computed in the previous turn was "...assistant\n<think>\n\n</think>\n\nanswer", but the history rendered this turn is "...assistant\nanswer", so the prefix stops matching right after `assistant\n`, and **the answer generated in the previous turn has to be recomputed every turn**; only the earlier history can be reused. This is a very common prefix-cache killer in multi-turn conversations with thinking models: either have the template keep the thinking markers in the history (some models offer such a switch), or account for it when estimating the hit rate.

!!! source "Source code"
    - **vLLM**: `--kv-offloading-size` (space on CPU for KV, in GiB) turns on offloading, and `--kv-offloading-backend` picks `native` (vLLM's built-in CPU offloading) or `lmcache`. The native implementation is in `vllm/v1/kv_offload/`, hooked into the scheduler and workers through the KV connector interface (`offloading_connector.py`), sharing the same mechanism as PD disaggregation: to the scheduler, reading KV back from CPU is no different from receiving KV from a prefill instance; both are "some tokens' KV can be loaded from outside" (`get_num_new_matched_tokens`).
    - **SGLang**: HiCache (`--enable-hierarchical-cache`) uses `HiRadixCache` (`srt/mem_cache/hiradix_cache.py`) to record on radix tree nodes which tier the KV is in; `--hicache-ratio` / `--hicache-size` set the size of the host memory tier, and `--hicache-storage-backend` picks the third tier (`srt/mem_cache/storage/` has implementations for file, mooncake_store, hf3fs, lmcache and others). `HiCacheController` moves data between GPU and host asynchronously on a separate CUDA stream.

!!! interview "How to explain it"
    To explain "what if the KV Cache doesn't fit", go in layers: **reduce KV** (GQA/MLA, KV quantization, sliding windows) → **manage KV on the GPU better** (paging, prefix caching, preemption) → **expand capacity** (tiered caching: CPU memory, SSD, distributed KV storage) → **spread over more GPUs** (TP, context parallelism). When you get to tiered caching, use this chapter's estimates to show "reading back is tens of times faster than recomputing", then name the engineering difficulties: asynchronous transfers overlapping with compute, layer-by-layer loading, globally consistent block hashes, and consistency and eviction policy when several instances share it.

## Exercises {#练习}

**1. Is PCIe bandwidth enough?** An 8-GPU server with PCIe 5.0 x16 at about 50 GB/s per GPU. If a service has 20 requests per second that each hit 32K tokens in the CPU cache (Qwen2.5-7B, TP=1), is PCIe bandwidth enough?

??? success "Answer"
    Each request reads back 32768 × 56 KB ≈ 1.9 GB, so 20 per second is 38 GB/s. Spread evenly over 8 GPUs, that is about 4.7 GB/s per GPU, far below 50 GB/s and plenty; concentrated on one GPU, it approaches the limit. Also watch the bandwidth of CPU memory itself, and contention when several GPUs share a PCIe switch.

**2. To offload or not?** Someone suggests: since reading back is so fast, write every request's KV to CPU in real time (write-through), so the GPU can evict more aggressively. What does this cost?

??? success "Approach"
    Write-through keeps occupying PCIe bandwidth (every generated token writes a copy of its KV), competing with reads and with PD-disaggregation transfers; CPU memory is also finite, and most of the KV written there may never be hit again. A more sensible approach is to write by value: write only what is likely to be reused (for example full blocks, multi-turn history, system prompts), or write only on eviction (write-back), and decide retention by access frequency.

## Summary {#小结}

- [x] Reading one token's KV back is several times (SSD) to tens or hundreds of times (CPU memory) faster than recomputing, more so for MLA models.
- [x] Tiered caching: GPU → CPU memory → SSD / distributed storage; block hashes are the global key across tiers and instances.
- [x] Write to the next tier on eviction, read back on a hit; the implementation must protect blocks this lookup has already hit, and should overlap transfers with compute.
- [x] vLLM implements it through the KV connector (native offloading or LMCache), sharing the interface with PD disaggregation; SGLang implements it with HiCache.
