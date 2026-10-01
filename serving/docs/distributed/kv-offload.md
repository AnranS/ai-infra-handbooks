# KV 分层缓存与卸载

<p class="lead">GPU 显存能装下的 KV Cache 有限，前缀缓存的命中率也因此受限：多轮对话的用户隔几分钟回来，Agent 反复调用同一段长上下文，这些前缀早就被淘汰了，只能重算。而一台服务器的 CPU 内存有 TB 级，SSD 有几十 TB。KV 分层缓存把 GPU 上被淘汰的 KV 存到更便宜、更大的存储里，命中时再读回来。这一章先算清楚"读回来"比"重算"快多少，再给迷你引擎加一层 CPU 缓存，在多轮对话上验证效果。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 从 CPU 内存读回 KV 与重新 prefill，哪个快？快多少？和模型结构有什么关系？
    2. 分层缓存中，KV 什么时候写到下一层？什么时候读回来？
    3. vLLM 和 SGLang 分别用什么机制支持 KV 卸载？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 读回快得多：Qwen2.5-7B 每 token 重算约 31 μs，从 CPU 内存读回约 1.2 μs（快约 27 倍），从本地 NVMe SSD 读回约 9.6 μs。和模型结构有关：MLA 的 KV 小、每 token 的计算量大，读回的优势更大（DeepSeek-V3 重算 152 μs、从内存读回 1.4 μs）。
    2. 被 GPU 上的缓存淘汰时写到下一层（CPU 内存，再往下是 SSD 或分布式存储）；新请求的前缀在 GPU 上没命中、在下一层命中时读回来。
    3. vLLM 通过 KV connector 接口（原生的 CPU 卸载，或者接 LMCache），与 PD 分离共用同一套接口；SGLang 用 HiCache 实现分层缓存。

## 读回来还是重算

重算一个 token 的 KV，要做一次完整的前向（大约 2 × 参数量次浮点运算）；读回一个 token 的 KV，只需要搬运"每 token KV 大小"那么多字节。估算一下：

```python
gpu_flops, mfu = 989e12, 0.5                            # H100 BF16 峰值与 prefill 的利用率
tiers = {"CPU 内存（PCIe 5.0，约 50 GB/s）": 50e9, "本地 NVMe SSD（约 6 GB/s）": 6e9}
models = {"Qwen2.5-7B（GQA）": (7.6e9, 56 * 1024), "LLaMA-3-70B（GQA，8 卡 TP）": (70.6e9 / 8, 320 * 1024 / 8),
          "DeepSeek-V3（MLA，激活 37B，按单卡算力折算）": (37.6e9, 70272)}
for name, (params, kv_bytes) in models.items():
    recompute_us = 2 * params / (gpu_flops * mfu) * 1e6
    loads = "，".join(f"{tier.split('（')[0]}读回 {kv_bytes / bw * 1e6:.2f} μs" for tier, bw in tiers.items())
    print(f"{name}：每 token 重算 {recompute_us:.1f} μs；{loads}")
```

```text title="输出"
Qwen2.5-7B（GQA）：每 token 重算 30.7 μs；CPU 内存读回 1.15 μs，本地 NVMe SSD读回 9.56 μs
LLaMA-3-70B（GQA，8 卡 TP）：每 token 重算 35.7 μs；CPU 内存读回 0.82 μs，本地 NVMe SSD读回 6.83 μs
DeepSeek-V3（MLA，激活 37B，按单卡算力折算）：每 token 重算 152.1 μs；CPU 内存读回 1.41 μs，本地 NVMe SSD读回 11.71 μs
```

（这里忽略了注意力本身随上下文增长的计算量，重算的真实代价只会更高。）即使从 SSD 读回，也比重算快几倍到十几倍；从 CPU 内存读回快几十倍到上百倍。MLA 模型每 token 的 KV 很小、计算量又大，读回的优势最明显。所以只要命中，分层缓存几乎总是划算的，关键在于**容量**和**别让读取挡住计算**。

## 分层结构

```text
L1  GPU 显存（几十 GB）      ← 正在使用的 KV + 最热的前缀缓存
 │   被淘汰时写到下一层（或提前写：write-through）
L2  CPU 内存（几百 GB～TB）   ← 本机共享的前缀缓存
 │
L3  SSD / 分布式存储（TB～PB） ← 跨机器、跨实例共享（Mooncake Store、3FS、LMCache 等）
```

![图：KV Cache 的分层——越往下容量越大、带宽越低](../assets/figures/kv-tiers.svg){.aig-svg}

设计要点：

- **何时写下去**：写回式（被淘汰时才写，本章的做法）写入量最少；写穿式（算完就异步写一份）在淘汰时不需要等待，也便于其他实例共享；
- **何时读上来**：请求到达、查到命中时就开始读，与排队、与其他请求的计算重叠（预取）；逐层加载，第一层读完就可以开始计算；
- **粒度与索引**：沿用前缀缓存的块哈希作为全局的键，各层之间、各实例之间都能用同一个键找到同一段 KV。vLLM 的块哈希能直接用于这个目的，正是它的设计优点之一（见[前缀缓存](../engine/prefix-cache.md#两种思路)）。

## 给迷你引擎加一层 CPU 缓存

在上一章的 `PrefixCachingBlockPool` 基础上加一层：块在 GPU 上被复用（驱逐）之前，把它的内容按哈希存进 CPU 内存；查找时 GPU 未命中但 CPU 命中，就分配一个 GPU 块，把内容拷回来，当作命中处理：

```python title="tiered.py"
"""tiered.py —— 两级前缀缓存：GPU 上的块被驱逐时，把内容存到 CPU 内存；之后命中时再拷回来，而不是重算。"""

from collections import OrderedDict

from prefix_cache import PrefixCachingBlockPool


class TieredPrefixCachingBlockPool(PrefixCachingBlockPool):
    def __init__(self, num_blocks: int, block_size: int, host_capacity_blocks: int):
        super().__init__(num_blocks, block_size)
        self.kv = None                                   # 引擎创建好 KV 张量后绑定
        self.host: OrderedDict[int, list] = OrderedDict()  # 块哈希 -> 各层的 (K, V)，按 LRU 排列
        self.host_capacity = host_capacity_blocks
        self.num_offloaded = self.num_loaded = 0

    def allocate(self, n: int):
        if n > len(self.free_queue):
            return None
        for b in list(self.free_queue)[:n]:              # 即将被复用的块：如果带着哈希，先卸载到 CPU
            h = self.block_hash[b]
            if h is not None and h not in self.host:
                self.host[h] = [(self.kv.k[l][b].clone(), self.kv.v[l][b].clone()) for l in range(len(self.kv.k))]
                self.num_offloaded += 1
                if len(self.host) > self.host_capacity:
                    self.host.popitem(last=False)
        return super().allocate(n)

    def lookup(self, token_ids: list[int]) -> list[int]:
        plan = []                                        # 每个命中块：(哈希, GPU 块号或 None 表示在 CPU 上)
        for h in self.block_hashes(token_ids):
            b = self.hash_to_block.get(h)
            if b is None and h not in self.host:
                break
            plan.append((h, b))
        # 先把 GPU 上命中、但处于空闲队列中的块摘出来，免得下面为 CPU 命中分配新块时把它们复用掉
        protected = [b for _, b in plan if b is not None and self.ref_cnt[b] == 0]
        for b in protected:
            del self.free_queue[b]
        need = sum(b is None for _, b in plan)
        new = self.allocate(need) if need else []
        if new is None:                                  # 放不下：只保留 GPU 上连续命中的部分
            new = []
            plan = plan[:next((i for i, (_, b) in enumerate(plan) if b is None), len(plan))]
        loaded = iter(new)
        hit = []
        for h, b in plan:
            if b is None:                                # CPU 命中：拷回刚分配的 GPU 块
                b = next(loaded)
                for l, (k, v) in enumerate(self.host[h]):
                    self.kv.k[l][b].copy_(k)
                    self.kv.v[l][b].copy_(v)
                self.host.move_to_end(h)
                self.block_hash[b], self.hash_to_block[h] = h, b
                self.ref_cnt[b] = 0                      # 以"空闲但已缓存"的状态登记，调度器随后的 touch 会引用它
                self.free_queue[b] = None
                self.num_loaded += 1
            hit.append(b)
        for b in protected:
            self.free_queue[b] = None
        self.num_queries += len(token_ids)
        self.num_hits += len(hit) * self.block_size
        return hit
```

一个细节值得注意：从 CPU 拷回一个块时需要分配新的 GPU 块，而分配可能复用空闲队列里的块。如果被复用的恰好是**本次查找已经在 GPU 上命中的块**（它们的引用计数还是 0），就会把刚找到的前缀覆盖掉。所以查找时要先把这些块从空闲队列里摘出来，保护起来，拷完再放回去。第一版实现漏掉了这一点，输出与参考结果不一致；这类问题只有与参考结果逐 token 比较才能发现。

实验：6 个城市各开一段对话，每段 3 轮；每一轮所有对话一起处理，下一轮的提示词是完整的历史。GPU 上只有 24 个 KV 块，远远放不下 6 段对话，前一轮的前缀在下一轮到来之前就会被驱逐：

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

```text title="输出"
不缓存：            共计算 1512 个 token
只有 GPU 前缀缓存：  共计算 1400 个 token，输出一致：True
GPU + CPU 两级缓存：共计算 1049 个 token，输出一致：True，卸载 47 个块，读回 22 个块
```

GPU 容量太小时，前缀缓存只能省下 7% 的计算；加上 CPU 这一层，被挤出 GPU 的历史都能读回来，需要计算的 token 少了约 30%，输出完全不变。

省下的比例没有想象中多，还有一个和缓存容量无关的原因：Qwen3 的对话模板在渲染历史轮次时，会删掉助手回答前面那段空的 `<think>\n\n</think>\n\n`。上一轮实际算过的序列是"……assistant\n<think>\n\n</think>\n\n回答"，这一轮渲染出来的历史却是"……assistant\n回答"，前缀在 `assistant\n` 之后就对不上了，**上一轮生成的回答每轮都要重算**，能复用的只有更早的历史。思考模型的多轮对话里这是很常见的前缀缓存杀手：要么让模板在历史里保留思考标记（有的模型提供了这样的开关），要么在估算命中率时把它算进去。

!!! source "源码对照"
    - **vLLM**：`--kv-offloading-size`（CPU 上用于 KV 的空间，GiB）开启卸载，`--kv-offloading-backend` 选择 `native`（vLLM 自带的 CPU 卸载）或 `lmcache`。原生实现在 `vllm/v1/kv_offload/`，通过 KV connector 接口（`offloading_connector.py`）接入调度器和 worker，与 PD 分离共用同一套机制：对调度器来说，从 CPU 读回 KV 和从 prefill 实例收到 KV 没有区别，都是"有一部分 token 的 KV 可以从外部加载"（`get_num_new_matched_tokens`）。
    - **SGLang**：HiCache（`--enable-hierarchical-cache`），用 `HiRadixCache`（`srt/mem_cache/hiradix_cache.py`）在基数树的节点上记录 KV 位于哪一层；`--hicache-ratio` / `--hicache-size` 设置主机内存层的大小，`--hicache-storage-backend` 选择第三层存储（`srt/mem_cache/storage/` 下有 file、mooncake_store、hf3fs、lmcache 等实现）。GPU 与主机之间的数据搬运由 `HiCacheController` 在独立的 CUDA stream 上异步完成。

!!! interview "面试怎么答"
    被问到"KV Cache 放不下怎么办"时，按层次回答：**减少 KV**（GQA/MLA、KV 量化、滑动窗口）→ **更好地管理 GPU 上的 KV**（分页、前缀缓存、抢占）→ **扩展容量**（分层缓存：CPU 内存、SSD、分布式 KV 存储）→ **分摊到更多卡**（TP、上下文并行）。说到分层缓存时，用本章的估算说明"读回比重算快几十倍"，再点出工程难点：异步搬运与计算重叠、逐层加载、全局一致的块哈希、多实例共享时的一致性与淘汰策略。

## 练习

**1. PCIe 的带宽够吗？** 一台 8 卡服务器，每张卡的 PCIe 5.0 x16 约 50 GB/s。若某服务每秒有 20 个请求各命中 32K token 的 CPU 缓存（Qwen2.5-7B，TP=1），PCIe 带宽够不够？

??? success "参考答案"
    每个请求要读回 32768 × 56 KB ≈ 1.9 GB，每秒 20 个就是 38 GB/s。如果这些请求平均分布在 8 张卡上，每张卡约 4.7 GB/s，远低于 50 GB/s，完全够用；如果都集中在一张卡上，就接近上限了。还要注意 CPU 内存本身的带宽，以及多张卡共享 PCIe 交换芯片时的争用。

**2. 该不该卸载？** 有人建议：既然读回这么快，干脆把所有请求的 KV 都实时写一份到 CPU（写穿），这样 GPU 可以更激进地淘汰。这样做有什么代价？

??? success "参考思路"
    写穿会持续占用 PCIe 带宽（每生成一个 token 都要写一份 KV），与读回、与 PD 分离的传输争用；CPU 内存的容量同样有限，写进去的大部分 KV 可能永远不会被再次命中。更合理的做法是按价值写：只写可能被复用的部分（例如装满的块、多轮对话的历史、系统提示词），或者只在被驱逐时写（写回），并结合访问频率决定保留时间。

## 小结

- [x] 读回一个 token 的 KV 比重算快几倍（SSD）到几十上百倍（CPU 内存），MLA 模型优势更大。
- [x] 分层缓存：GPU → CPU 内存 → SSD / 分布式存储；块哈希是跨层、跨实例的全局键。
- [x] 驱逐时写到下一层，命中时读回；实现时要保护本次查找已经命中的块，并尽量让搬运与计算重叠。
- [x] vLLM 通过 KV connector（原生卸载或 LMCache）实现，与 PD 分离共用接口；SGLang 通过 HiCache 实现。
