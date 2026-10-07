# Prefix caching: hashed blocks and radix trees

<p class="lead">Many requests share the same beginning: a system prompt, the history of a multi-turn conversation, few-shot examples, the ever-growing context of an agent calling repeatedly. The same tokens at the same positions have exactly the same KV, so they need computing only once. This chapter implements two kinds of prefix cache, vLLM's "chained-hash blocks + LRU free queue" and SGLang's "radix tree", verifies on a real model that reuse leaves the output unchanged, and compares their hit rates under different workloads.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. Why does vLLM's block hash include the parent block's hash?
    2. After a block's reference count drops to 0 and it goes back to the free queue, can it still be hit? When does it really become invalid?
    3. When a prompt hits the cache completely, why must the last token still be recomputed?
    4. Why do radix tree nodes need a "lock"? Why does eviction only start from leaves?
    5. What is cache-aware scheduling? What is cache-aware routing?

??? success "Answers (try first, then expand to compare)"
    1. A block's KV depends not only on the tokens in the block but on every token before it (attention sees the whole prefix). With the parent's hash included, "same hash" means "the whole prefix is the same", so comparing one hash decides whether it can be reused.
    2. Yes: when a block with reference count 0 returns to the free queue, it keeps its content and hash, and the free queue doubles as an LRU cache, so a new request can hit it and take it back; it becomes invalid only when it is allocated to other content and actually overwritten.
    3. Every step must compute at least one token to get the last position's hidden state and compute the logits for sampling the next token; so the last token is always recomputed.
    4. The lock (a reference count) protects prefixes used by running requests, so their KV is not evicted. Evict only from leaves: an internal node is a prefix of other nodes, and evicting it first would make its children unreachable; starting from leaves keeps the tree a valid prefix structure.
    5. Cache-aware scheduling: within one instance, schedule first the requests that hit the cache (for example, ordered by longest prefix match); cache-aware routing: across instances, send a request to the instance most likely to have its prefix cached already.

<!-- comic ../assets/comics/prefix-cache.webp is in Chinese; put it back once the English version exists -->

## Two approaches {#两种思路}

| | vLLM: hashed blocks | SGLang: radix tree |
| --- | --- | --- |
| Unit of caching | full KV blocks (16 tokens by default) | tokens (with a page size of 1); pages are also supported |
| Lookup | compute chained hashes block by block and look them up in a hash table | match from the root down the tree |
| Where the cache lives | in the blocks themselves: free blocks keep their content and hash, invalidated only when reallocated | tree nodes hold the KV slots, released only on eviction |
| Eviction | the free queue is the LRU queue; reuse from its head | delete from the least recently accessed leaf |
| Strengths | merged with the block allocator, simple to implement; hashes serve as global identifiers, easy to share across machines (KV events, offloading, PD disaggregation) | naturally expresses tree-shaped sharing (many branches, many turns), fine-grained |

## Hashed blocks: vLLM's approach {#哈希块vllm-的做法}

Each full block has a hash: `hash(parent block hash, this block's tokens)`. Because it includes the parent's hash, two blocks with the same hash mean **all tokens from the start of the sequence to this block are the same**, not just this block's 16 tokens. This is crucial: the same 16 tokens after a different context have different KV (attention sees a different history, and the positions may differ too).

```python title="prefix_cache.py"
"""prefix_cache.py —— vLLM 式的前缀缓存：按块做链式哈希，空闲块兼作 LRU 缓存。

- 每个"装满的"块有一个哈希 = hash(父块哈希, 本块 token)，因此哈希相同 ⇔ 从开头到这个块的所有 token 都相同；
- 引用计数降为 0 的块进入空闲队列，但内容和哈希保留，仍可被命中；
- 分配新块时从空闲队列头部取，如果取到的块带着哈希，就把它从缓存中"驱逐"——队列顺序即 LRU 顺序。
"""

from collections import OrderedDict

from paged import BlockPool


class PrefixCachingBlockPool(BlockPool):
    enable_caching = True

    def __init__(self, num_blocks: int, block_size: int):
        super().__init__(num_blocks)
        self.block_size = block_size
        self.free_queue = OrderedDict((b, None) for b in range(num_blocks))   # the head is reused first
        self.hash_to_block: dict[int, int] = {}
        self.block_hash: list[int | None] = [None] * num_blocks
        self.num_queries = self.num_hits = 0                                   # counted in tokens

    def allocate(self, n: int) -> list[int] | None:
        if n > len(self.free_queue):
            return None
        blocks = []
        for _ in range(n):
            b, _ = self.free_queue.popitem(last=False)
            if self.block_hash[b] is not None:                                 # evict: this block is about to be overwritten with new content
                del self.hash_to_block[self.block_hash[b]]
                self.block_hash[b] = None
            self.ref_cnt[b] = 1
            blocks.append(b)
        return blocks

    def free(self, blocks: list[int]) -> None:
        # free in reverse: blocks at the end of a sequence are least likely to be shared, so put them first to be evicted first
        for b in reversed(blocks):
            self.ref_cnt[b] -= 1
            if self.ref_cnt[b] == 0:
                self.free_queue[b] = None

    def touch(self, blocks: list[int]) -> None:
        """命中的块被新请求引用：引用计数 +1；如果它在空闲队列里，就移出来。"""
        for b in blocks:
            if self.ref_cnt[b] == 0:
                del self.free_queue[b]
            self.ref_cnt[b] += 1

    def block_hashes(self, token_ids: list[int]) -> list[int]:
        hashes, parent = [], None
        for start in range(0, len(token_ids) - self.block_size + 1, self.block_size):
            parent = hash((parent, tuple(token_ids[start:start + self.block_size])))
            hashes.append(parent)
        return hashes

    def lookup(self, token_ids: list[int]) -> list[int]:
        """返回最长的已缓存前缀对应的块（只看装满的块）。"""
        hit = []
        for h in self.block_hashes(token_ids):
            b = self.hash_to_block.get(h)
            if b is None:
                break
            hit.append(b)
        self.num_queries += len(token_ids)
        self.num_hits += len(hit) * self.block_size
        return hit

    def cache_blocks(self, token_ids: list[int], block_ids: list[int], num_computed: int) -> None:
        """前向之后调用：把新装满的块登记进缓存。"""
        for i, h in enumerate(self.block_hashes(token_ids[:num_computed])):
            b = block_ids[i]
            if self.block_hash[b] is None and h not in self.hash_to_block:
                self.block_hash[b] = h
                self.hash_to_block[h] = b
```

The cleverest part of this implementation is that **the free queue doubles as an LRU cache**. When a request finishes, its blocks' reference counts drop to 0 and they return to the free queue, but their content and hashes remain, so the next request can still hit them; only when a block is allocated to someone else and about to be overwritten is its hash deleted. Since allocation always takes from the head and freed blocks go to the tail, the blocks unused for the longest are overwritten first: that is LRU.

```pycon
>>> from prefix_cache import PrefixCachingBlockPool
>>> pool = PrefixCachingBlockPool(num_blocks=4, block_size=2)
>>> a = [1, 2, 3, 4, 5]                          # 5 tokens: two full blocks plus one partial block
>>> blocks = pool.allocate(3)
>>> pool.cache_blocks(a, blocks, num_computed=5) # only full blocks enter the cache
>>> [pool.block_hash[b] is not None for b in blocks]
[True, True, False]
>>> pool.free(blocks)                            # the request finishes: blocks return to the free queue with content and hashes intact
>>> list(pool.free_queue)                        # freed in reverse: the tail blocks come first and are reused first
[3, 2, 1, 0]
>>> hit = pool.lookup([1, 2, 3, 4, 9])           # the new request shares the first 4 tokens
>>> hit
[0, 1]
>>> pool.touch(hit)                              # the hit blocks are referenced and leave the free queue
>>> list(pool.free_queue)
[3, 2]
>>> pool.free(hit)                               # the second request finishes too
>>> pool.allocate(3)                             # 3 new blocks needed: block 1 is reused and its cache entry is invalidated
[3, 2, 1]
>>> pool.lookup([1, 2, 3, 4, 9])                 # now only the first block hits
[0]
```

### Hooking into the scheduler {#接入调度器}

The previous chapter's scheduler calls `_lookup_prefix_cache` when admitting a new request: the blocks it hits become the start of the request's block table, and `num_computed` is set to the number of tokens hit, after which scheduling and computation proceed as if those tokens had been computed. One detail: **leave at least the last token as a miss**. Even if the whole prompt is in the cache, a forward pass on the last token is needed to get the next token's logits.

Check it on a real model: 6 requests share a fairly long system prompt. Run the first request alone, then the other 5 together:

```python
import time
import torch
from transformers import AutoTokenizer
from mini_llm import Transformer, generate
from nano_engine import LLMEngine, SamplingParams
from prefix_cache import PrefixCachingBlockPool

torch.set_num_threads(16)
path = "models/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(path)
model = Transformer.from_pretrained(path)

system = "你是一个推理优化专家，回答要简洁、准确，必要时给出数字。" * 6
questions = ["什么是 KV Cache？", "用一句话解释连续批处理。", "Python 的 GIL 是什么？",
             "写一个关于月亮的比喻。", "Explain tensor parallelism in one sentence.", "1+1 等于几？请直接回答。"]
prompts = [tok(tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": q}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)).input_ids for q in questions]
reference = [generate(model, torch.tensor([p]), 16, eos_token_id=tok.eos_token_id)[0].tolist() for p in prompts]

for caching in (False, True):
    pool = PrefixCachingBlockPool(256, 16) if caching else None
    engine = LLMEngine(model, eos_token_id=tok.eos_token_id, pool=pool)
    first = engine.generate(prompts[:1], SamplingParams(max_tokens=16))
    n_steps = len(engine.step_log)
    t0 = time.perf_counter()
    rest = engine.generate(prompts[1:], SamplingParams(max_tokens=16))
    elapsed = time.perf_counter() - t0
    computed = sum(s["num_tokens"] for s in engine.step_log[n_steps:])
    print(f"前缀缓存 {'开' if caching else '关'}：后 5 个请求共计算 {computed} 个 token，用时 {elapsed:.2f} s，"
          f"输出一致：{first + rest == reference}")
    assert first + rest == reference
print(f"提示词长度 {[len(p) for p in prompts]}，命中 {pool.num_hits} / {pool.num_queries} 个 token")
```

```text
前缀缓存 关：后 5 个请求共计算 705 个 token，用时 2.07 s，输出一致：True
前缀缓存 开：后 5 个请求共计算 225 个 token，用时 1.66 s，输出一致：True
提示词长度 [123, 126, 126, 126, 128, 132]，命中 480 / 755 个 token
```

With reuse, the tokens to compute drop by about 70%, and the output is unchanged. The time here shrinks by only about 20% because decode takes most of the time in this CPU experiment; on a GPU with long prompts, the savings are mainly in prefill time, and TTFT drops significantly.

## Radix trees: SGLang's approach {#基数树sglang-的做法}

SGLang organizes all cached sequences in a radix tree (a compressed prefix tree). Each edge holds a run of tokens and their KV slots, and the path from the root to a node is a cached prefix:

![Figure: a radix tree: requests sharing a prefix reuse KV along the same path](../assets/figures/radix-tree.svg){.aig-svg}

Add requests one at a time to see how the tree grows and how the hit rate changes:

<div class="aig-widget" data-widget="radixcache"></div>

```python title="radix.py"
"""radix.py —— SGLang 式的基数树前缀缓存（token 粒度）。

树的每条边保存一段 token 序列（key）和这些 token 的 KV 槽位（value）。从根到某个节点的路径拼起来，
就是一个被缓存过的前缀。匹配时沿树向下走，必要时把一条边劈成两段；淘汰时按最近访问时间从叶子开始删除，
正在被请求使用的节点通过 lock_ref 保护起来，不会被淘汰。
"""

import heapq
import itertools


class Node:
    def __init__(self, key=(), value=(), parent=None):
        self.children: dict[int, Node] = {}   # children, keyed by the first token of the edge
        self.key, self.value = list(key), list(value)
        self.parent = parent
        self.lock_ref = 0
        self.last_access = 0


def _common_len(a, b) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class RadixCache:
    def __init__(self):
        self.root = Node()
        self.root.lock_ref = 1                # the root is never evicted
        self._clock = itertools.count(1)
        self.evictable_size = 0               # tokens not locked and evictable
        self.protected_size = 0               # tokens locked by running requests

    def match_prefix(self, key: list[int]) -> tuple[list[int], Node]:
        """返回最长匹配前缀的 KV 槽位，以及匹配到的最后一个节点（用于加锁）。"""
        node, value = self.root, []
        now = next(self._clock)
        while key:
            child = node.children.get(key[0])
            if child is None:
                break
            n = _common_len(child.key, key)
            if n < len(child.key):
                child = self._split(child, n)   # matched only part of an edge: split it so the match ends on a node
            child.last_access = now
            value += child.value
            node, key = child, key[n:]
        return value, node

    def insert(self, key: list[int], value: list[int]) -> int:
        """插入一条序列，返回其中已经在树里的前缀长度（这部分的 value 是重复的，调用方应释放）。"""
        node, matched = self.root, 0
        now = next(self._clock)
        while key:
            child = node.children.get(key[0])
            if child is None:
                new = Node(key, value, node)
                new.last_access = now
                node.children[key[0]] = new
                self.evictable_size += len(key)
                return matched
            n = _common_len(child.key, key)
            if n < len(child.key):
                child = self._split(child, n)
            child.last_access = now
            matched += n
            node, key, value = child, key[n:], value[n:]
        return matched

    def _split(self, child: Node, n: int) -> Node:
        """把 child 的边从第 n 个 token 处劈开，返回新的中间节点。"""
        mid = Node(child.key[:n], child.value[:n], child.parent)
        mid.lock_ref, mid.last_access = child.lock_ref, child.last_access
        child.parent.children[mid.key[0]] = mid
        child.key, child.value, child.parent = child.key[n:], child.value[n:], mid
        mid.children[child.key[0]] = child
        return mid

    def inc_lock_ref(self, node: Node) -> None:
        while node is not self.root:
            if node.lock_ref == 0:
                self.evictable_size -= len(node.key)
                self.protected_size += len(node.key)
            node.lock_ref += 1
            node = node.parent

    def dec_lock_ref(self, node: Node) -> None:
        while node is not self.root:
            node.lock_ref -= 1
            if node.lock_ref == 0:
                self.evictable_size += len(node.key)
                self.protected_size -= len(node.key)
            node = node.parent

    def evict(self, num_tokens: int) -> list[int]:
        """按 LRU 从叶子开始淘汰至少 num_tokens 个 token，返回释放出来的 KV 槽位。"""
        leaves = [(n.last_access, id(n), n) for n in self._nodes() if not n.children and n.lock_ref == 0]
        heapq.heapify(leaves)
        freed = []
        while leaves and len(freed) < num_tokens:
            _, _, node = heapq.heappop(leaves)
            freed += node.value
            parent = node.parent
            del parent.children[node.key[0]]
            self.evictable_size -= len(node.key)
            if parent is not self.root and not parent.children and parent.lock_ref == 0:
                heapq.heappush(leaves, (parent.last_access, id(parent), parent))
        return freed

    def _nodes(self):
        stack = [self.root]
        while stack:
            n = stack.pop()
            if n is not self.root:
                yield n
            stack.extend(n.children.values())

    def total_size(self) -> int:
        return sum(len(n.key) for n in self._nodes())
```

A few key operations:

- **When matching**, if a new request shares only the first part of an edge, the edge is split in two (`_split`), so the match ends exactly on a node.
- **Locking**: when a request starts using a prefix, `lock_ref` is incremented on every node from the matched node up to the root. Locked nodes are never evicted, or the KV of a request being computed would be overwritten by someone else.
- **Eviction** starts only from leaves: an internal node is someone else's prefix, and deleting it would make its descendants unmatchable. After a leaf is deleted, if its parent becomes an unlocked leaf, the parent becomes a candidate too.

```pycon
>>> from radix import RadixCache
>>> tree = RadixCache()
>>> tree.insert([1, 2, 3, 4, 5], [10, 11, 12, 13, 14])   # value is the KV slots of these tokens
0
>>> tree.insert([1, 2, 3, 9, 9], [10, 11, 12, 20, 21])   # the first 3 tokens are already in the tree; the edge is split
3
>>> value, node = tree.match_prefix([1, 2, 3, 4, 7])
>>> value, node.key
([10, 11, 12, 13], [4])
>>> tree.inc_lock_ref(node)                               # [1, 2, 3, 4] is in use: lock it
>>> tree.evictable_size, tree.protected_size
(3, 4)
>>> tree.evict(10)                                        # try to free 10 tokens: only unlocked leaves can go
[14, 20, 21]
>>> tree.dec_lock_ref(node)                               # the request finishes: unlock
>>> sorted(tree.evict(10)), tree.total_size()
([10, 11, 12, 13], 0)
```

## Comparing hit rates {#命中率比较}

How different are the two methods' hit rates? Simulate four typical workloads (no model needed; just compare the reusable tokens):

```python
import random
from radix import RadixCache

random.seed(0)

def rand_tokens(n):
    return [random.randrange(1000, 50000) for _ in range(n)]

def block_hashes(tokens, block_size):
    hashes, parent = [], None
    for start in range(0, len(tokens) - block_size + 1, block_size):
        parent = hash((parent, tuple(tokens[start:start + block_size])))
        hashes.append(parent)
    return hashes

def hit_rate(workload, method):
    hits = total = 0
    if method == "radix":
        tree = RadixCache()
        for req in workload:
            hits += len(tree.match_prefix(req[:-1])[0])
            total += len(req)
            tree.insert(req, list(range(len(req))))
        return hits / total
    cached = set()
    for req in workload:
        n = 0
        for h in block_hashes(req[:-1], method):
            if h not in cached:
                break
            n += 1
        hits += n * method
        total += len(req)
        cached.update(block_hashes(req, method))
    return hits / total

system = rand_tokens(1000)
shared_system = [system + rand_tokens(random.randint(20, 200)) for _ in range(100)]
multi_turn = []
for _ in range(20):                                           # 20 conversations, 8 turns each
    history = rand_tokens(300)
    for _ in range(8):
        history = history + rand_tokens(random.randint(10, 60))   # the user's new message
        multi_turn.append(list(history))
        history = history + rand_tokens(random.randint(50, 300))  # the model's answer
examples, header = [rand_tokens(random.randint(100, 200)) for _ in range(5)], rand_tokens(37)
few_shot = [header + sum(examples[:random.randint(2, 5)], []) + rand_tokens(random.randint(20, 80)) for _ in range(100)]
parallel = []
for _ in range(30):                                           # 30 questions, 8 samples each
    problem = rand_tokens(random.randint(80, 400))
    parallel += [problem + rand_tokens(1) for _ in range(8)]

print(f"{'负载':16s}{'基数树':>8s}{'块 16':>8s}{'块 64':>8s}")
for name, w in [("共享系统提示词", shared_system), ("多轮对话", multi_turn),
                ("few-shot 示例数不同", few_shot), ("同一问题并行采样", parallel)]:
    print(f"{name:16s}" + "".join(f"{hit_rate(w, m):8.1%}" for m in ("radix", 16, 64)))
```

```text title="output"
负载                   基数树    块 16    块 64
共享系统提示词            89.5%   88.8%   86.0%
多轮对话               79.2%   78.6%   76.8%
few-shot 示例数不同     89.6%   87.8%   81.1%
同一问题并行采样           87.1%   84.2%   73.3%
```

The token-granular radix tree always hits the most, but compared with 16-token blocks the gap is at most about three percentage points: the block method loses only the partial block at the end of each shared prefix. The larger the block, the more noticeable the loss. So the real difference between the two is not the hit rate but engineering: vLLM's scheme fits naturally with the block allocator, KV events and KV sharing across machines; SGLang's tree makes complex sharing relationships easy to express and underpins its cache-aware scheduling.

## Cache-aware scheduling and routing {#缓存感知的调度与路由}

With a prefix cache, both **the order in which requests are processed** and **which instance they go to** affect the hit rate:

- **Cache-aware scheduling** (within one instance): SGLang's default policy is LPM (longest prefix match), scheduling first the requests with the longest match against the cache, so requests sharing a prefix run back to back and the prefix is not evicted before it is reused. The price is that requests with short matches may wait longer, so it falls back to FCFS when there are many requests.
- **Cache-aware routing** (across instances): requests with the same prefix should go to the same instance, or every instance computes and stores its own copy. SGLang Model Gateway (formerly sgl-router) keeps an approximate radix tree in the router to estimate what each instance has cached; vLLM publishes each instance's cache state through **KV events** (block cached, block evicted) for external routers (such as llm-d and the vLLM production stack). Routing must also balance load: if every request goes to the instance with the highest hit rate, that instance overloads.

!!! source "Source code"
    - **vLLM**: `KVCacheManager.get_computed_blocks` (`vllm/v1/core/kv_cache_manager.py`) finds the longest hit, and its comment states that at most `num_tokens - 1` tokens can hit, "because we need to recompute the last token" to get the logits. Block hashes are computed in `hash_block_tokens` in `kv_cache_utils.py`: `hash((parent block hash, this block's tokens, extra_keys))`. `extra_keys` includes hashes of multimodal inputs, the LoRA ID and a `cache_salt` for multi-tenant isolation; without them, requests with different content could wrongly share KV. The hash algorithm can be chosen with `--prefix-caching-hash-algo`, such as `sha256` (the default) or `xxhash`.
    - Caching and eviction live in `BlockPool` (`block_pool.py`): `cache_full_blocks` registers blocks, `touch` references them, `free_blocks` puts blocks without hashes at the head of the queue (LIFO, good for GPU locality) and blocks with hashes at the tail (LRU), and `_maybe_evict_cached_block` evicts when reusing. Caching or evicting a block produces a KV event (`kv_event_queue`).
    - **SGLang**: `RadixCache` (`srt/mem_cache/radix_cache.py`) provides `match_prefix`, `insert`, `evict`, `inc_lock_ref` and `dec_lock_ref`; when a request finishes or is chunked, `cache_finished_req` and `cache_unfinished_req` insert its KV into the tree. Scheduling policies are in `srt/managers/schedule_policy.py`: `CacheAwarePolicy` (LPM, DFS_WEIGHT) and `CacheAgnosticPolicy` (FCFS, LOF and others). Under `mem_cache/` there are also variants for hybrid models with sliding windows, Mamba and so on, plus `unified_cache/`, which is unifying them.

!!! interview "In an interview"
    "How do vLLM's and SGLang's prefix caches differ?" can be answered on three levels: **data structure** (chained-hash blocks vs. a radix tree), **granularity** (blocks vs. tokens) and **eviction** (the free queue as LRU vs. leaf LRU with reference locks), ending with the **effect**: hit rates are close (within a few percentage points in this chapter's simulation), and the difference is mostly in engineering trade-offs and the scheduling policies built on top. Mentioning the two details "a block hash includes its parent's hash" and "the last token is always recomputed" earns extra credit.

## Exercises {#练习}

**1. Multi-tenant isolation.** Two tenants use exactly the same system prompt. What is the risk of sharing KV? How can they be isolated without giving up reuse within each tenant?

??? success "Answer"
    The risk is a timing side channel: by measuring TTFT, an attacker can tell whether some content is already in the cache (TTFT is clearly shorter on a hit), and so infer what other tenants have sent. The fix is to mix a tenant-specific "salt" into the first block's hash, which is vLLM's `cache_salt`: requests from the same tenant hash the same and can reuse, while different tenants hash differently and never hit each other.

**2. Warming and pinning.** All requests to a service start with the same 8K-token system prompt, but occasionally it gets evicted under cache pressure, and a batch of requests suddenly has much longer TTFT. What can be done?

??? success "Approach"
    - Raise the prefix's "value": LRU evicts by access time, and this prefix is already hot, so being evicted means the cache is too small; first check the KV memory configuration and concurrency;
    - Tiered caching: offload evicted blocks to CPU memory or SSD and load them back when needed, which is faster than recomputing (see [hierarchical KV caching](../distributed/kv-offload.md));
    - If the engine supports it, "pin" it in the cache (an extra reference that is never released);
    - Send a warm-up request when the service starts, so the prefix enters the cache first.

## Summary {#小结}

- [x] Prefix caching reuses the KV of identical prefixes; vLLM uses chained-hash blocks, SGLang a radix tree.
- [x] A block hash includes its parent's hash, so "same hash means the whole prefix is the same"; the free queue doubles as an LRU cache, and a block becomes invalid only when reused.
- [x] The last token must always be recomputed to get the logits; the radix tree protects prefixes in use with reference locks and evicts only from leaves.
- [x] The two have similar hit rates, with larger blocks losing more; cache-aware scheduling and routing raise the hit rate further.
