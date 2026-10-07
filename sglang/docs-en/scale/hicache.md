# HiCache, the tiered cache: the GPU, the CPU and storage

<p class="lead">The radix tree governs only the KV in GPU memory; when memory fills up a node is evicted, and an evicted node is recomputed. #2693 "Hierarchical Caching for SGLang", merged on 23 February 2025, gave the tree a second tier: a node's KV can be backed up to CPU memory (<code>host_value</code>), so after being evicted from the GPU it is still on the tree, and a later hit moves it back instead of recomputing it. Half a year later came a third tier: remote storage like Mooncake, 3FS and NIXL, letting the cache be shared across instances. This chapter reads the first version of <code>HiRadixCache</code> and <code>HiCacheController</code>, sees how the small change of "a node with two values" holds up three tiers, and goes through every reconciliation with PD disaggregation, the page size and TP.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What state does `HiRadixCache` add on top of `RadixCache`? How does evicting a backed-up node differ from evicting one that is not?
    2. Why does `HiCacheController` need threads of its own? When does each of `write_through`, `write_through_selective` and `write_back` write KV to the CPU?
    3. What happens when a prefix matched is only on the CPU? What is `load_back_threshold`?
    4. How was the storage tier (the third) connected? Why does a node carry a hash?

??? success "Answers for the self-test (answer first, then open this)"
    1. Each node gains a `host_value` (its position in the CPU memory pool) and a `backuped` state, plus a CPU-side memory pool (`MHATokenToKVPoolHost` / `MLATokenToKVPoolHost`, laid out like the GPU pool). On eviction, a backed-up node only has its GPU memory freed and stays on the tree (`_evict_backuped`) while one that is not backed up is deleted entirely (`_evict_regular`); when the CPU memory fills up there is a separate `evict_host`.
    2. Copies between GPU and CPU memory go over PCIe and are asynchronous to the forward pass; the controller has its own writing and reading threads consuming `TransferBuffer` queues, copying layer by layer and using a `LayerDoneCounter` so that reads can be waited on a layer at a time. `write_through`: back up as soon as the node is inserted; `write_through_selective`: back up only once the hit count reaches a threshold (3 by default), so a prefix used once does not occupy memory; `write_back`: write to the CPU only when evicting from the GPU.
    3. `match_prefix(include_evicted=True)` matches backed-up nodes; if the CPU-side prefix that hits is longer than `load_back_threshold` (10 tokens by default), GPU memory is allocated and a `load_back` is issued, and the request waits for the copy before entering a batch; a shorter one is not worth moving and is simply recomputed.
    4. "HiCache Storage Layer Prototype" (#7704) of 2025-07 added a `HiCacheStorage` interface (get / set / exists), with backends for files, 3FS, Mooncake Store, NIXL and others; for a node's KV to be shared between instances the key cannot be a local slot number, so a hash is computed over the tokens' content at insertion (#9053), and that hash is the storage tier's key.

A six-panel strip first, then the chapter:

<!-- comic ../assets/comics/hicache.webp is in Chinese; put it back once the English version exists -->

## The timing, read from the PR numbers {#从-pr-号看时间}

```bash title="hicache-commits.sh"
for h in 51caee740f 5d6e9467d4 13387e6b7a 6c7a152c5a 10b544ae9b 0e0ec70200 e119f04215 a023856b12 9d33fcfb8e 299803343d 2cd2e27f80 9f78f391ae 8b6966d020 1ccd59c715; do
  git log -1 --date=short --format='%ad  %h  %s' $h | cut -c1-96
done
```

```text title="output"
2025-01-07  51caee740f  Host memory pool for hierarchical caching (#2771)
2025-01-10  5d6e9467d4  Cache controller for hierarchical caching (#2804)
2025-01-17  13387e6b7a  Multi-turn benchmark for hierarchical caching (#2942)
2025-02-23  6c7a152c5a  Hierarchical Caching for SGLang (#2693)
2025-03-12  10b544ae9b  Hierarchical Caching Refactoring and Fixing TP issue (#4082)
2025-03-14  0e0ec70200  Hierarchical Caching supports MLA (#4009)
2025-04-01  e119f04215  Large page size aligned hierarchical caching (#4581)
2025-06-14  a023856b12  Move host memory pools into a separate file (#7200)
2025-07-18  9d33fcfb8e  Hicache Storage Layer Prototype (#7704)
2025-07-31  299803343d  Add hf3fs support for hicache storage (based on #7704) (#7280)
2025-07-31  2cd2e27f80  SGLang HiCache NIXL Connector (#8488)
2025-08-11  9f78f391ae  HiCache Storage: generate hash when inserting new nodes (#9053)
2025-08-31  8b6966d020  [HiCache] Storage Refactoring (#9797)
2025-09-18  1ccd59c715  [HICache] introduce evict policy (#10190)
```

The main PR #2693's number is from December 2024 and it merged on 23 February 2025; the two pieces it depends on merged first: #2771 "Host memory pool" of 7 January (a CPU-side pool laid out like the GPU's) and #2804 "Cache controller" of 10 January (the transfer threads), with #2942, a multi-turn benchmark, on 17 January — the measuring tool prepared before the feature merged. This is the landing of "multi-layer radix cache #2693" from the Q4 2024 roadmap (issue #1487), and the v0.4 blog's "what's next" named it too.

## A node with two values {#树的节点有两个值}

v0.4.6's `HiRadixCache` inherits from `RadixCache`:

```python title="python/sglang/srt/mem_cache/hiradix_cache.py @ v0.4.6 L23-69" linenums="23"
class HiRadixCache(RadixCache):

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: TokenToKVPoolAllocator,
        tp_cache_group: torch.distributed.ProcessGroup,
        page_size: int,
        hicache_ratio: float,
        hicache_size: int,
        hicache_write_policy: str,
    ):
        self.kv_cache = token_to_kv_pool_allocator.get_kvcache()
        if isinstance(self.kv_cache, MHATokenToKVPool):
            self.token_to_kv_pool_host = MHATokenToKVPoolHost(
                self.kv_cache, hicache_ratio, hicache_size, page_size
            )
        elif isinstance(self.kv_cache, MLATokenToKVPool):
            self.token_to_kv_pool_host = MLATokenToKVPoolHost(
                self.kv_cache, hicache_ratio, hicache_size, page_size
            )
        else:
            raise ValueError(f"HiRadixCache only supports MHA and MLA yet")

        self.tp_group = tp_cache_group

        self.load_cache_event = threading.Event()
        self.cache_controller = HiCacheController(
            token_to_kv_pool_allocator,
            self.token_to_kv_pool_host,
            page_size,
            load_cache_event=self.load_cache_event,
            write_policy=hicache_write_policy,
        )

        # record the nodes with ongoing write through
        self.ongoing_write_through = {}
        # record the node segments with ongoing load back
        self.ongoing_load_back = {}
        # todo: dynamically adjust the threshold
        self.write_through_threshold = (
            1 if hicache_write_policy == "write_through" else 3
        )
        self.load_back_threshold = 10
        super().__init__(
            req_to_token_pool, token_to_kv_pool_allocator, page_size, disable=False
        )
```

Initialisation does three things: build a CPU pool matching the GPU pool's type (MHA or MLA, sized by `hicache_ratio` or `hicache_size`), build a `HiCacheController`, and set the two thresholds (the write threshold 1 or 3 by policy, the load threshold 10). Backing a node up:

```python title="python/sglang/srt/mem_cache/hiradix_cache.py @ v0.4.6 L84-104" linenums="84"
    def write_backup(self, node: TreeNode, write_back=False):
        host_indices = self.cache_controller.write(
            device_indices=node.value,
            node_id=node.id,
        )
        if host_indices is None:
            self.evict_host(len(node.value))
            host_indices = self.cache_controller.write(
                device_indices=node.value,
                node_id=node.id,
            )
        if host_indices is not None:
            node.host_value = host_indices
            self.ongoing_write_through[node.id] = node
            if not write_back:
                # no need to lock nodes if write back
                self.inc_lock_ref(node)
        else:
            return 0

        return len(host_indices)
```

`cache_controller.write` allocates a place in the CPU pool and queues the copy, calling `evict_host` and retrying if the allocation fails; on success the node records its `host_value`, enters `ongoing_write_through` and is locked — it cannot be evicted from GPU memory until the copy finishes (except under `write_back`, where the node is on the eviction path to begin with). The eviction logic takes two paths:

```python title="python/sglang/srt/mem_cache/hiradix_cache.py @ v0.4.6 L154-204" linenums="154"
    def evict(self, num_tokens: int):
        leaves = self._collect_leaves_device()
        heapq.heapify(leaves)

        num_evicted = 0
        write_back_nodes = []
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)

            if x.lock_ref > 0:
                continue

            if not x.backuped:
                if self.cache_controller.write_policy == "write_back":
                    # write to host if the node is not backuped
                    num_evicted += self.write_backup(x, write_back=True)
                    write_back_nodes.append(x)
                else:
                    num_evicted += self._evict_regular(x)
            else:
                num_evicted += self._evict_backuped(x)

            for child in x.parent.children.values():
                if child in write_back_nodes:
                    continue
                if not child.evicted:
                    break
            else:
                # all children are evicted or no children
                heapq.heappush(leaves, x.parent)

        if self.cache_controller.write_policy == "write_back":
            self.writing_check(write_back=True)
            for node in write_back_nodes:
                assert node.backuped
                self._evict_backuped(node)

    def _evict_backuped(self, node: TreeNode):
        # evict a node already written to host
        num_evicted = self.cache_controller.evict_device(node.value, node.host_value)
        assert num_evicted > 0
        self.evictable_size_ -= num_evicted
        node.value = None
        return num_evicted

    def _evict_regular(self, node: TreeNode):
        # evict a node not initiated write to host
        self.cache_controller.mem_pool_device_allocator.free(node.value)
        num_evicted = len(node.value)
        self._delete_leaf(node)
        return num_evicted
```

The leaf LRU's framework is [chapter three](../origins/radix-v1.md)'s, and the difference comes after a leaf is popped: a node with a `host_value` only has its GPU memory freed (`_evict_backuped`) and stays on the tree with an empty `value`, while one without is deleted entirely (`_evict_regular`). Under the `write_back` policy the backup is triggered at eviction time. So the tree acquires nodes that exist only on the CPU, which `match_prefix(include_evicted=True)` can match and from which the scheduler decides whether to `load_back`.

![Figure: the three tiers and the controller's two queues](../assets/figures/sgl-hicache-tiers.svg){.aig-svg}

## The controller: two threads, moving layer by layer {#控制器两个线程按层搬运}

```python title="python/sglang/srt/managers/cache_controller.py @ v0.4.6 L146-200" linenums="146"
class HiCacheController:

    def __init__(
        self,
        token_to_kv_pool_allocator: TokenToKVPoolAllocator,
        mem_pool_host: HostKVCache,
        page_size: int,
        load_cache_event: threading.Event = None,
        write_policy: str = "write_through_selective",
    ):
        self.mem_pool_device_allocator = token_to_kv_pool_allocator
        self.mem_pool_device = token_to_kv_pool_allocator.get_kvcache()
        self.mem_pool_host = mem_pool_host
        self.write_policy = write_policy
        self.page_size = page_size

        self.load_cache_event = load_cache_event
        self.layer_done_counter = LayerDoneCounter(self.mem_pool_device.layer_num)
        self.mem_pool_device.register_layer_transfer_counter(self.layer_done_counter)

        if write_policy not in [
            "write_through",
            "write_through_selective",
            "write_back",
        ]:
            raise ValueError(f"Invalid write policy: {write_policy}")

        self.write_queue = PriorityQueue()
        self.load_queue = PriorityQueue()

        self.ack_write_queue = Queue()
        self.ack_load_queue = Queue()

        self.stop_event = threading.Event()
        self.write_buffer = TransferBuffer(self.stop_event)
        self.load_buffer = TransferBuffer(
            self.stop_event, buffer_count=10, max_buffer_size=100
        )

        self.write_stream = torch.cuda.Stream()
        self.load_stream = torch.cuda.Stream()

        self.write_thread = threading.Thread(
            target=(
                self.write_thread_func_buffer
                if self.page_size == 1
                else self.write_thread_func_direct
            ),
            daemon=True,
        )
        self.load_thread = threading.Thread(
            target=self.load_thread_func_layer_by_layer, daemon=True
        )
        self.write_thread.start()
        self.load_thread.start()
```

`HiCacheController` holds two `TransferBuffer`s (a write queue and a read queue) and two threads; each `CacheOperation` is "a stretch of device indices ↔ a stretch of host indices", which can be merged (`merge`) or split by a factor (`split`) into pieces for pipelining. On a read, the `LayerDoneCounter` makes "the first k layers have arrived" something that can be waited on — the forward pass goes layer by layer, so the first layer can begin as soon as its KV arrives rather than waiting for everything (the "layer-wise load" of later versions). The threads use their own CUDA stream for the `copy_`, keeping off the forward stream.

## Reconciliation {#磨合}

HiCache ran into every part of the runtime:

| Date | Commit | What it reconciled with |
| --- | --- | --- |
| 2025-03-12 | #4082 "Refactoring and Fixing TP issue" | under TP the ranks' backup and load decisions have to agree (`tp_cache_group`'s synchronisation) |
| 2025-03-14 | #4009 | MLA's CPU pool |
| 2025-03-13 / 04-01 | #4397, #4581 | with a page size above 1 the tree is page-aligned, and so must the backups be |
| 2025-06-20 | #7159 | DP attention plus HiCache hanging |
| 2025-07-30 | PD plus HiCache (the #8211 series) | a prefill instance reusing a cache in remote storage |
| 2025-09 → 10 | #10190, #11506 | eviction policies made pluggable (LRU, LFU, SLRU…) |

Every row is a combination problem of "one new dimension x the existing ones", which is also why the Q3 2025 roadmap made restructuring `mem_cache` a priority.

## The third tier: storage {#第三层存储}

#7704 "Hicache Storage Layer Prototype" of 18 July 2025 defined the `HiCacheStorage` interface, #7280 added a 3FS backend and #8488 a NIXL connector on 31 July, and Mooncake Store was connected around August; #9053 (11 August) had a tree node compute a hash at insertion (per page and chained: each page's hash includes the previous page's) as the key for sharing across instances; #9797 (31 August) restructured the storage layer. Today's backends:

```bash title="hicache-storage.sh"
REF=${REF:-29f6d408c0}
echo "mem_cache/storage/ 下的后端目录："
git ls-tree -r --name-only "$REF" -- python/sglang/srt/mem_cache/storage | sed 's|python/sglang/srt/mem_cache/storage/||' | awk -F/ 'NF > 1 {print $1}' | sort | uniq -c | awk '{printf "   %2d 个文件  %s\n", $1, $2}'
printf 'hiradix_cache.py 行数：v0.4.6 %d，v0.5.0rc0 %d\n' "$(git show v0.4.6:python/sglang/srt/mem_cache/hiradix_cache.py | wc -l)" "$(git show v0.5.0rc0:python/sglang/srt/mem_cache/hiradix_cache.py | wc -l)"
echo "hiradix_cache.py 的最后一个提交：$(git log -1 --date=short --format='%ad %h %s' "$REF" -- python/sglang/srt/mem_cache/hiradix_cache.py | cut -c1-80)"
echo "今天的统一树实现：$(git ls-tree -r --name-only "$REF" -- python/sglang/srt/mem_cache | grep -E 'unified_radix_cache.py|unified_cache/|rust_tree_core/' | wc -l) 个文件（unified_radix_cache.py、unified_cache/、rust_tree_core/）"
```

```text title="output"
mem_cache/storage/ 下的后端目录：
    3 个文件  aibrix_kvcache
    3 个文件  eic
    2 个文件  file
    7 个文件  flexkv
    9 个文件  hf3fs
    4 个文件  lmcache
    2 个文件  mmap
    5 个文件  mooncake_store
    7 个文件  nixl
    4 个文件  npu_memcache
    2 个文件  shm
    3 个文件  simm
    7 个文件  tensorcast_store
    4 个文件  umbp
hiradix_cache.py 行数：v0.4.6 464，v0.5.0rc0 747
hiradix_cache.py 的最后一个提交：2026-09-22 4ce23542bf [HiCache] Remove the unused HiRadixCache (#40787)
今天的统一树实现：22 个文件（unified_radix_cache.py、unified_cache/、rust_tree_core/）
```

The storage tier's typical use: a prefill instance writes the KV it computed to shared storage, and an instance on another machine that hits the same prefix prefetches it from storage (`storage_prefetch.py`); under PD disaggregation a prefill node can reuse a prefix another prefill node wrote to storage. That turns the prefix cache from a single machine's resource into a cluster's — and [chapter 21](../platform/gateway.md)'s gateway does cache-aware routing across instances on that basis.

## Design trade-offs {#设计取舍}

- **Extend the node rather than build a second tree.** One node with two values (`value` and `host_value`) lets the matching, the locking and the eviction logic all be reused; the price is that every method of `RadixCache` has to consider the state of "the value is empty but the node is there".
- **The CPU pool laid out like the GPU pool.** A copy is one contiguous `copy_` with no rearrangement; the price is that CPU memory is managed at the GPU's slot granularity, which is less compact than storing a request's stretch whole.
- **Asynchronous threads plus locks.** A node is locked during the copy rather than having its data duplicated, which saves memory but lengthens the time a node cannot be evicted.
- **Three write policies.** The default `write_through_selective` filters "prefixes worth backing up" by hit count, a compromise between the memory's capacity and the backup bandwidth.

## What happened afterwards {#后来怎么样了}

- `hiradix_cache.py` grew from 464 lines to v0.5.0rc0's 747; in 2026 the tiering logic merged into a unified tree implementation (`unified_radix_cache.py`, `unified_cache/`, and `rust_tree_core/` written in Rust), and #40787 of 2026-09-22 deleted the no-longer-used `HiRadixCache` — the script's last two lines give both facts; beside it are a dozen or so more files, `storage/`, `hicache_storage.py`, `storage_prefetch.py`, `memory_pool_host.py` and others.
- 2026: `evict_policy.py`'s several policies, tiered caching for SWA and hybrid models, and DeepSeek-V4's sparse index cache with its host-side pool.
- The inference-systems handbook's [KV offload chapter](serving://distributed/kv-offload/) compares several tiering designs from a systems point of view; what is here is the shape it takes in SGLang.

## Exercises {#练习}

**1. How long the lock lasts.** `write_backup` calls `inc_lock_ref` for a node outside write_back mode; when is it unlocked? What does a slow copy do to eviction?

??? success "A way to approach it"
    `writing_check` unlocks (the nodes in `ongoing_write_through`) while checking the write queue's completions; with slow copies many nodes stay locked, the evictable memory shrinks, and admission becomes more conservative.

**2. What the threshold does.** With `load_back_threshold = 10`, what happens when a CPU prefix of 8 tokens hits? Why not move it?

??? success "Answer"
    It is not moved and those 8 tokens are recomputed as a miss. Moving it back means allocating GPU memory, queueing and waiting for a copy, and prefilling 8 tokens costs less than all of that.

**3. The hash's chaining.** Read how `hash_value` is computed in the baseline commit's `radix_cache.py` and explain why a page's hash has to include the previous page's rather than hashing only this page's tokens.

??? success "A way to approach it"
    The KV's content depends on the whole prefix and not only on this page's tokens; hashing this page alone would treat the same stretch of tokens in different contexts as the same KV. A chained hash makes the key imply the whole prefix.

!!! interview "How to answer in an interview"
    "How do you make a KV cache multi-tier?" — Use SGLang's three: a tree node carries a `host_value` and a backed-up node only has its GPU memory freed on eviction; the controller's two threads move data asynchronously layer by layer; three write policies filter the prefixes worth backing up; and the third tier uses a content hash as the key into remote storage, making the cache a cluster resource. Then mention the reconciliations: TP consistency, page alignment, DP attention, PD — which shows that a multi-tier cache's difficulty is in combining with everything else.

## Summary {#小结}

- [x] #2693 (2025-02-23): `HiRadixCache` gives a node a `host_value` and a backed-up node only has its GPU memory freed on eviction; `HiCacheController`'s two threads move data asynchronously under three write policies.
- [x] The memory pool and the controller merged first, and the benchmark existed before the main PR did.
- [x] The storage tier from 2025-07 uses a content hash as the key, turning the cache from a single machine's resource into a cluster's; every new dimension has to be reconciled afresh with TP, the page size, DP and PD.
