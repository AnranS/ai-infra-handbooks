# Several cards: DP attention, EP and the birth of the Rust router

<p class="lead">In the fourth quarter of 2024 SGLang crossed the boundary of "one group of TP ranks" in three directions at once: data-parallel attention for DeepSeek's MLA (within one process group, attention computed per DP replica and the MoE layers brought together), expert parallelism cutting a MoE's experts across cards, and a Rust router outside the engine — one that keeps an approximate copy of every worker's radix tree and sends a request to the one whose prefix matches most. All three went from their first commit to usable between November and December, and the v0.4 blog gives two numbers: DP attention raises DeepSeek's decode throughput 1.9 times, and cache-aware routing raises the throughput 1.9 times and the hit rate 3.8 times.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Why does tensor parallelism waste KV memory on an MLA model? How does DP attention solve it? Where do the ranks have to synchronise?
    2. What does `prepare_dp_attn_batch` do? Why does the `IDLE` forward mode exist?
    3. How does the cache-aware router know what each worker has cached? When does it give up hit rate for load balance?
    4. Why is the router written in Rust? How does it relate to the data-parallel controller inside the engine?

??? success "Answers for the self-test (answer first, then open this)"
    1. MLA has only one compressed KV per layer (amounting to one KV head), which tensor parallelism cannot split, so every rank has to hold a complete copy of the KV; DP attention has each rank serve its own group of requests and hold its own group's KV, computing attention independently and only gathering every rank's tokens together (an all-gather) for the layers that work per token, the MoE and the dense FFN, scattering them back afterwards. Every step the ranks have to synchronise "how many tokens I have this batch" (which decides the gather's shape) and "whether my batch is decode or extend" (which decides whether a CUDA graph can be replayed).
    2. It collects every rank's token count with an `all_gather`; if this rank has no batch while others do, it makes an empty `IDLE` batch to keep step — because the MoE layer's collective needs every rank to take part and no one may be absent; then an `all_reduce(MIN)` tells whether every rank is decoding, which decides whether this step can replay a CUDA graph.
    3. It keeps an "approximate radix tree" per worker: when a request goes to a worker, the request's text is inserted into that worker's tree; on a query it matches the prefix against every tree, and if the match rate exceeds `cache_threshold` it goes to the worker that matched most. When the workers' loads differ by more than the thresholds (`balance_abs_threshold` and `balance_rel_threshold`), it goes to the idlest worker instead. The trees are evicted LRU periodically and are only approximately in step with the workers' real ones.
    4. Routing is on the request path, where Python's GIL and asyncio are a bottleneck under high concurrency, while Rust's tree operations and HTTP forwarding are both faster and leaner; and it has to be deployed independently of the engine (in front of several engines across several machines). The engine's `DataParallelController` dispatches among the replicas within one instance, and the router between instances.

A six-panel strip first, then the chapter:

<!-- comic ../assets/comics/multi-gpu.webp is in Chinese; put it back once the English version exists -->

## The timeline {#时间线}

```bash title="multi-gpu-commits.sh"
for h in 3839be2913 530ff541cf f9633fa9b9 976bc302e5 62832bb272 699384cb01 cbedd1db1d 4b0a1c9365 3d32e4a32c 2e4a5907c9 e3b3acfa6f e835a50021; do
  git log -1 --date=short --format='%ad  %h  %s' $h | cut -c1-96
done
```

```text title="output"
2024-10-28  3839be2913  [Router] Add a rust-based router (#1790)
2024-11-04  530ff541cf  [router] Impl radix tree and set up CI (#1893)
2024-11-10  f9633fa9b9  [rust] cache-aware DP - approx tree (#1934)
2024-11-16  976bc302e5  Support DP MLA (#1970)
2024-11-18  62832bb272  Support cuda graph for DP attention (#2061)
2024-11-20  699384cb01  Set schedule policy more conservative for DP attention (#2096)
2024-11-23  cbedd1db1d  [router] cache-aware load-balancing router v1 (#2114)
2024-11-24  4b0a1c9365  Replace prob based with threshold based load balancing  (#2170)
2024-12-06  3d32e4a32c  Resubmit MoE-EP (#2371)
2024-12-11  2e4a5907c9  [router] Release router 0.1.0 with dynamic scaling and fault tolerance (
2024-12-12  e3b3acfa6f  Rename rust folder to sgl-router (#2464)
2024-12-24  e835a50021  Reorg moe code (#2563)
```

## DP attention: made to measure for MLA (#1970) {#dp-attention为-mla-量身定做1970}

[The previous chapter](mla-compile.md) covered MLA's compression of each layer's KV into one latent vector. It has a side effect on parallelism: ordinary multi-head attention can be split by KV head across cards (8 KV heads on 8 cards), while MLA has only one latent vector, so under TP **every card has to hold a complete copy of the KV** and 8-card TP's KV memory efficiency is 1/8. Ke Bao's #1970 "Support DP MLA" of 16 November introduced `--enable-dp-attention`: the attention part becomes data-parallel — each rank has its own scheduler, its own requests and its own KV; the MoE and dense FFN layers are parallelised as before (expert sharding or TP), with every rank's tokens all-gathered before entering those layers and scattered back by rank afterwards.

So the ranks under DP attention are both independent (each scheduling its own) and coupled (going through the MoE layer together every step). v0.4.0's `Scheduler` gains one step after forming its local batch:

```python title="python/sglang/srt/managers/scheduler.py @ v0.4.0 L436-475" linenums="436"
    def prepare_dp_attn_batch(self, local_batch: ScheduleBatch):
        # Check if other DP workers have running batches
        if local_batch is None:
            num_tokens = 0
        elif local_batch.forward_mode.is_decode():
            num_tokens = local_batch.batch_size()
        else:
            num_tokens = local_batch.extend_num_tokens

        local_num_tokens = torch.tensor([num_tokens], dtype=torch.int64)
        global_num_tokens = torch.empty(self.tp_size, dtype=torch.int64)
        torch.distributed.all_gather_into_tensor(
            global_num_tokens,
            local_num_tokens,
            group=self.tp_cpu_group,
        )

        if local_batch is None and global_num_tokens.max().item() > 0:
            local_batch = self.get_idle_batch()

        if local_batch is not None:
            local_batch.global_num_tokens = global_num_tokens.tolist()

            # Check forward mode for cuda graph
            if not self.server_args.disable_cuda_graph:
                forward_mode_state = torch.tensor(
                    (
                        1
                        if local_batch.forward_mode.is_decode()
                        or local_batch.forward_mode.is_idle()
                        else 0
                    ),
                    dtype=torch.int32,
                )
                torch.distributed.all_reduce(
                    forward_mode_state,
                    op=torch.distributed.ReduceOp.MIN,
                    group=self.tp_cpu_group,
                )
                local_batch.can_run_dp_cuda_graph = forward_mode_state.item() == 1
```

Three things: exchange every rank's token count for this step with an `all_gather` (the gather's shape has to agree); make an `IDLE` batch to keep step when this rank has nothing and others do — a collective tolerates no absence; and `all_reduce(MIN)` to confirm every rank is decoding before a CUDA graph can be used (added by #2061 on 18 November). #2096 of 20 November made the scheduling policy under DP attention more conservative — the ranks' batch sizes affect each other, so the estimate needs headroom.

The v0.4 blog's number: 1.9 times the decode throughput for the DeepSeek family under DP attention, which comes from the KV no longer being duplicated and a larger batch fitting. This is an extension of [chapter six](../service/processes.md)'s "every rank schedules again": under DP attention the ranks schedule **different** requests, but the all-gather each step keeps the collectives consistent.

![Figure: how the KV is spread under TP and under DP attention with MLA](../assets/figures/sgl-dp-attention.svg){.aig-svg}

## EP: experts across cards (#2371) {#ep专家切到不同的卡2371}

The other parallel route for a MoE model is expert parallelism: each card holds some of the experts and tokens go to the card holding the expert the routing chose. SGLang's EP entered the main line on 6 December with #2371 "Resubmit MoE-EP" (an earlier merge had been reverted), and #2563 of 24 December tidied the MoE code into `ep_moe/` and `fused_moe_triton/` under `layers/moe/`: the former is EP's expert dispatch and grouped GEMM, the latter a Triton fused MoE adapted from vLLM. EP at this point is still the plain version of "all-to-all over NCCL, expert GEMMs in Triton"; only after DeepEP arrived in March 2025 did [chapter 18](../scale/large-ep.md)'s large-scale EP become possible. From that month on, DP attention plus EP became the standard shape for deploying DeepSeek.

## The Rust router: an approximate copy of the tree (#1790 → #2114) {#rust-路由器树的近似副本1790--2114}

The paper's idea of a router keeping a meta tree ([chapter one](../origins/paper.md)) got its implementation on 28 October: #1790 "Add a rust-based router" — a standalone HTTP service with clients in front and several SGLang workers behind. The first version has only round robin and random; #1893 of 4 November added a radix tree, #1934 of 10 November "cache-aware DP - approx tree", and #2114 of 23 November "cache-aware load-balancing router v1". v0.4.0's `rust/src/router.rs` explains the policy in a long comment:

```rust title="rust/src/router.rs @ v0.4.0 L15-27,44-92"
pub enum Router {
    RoundRobin {
        worker_urls: Vec<String>,
        current_index: AtomicUsize,
    },
    Random {
        worker_urls: Vec<String>,
    },
    CacheAware {
        /*
            Cache-Aware Load Balancing Router

            This router combines two strategies to optimize both cache utilization and request distribution:
...
            This strategy maintains an approximate radix tree for each worker based on request history,
            eliminating the need for direct cache state queries. The tree stores raw text characters
            instead of token IDs to avoid tokenization overhead.

            Process:
            a. For each request, find the worker with the highest prefix match
            b. If match rate > cache_threshold:
            Route to the worker with highest match (likely has relevant data cached)
            c. If match rate ≤ cache_threshold:
            Route to the worker with smallest tree size (most available cache capacity)
            d. Background maintenance:
            Periodically evict least recently used leaf nodes to prevent memory overflow

            2. Load Balancing (Shortest Queue)
            -------------------------------------------
            This strategy tracks pending request counts per worker and routes new requests
            to the least busy worker when the system is detected to be imbalanced.

            Configuration Parameters:
            ------------------------
            1. cache_threshold: (float, 0.0 to 1.0)
            Minimum prefix match ratio to use highest-match routing.
            Below this threshold, routes to worker with most available cache space.

            2. balance_abs_threshold: (integer)
            Absolute difference threshold for load imbalance detection.
            System is potentially imbalanced if (max_load - min_load) > abs_threshold

            3. balance_rel_threshold: (float)
            Relative ratio threshold for load imbalance detection.
            System is potentially imbalanced if max_load > min_load * rel_threshold
            Used in conjunction with abs_threshold to determine final imbalance state.

            4. eviction_interval_secs: (integer)
            Interval between LRU eviction cycles for the approximate trees.

            5. max_tree_size: (integer)
            Maximum nodes per tree. When exceeded, LRU leaf nodes are evicted
            during the next eviction cycle.
        */
        worker_urls: Vec<String>,
        tree: Arc<Mutex<Tree>>,
        running_queue: Arc<Mutex<HashMap<String, usize>>>,
        processed_queue: Arc<Mutex<HashMap<String, usize>>>,
        cache_threshold: f32,
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
        _eviction_thread: Option<thread::JoinHandle<()>>,
    },
```

One approximate tree per worker (`tree.rs`, a radix tree in Rust supporting nodes shared across tenants and workers): when a request goes to a worker, its text is inserted into that worker's tree; on a query the prefix is matched against every tree, and a match rate above `cache_threshold` picks the worker that matched most; but if the busiest and idlest workers' loads differ by more than `balance_abs_threshold` and their ratio exceeds `balance_rel_threshold`, it gives up on the hit rate and goes to the idlest (#2170 of 24 November changed a probabilistic balance into this threshold-based one). A background thread evicts LRU every `eviction_interval_secs`, and the tree is only an approximation of the worker's real cache. The blog's numbers: 1.9 times the throughput and 3.8 times the hit rate (on multi-turn conversational traffic).

Version 0.1.0 was released on 11 December: workers added and removed on the fly (`/add_worker`, `/remove_worker`), health checks and retry-based fault tolerance; the directory was renamed from `rust/` to `sgl-router/` the next day. Its size:

```bash title="router-size.sh"
REF=${REF:-29f6d408c0}
for spec in "v0.4.0 rust" "v0.4.6 sgl-router" "v0.5.0rc0 sgl-router" "$REF sgl-model-gateway"; do
  set -- $spec
  printf '%-11s %-18s %4d 个文件，其中 .rs %3d 个，共 %6d 行 Rust\n' "$1" "$2" "$(git ls-tree -r --name-only "$1" -- "$2" | wc -l)" "$(git ls-tree -r --name-only "$1" -- "$2" | grep -c '\.rs$')" "$(git ls-tree -r --name-only "$1" -- "$2" | grep '\.rs$' | while read -r f; do git show "$1:$f"; done | wc -l)"
done
```

```text title="output"
v0.4.0      rust                 17 个文件，其中 .rs   5 个，共   2169 行 Rust
v0.4.6      sgl-router           18 个文件，其中 .rs   4 个，共   2660 行 Rust
v0.5.0rc0   sgl-router           50 个文件，其中 .rs  34 个，共  18918 行 Rust
29f6d408c0  sgl-model-gateway   403 个文件，其中 .rs 252 个，共  94747 行 Rust
```

## Design trade-offs {#设计取舍}

| Decision | The reason | The price |
| --- | --- | --- |
| DP attention inside one process group | attention independent, MoE merged, so the shapes have to be synchronised every step | the ranks wait for each other; the scheduling has to be more conservative; IDLE batches keep step |
| EP first with a NCCL all-to-all plus Triton | usable quickly | the communication and the computation are serial, not enough at scale (which DeepEP solves) |
| The router as a standalone Rust service | deployable across machines, high concurrency, fast tree operations | two load balancers (the engine's DP controller plus the external router); the approximate tree drifts |
| Threshold balancing ahead of hit rate | keeps a hot worker from overloading | the hit rate drops when the load is uneven |

## What happened afterwards {#后来怎么样了}

- DP attention went from MLA-only to a general option, with various combinations of `--dp-size` and `--enable-dp-attention` and an optimised LM head under DP added in 2025, and `moe_dense_tp_size` controlling how the dense layers are parallelised.
- EP: DeepEP in 2025-03, EPLB and dual-batch overlap in 2025-05, elastic EP in 2025-10 ([chapter 18](../scale/large-ep.md)).
- The router: the standalone `sgl-router/` kept growing from 2025-04, gaining PD-disaggregated routing, a dependency-injection restructuring of its policies (#7987) and tool parsing, and was renamed `sgl-model-gateway/` in 2025-12 (chapter 21).

## Exercises {#练习}

**1. Why IDLE is necessary.** If `get_idle_batch` were removed and a rank with nothing to do simply skipped the step, where would it hang?

??? success "Answer"
    The MoE layer's all-gather or all-to-all needs every rank in the process group; an absent rank leaves the others waiting forever (the NCCL collective hangs). An IDLE batch lets it take part with zero tokens.

**2. The approximate tree's drift.** When do the router's tree and a worker's real tree disagree? Give a case of "the router thinks it hits and it does not" and one of the opposite.

??? success "A way to approach it"
    A worker evicted a node under memory pressure before the router's LRU cycle came round (thinks it hits and does not); a prefix was inserted on a worker by another request that did not go through the router (hits in fact, unknown to the router). The approximation only costs a suboptimal route and never correctness.

**3. Two layers of load balancing.** Draw the hierarchy "router → N engine instances → each instance's DP controller → the DP replicas" and explain why DP attention's replicas are not suited to being scheduled by the external router directly.

??? success "A way to approach it"
    DP attention's ranks synchronise every step as one tightly coupled process group, and must be dispatched to step by step by one controller; the external router sees only an instance's HTTP entry point.

!!! interview "How to explain it"
    "How is an MLA plus MoE model like DeepSeek deployed?" — Say "attention under DP, experts under EP", explain why TP wastes MLA's KV, and then say what DP attention has to synchronise each step (the token count, the forward mode, the IDLE batches). If pressed on several instances, cover cache-aware routing: the approximate tree and the trade-off between the hit-rate and the load thresholds. Being able to say all of this landed within one quarter, from 2024-11 to 12, and that the later DeepEP and EPLB build on it, is a point in your favour.

## Summary {#小结}

- [x] #1970 (2024-11-16) DP attention: under MLA, attention goes data-parallel and the MoE is merged, with the token counts all-gathered each step, IDLE batches keeping step, and an all-reduce deciding the CUDA graph; the blog's 1.9 times decode throughput.
- [x] #2371 brought EP into the main line and #2563 tidied `layers/moe/`; large-scale EP had to wait for 2025's DeepEP.
- [x] The Rust router went from round robin (#1790) to cache-aware routing over approximate radix trees (#2114) to 0.1.0's dynamic scaling and fault tolerance; 1.9 times the throughput, 3.8 times the hit rate.
