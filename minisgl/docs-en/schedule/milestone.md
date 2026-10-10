# Stage 2 milestone: the scheduler takes over

<p class="lead">At the end of stage 1 you still fill the batch's fields yourself: which requests enter this step, which pages hold their KV, where their positions start. The five chapters of stage 2 turn those lines into a scheduler — continuous batching, admission control, page allocation and release, the radix cache, chunked prefill, overlap scheduling — leaving a single offline interface, <code>LLM.generate</code>. This page runs the end state first: three sets of numbers, each for something the skeleton could not do.</p>

**What you hold at the end of this stage**

- `LLM.generate(prompts)`: any number of requests at once, of any lengths; the ones that finish leave the batch, new ones join at any step;
- A shared system prompt has its KV computed once and reused by later requests;
- A long prompt is cut into fixed-size chunks that other requests can slip between;
- The CPU's preparation of the next step hides behind the computation (overlap scheduling; this page turns it off so the numbers reflect the scheduler alone).

## Run it first {#先跑起来}

```bash
cd minisgl && python examples/stage2_milestone.py
```

@@code examples/stage2_milestone.py@@

@@output stage2_milestone@@

## Reading the numbers {#读这几个数字}

**8 requests in 1.29 seconds, 99.5 tokens/s.** Step 0 did 26.8; stage 1's hand-assembled batch of 4 did 68.9. Two things stack: the batch is larger (8 requests per step, one read of the weights yields 8 tokens), and nobody fills fields by hand any more — at every step inside `generate`, the scheduler does what the few lines in stage 1's `step()` did: admission, page allocation, computing `positions` and `out_loc`, reclaiming the pages of finished requests. In a real service requests arrive one after another with different lengths, and the point of continuous batching is that **the batch's membership keeps changing** while the computation never stops.

**A shared system prompt: prefill drops from 231 tokens to 73, and the last three requests from 345 ms to 190 ms.** The four questions share a 158-token system prompt. With the naive cache every request recomputes it; with the radix cache, once the first request has computed the prefix it stays in the tree, and the next three compute only their own question. That is where the severalfold throughput difference in multi-turn and shared-system-prompt workloads comes from — not one of the 158 × 3 skipped tokens was wasted.

**A 302-token prompt in 4 steps: 96, 96, 96, 14.** Without chunked prefill a long prompt takes over the engine for a whole step and every decoding request waits for it; with chunks, each step computes at most `max_extend_tokens` tokens, decode requests slip in between, and a slightly longer time to first token buys steady per-token latency for everyone else.

## What the hand-written lines became {#手工的那几行变成了什么}

| What you did by hand in stage 1's `step()` | Stage 2 | Where |
| --- | --- | --- |
| decide which requests enter this step's batch | admission control, the continuous-batching main loop | [The scheduler skeleton and the offline interface](scheduler.md) |
| `page_table[row, :256] = arange(...)` — "allocating" KV by hand | `CacheManager`: allocate on demand, reclaim on finish, wait when full | [CacheManager](cache-manager.md) |
| `cached_len=0` — every request from scratch | the radix cache: match prefixes, reuse KV, reference counts, LRU eviction | [Radix Cache](radix-cache.md) |
| the whole prompt in one prefill step | chunked prefill: at most `max_extend_tokens` tokens per step | [Chunked prefill](chunked-prefill.md) |
| prepare the next step after this one finishes | overlap scheduling: prepare the next step while the GPU is still on this one | [Overlap scheduling](overlap.md) |

## How to read these five chapters {#怎么读这五章}

Read [the scheduler skeleton](scheduler.md) first — it defines the main loop, and the other four chapters add to that loop. At [Radix Cache](radix-cache.md), flip this page's `cache_type` back and forth and watch the "prefill actually computed" number; at [chunked prefill](chunked-prefill.md), change `max_extend_tokens` and watch the step count and tokens per step. After the last chapter, [overlap scheduling](overlap.md), delete this page's first line `ENV.DISABLE_OVERLAP_SCHEDULING.value = True` and rerun: on a CPU the gain is small (there is no GPU computing at the other end), but on a real card it is the last stretch of throughput.

!!! abstract "Checkpoint: do these before stage 3"
    - [ ] Draw a request's state changes from entering the queue to releasing its KV pages, marking each scheduler decision;
    - [ ] Explain why the radix tree splits nodes and what reference counting protects against;
    - [ ] Say what chunked prefill gives up and what it buys;
    - [ ] Set this page's `max_running_req` to 2 and explain why throughput drops, and to what.

## Summary {#小结}

- [x] Stage 2 turns the hand-filled batch lines of stage 1 into a scheduler, leaving only `LLM.generate`.
- [x] Throughput goes from 68.9 to 99.5 tokens/s: a larger batch whose membership changes while computation never stops.
- [x] The radix cache computes a shared prefix once: 231 prefill tokens become 73.
- [x] Chunked prefill stops long prompts from monopolising the engine: 302 tokens in 4 steps, with other requests in between.
