# CacheManager: allocation, release and admission control

<p class="lead">At any moment every page of the KV pool is in one of three states: free, owned by one request, or in the prefix cache (possibly shared by several requests). <code>CacheManager</code> moves pages between those states and answers the question the scheduler cares about most: if we admit one more request, does its KV still fit? mini-sglang's answer is conservative, reserving for the worst case, which is why it never needs preemption.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. vLLM preempts running requests when KV runs short. Why does mini-sglang not need to? What does that cost?
    2. Why does `available_size` count the evictable part of the prefix cache?
    3. Why does prefix matching stop at the prompt's second-to-last token?
    4. With a page size of 4, a request has `cached_len=6` and `device_len=9`. How many pages does this step allocate?

??? success "Answers (try it yourself first, then expand)"
    1. Admission control reserves KV for the worst case (the rest of the prompt plus `max_tokens`), so an admitted request is guaranteed to run to completion and preemption is never needed. The cost is that most of the reservation goes unused, fewer requests run at once, and concurrency is lower.
    2. Any node in the prefix cache that is not locked can be evicted at any moment to make room. It occupies memory but counts as available to a new request; leaving it out underestimates the available space and rejects requests for nothing.
    3. Every step has to compute at least one token: even with the whole prompt cached, the hidden state at the last position is still needed for the logits. So matching stops at the second-to-last token, keeping `cached_len < device_len`.
    4. It allocates the pages `[ceil(6/4), ceil(9/4)) = [2, 3)`, so one page: positions 6 and 7 are still in the already allocated page 1 (covering positions 4-7), and only position 8 needs a new page 2.

**Files you will write**: `scheduler/cache.py`; and fill in `PrefillAdder._try_allocate_one` in `scheduler/prefill.py`.

## The free-page list {#空闲页列表}

@@code python/minisgl/scheduler/cache.py:CacheManager.__init__@@

`free_slots` holds **the location of each free page's first token** (with a page size of 4 that is `[0, 4, 8, …]`), not the page number. That matches chapter 4's "the page table stores token locations": an allocated page only needs the offset within it added to give a token location, and freeing takes every page_size-th token location to get back to the page's start.

@@code python/minisgl/scheduler/cache.py:CacheManager._allocate@@

Allocating is slicing from the front of `free_slots`. When there are not enough free pages it first has the prefix cache evict some (chapter 9), adds what that frees, and then allocates.

## Allocating pages for this step's tokens {#为本轮的-token-分配页}

@@code python/minisgl/scheduler/cache.py:CacheManager.allocate_paged@@

For each request the pages already allocated are `[0, ceil(cached_len / page_size))` and the pages needed after this step are `[0, ceil(device_len / page_size))`, and the difference is what gets allocated. With a page size of 4, `cached_len=6` and `device_len=9`: there are 2 pages already (covering positions 0-7), 3 are needed, so 1 is allocated. During decode most steps need no new page, with one allocation every page_size steps.

Every request's pages are allocated in one go and written into the page table in one go:

@@code python/minisgl/scheduler/cache.py:_write_page_table@@

All the `(row, column)` indices are computed on the CPU first and then written with a single indexed assignment. On a GPU that is one asynchronous copy and one kernel, rather than one operation per request per token.

## Freeing, and freeing lazily {#释放与懒释放}

@@code python/minisgl/scheduler/cache.py:CacheManager.lazy_free_region@@

When a batch's results are handled, many requests may finish at once and each has pages to free. One `torch.cat` on `free_slots` per request would be N copies. Inside `lazy_free_region`, `_free` only records what to free, and the whole lot is concatenated once on exit. The scheduler's `_process_last_data` is wrapped entirely in that region.

## Admission control {#准入控制}

The second half of the animation at the start of chapter 7 shows the rules below: reserve for the worst case, and never preempt.

Whether a request can be admitted is decided by `PrefillAdder._try_allocate_one`:

@@code python/minisgl/scheduler/prefill.py:PrefillAdder._try_allocate_one@@

The criterion is the **worst case**: the KV this request still needs is the part of the prompt that missed the cache plus `max_tokens`. Together with `reserved_size`, what is already promised to others, it must not exceed `available_size` (the free pages plus the evictable part of the prefix cache).

`reserved_size` comes from two places:

- the space the decoding requests may still need before this batch was assembled, namely `DecodeManager.inflight_tokens`, the sum of each request's `remain_len` plus one extra page of slack per request (the last page may be only partly used);
- the requests already admitted into this batch (`reserved_size += remain_len + output_len` in `_add_one_req`).

The check runs twice with `lock(handle)` in between: before the lock the prefix that was hit is evictable in the prefix cache and counts towards `available_size`; after the lock it is protected and `available_size` shrinks accordingly. Failing the first check can return immediately and save a lock-then-unlock; the second check is the accurate one.

Because every request reserves its worst case at admission, a running request never stalls for want of KV, so mini-sglang needs neither preemption nor recomputation. The price is being conservative: a request with a large `max_tokens` that hits EOS early leaves most of its reservation unused, which lowers how many requests can run at once. vLLM takes the opposite line, admitting optimistically and, when KV runs short, preempting the most recently arrived request, freeing its KV and recomputing later.

Here is an example where the KV pool holds only 64 tokens:

@@code examples/ch08_cache_manager.py@@

@@output ch08_cache_manager@@

The first four requests need 13 + 13 + 12 + 23 = 61 tokens in the worst case, which fits; the fifth needs another 13, which is over 64, so it waits. Notice that the pool has free pages the whole time until all four finish (7 pages at the end), which is what conservative reservation costs. Once the first four are done the space comes back and the fifth is admitted. When everything has finished all 64 pages are back in the free list and the integrity check passes.

## Handing the KV to the prefix cache {#把-kv-交给前缀缓存}

When and how a request's KV goes to the prefix cache is the most delicate piece of logic in `CacheManager`:

@@code python/minisgl/scheduler/cache.py:CacheManager.cache_req@@

It is called at two moments: right after prefill (`finished=False`, the prompt's KV enters the cache while the request keeps running) and when the request finishes (`finished=True`). The comments split a request's KV into four regions:

@@diagram cache-regions cache_req handles a request's KV in four regions@@

Where does the middle region, "computed here but already in the cache", come from? Two requests with the same prefix arrive together and prefill together, so neither hits the cache; the first to finish inserts the prefix, and when the second tries to insert it finds it already there. Its own copy is redundant and has to be freed, or it leaks. With the naive cache `insert_prefix` keeps nothing, so a request's pages are all freed when it ends. Once chapter 9 implements the radix cache, all four regions really show up.

## The integrity check {#完整性检查}

@@code python/minisgl/scheduler/cache.py:CacheManager.check_integrity@@

The scheduler runs a check when it goes idle (`run_when_idle`): at that moment there are no running requests, so every page is either free or in the cache, and the two must add up to the total. Anything forgotten (or freed twice) shows up right here.

!!! upstream "The official implementation"
    - @@upstream scheduler/cache.py:CacheManager@@
    - admission control: @@upstream scheduler/prefill.py:PrefillAdder._try_allocate_one@@
    - the reservation: @@upstream scheduler/decode.py:DecodeManager.inflight_tokens@@

    The official repository's `tests/core/test_cache_allocate.py` specifically tests that with a page size above 1 the pages obtained by "evict, then allocate" are page-aligned and do not overlap.

## Tests {#测试}

@@code tests/test_ch08_cache_manager.py:test_allocate_paged_writes_page_aligned_token_positions@@

@@code tests/test_ch08_cache_manager.py:test_integrity_check_detects_leak@@

!!! interview "How to explain it"
    On KV management: the free-page list stores a start location per page, each step allocates only the pages `[ceil(cached_len / page size), ceil(device_len / page size))`, and decode allocates once every "page size" steps; when pages run short the prefix cache evicts, which is why the available space has to count the cache's evictable part; and frees are batched (lazy free). Admission control reserves for the worst case (the rest of the prompt plus max_tokens), so preemption is never needed, at the cost of lower concurrency than vLLM, which admits optimistically and preempts and recomputes when short. Prefix matching stops at the prompt's second-to-last token so that at least one token is computed and there are logits. When a request ends its KV goes to the prefix cache region by region, every page has exactly one owner, and the idle-time integrity check catches leaks.

## Exercises {#练习}

1. `inflight_tokens` reserves an extra `page_size - 1` tokens per running request. Give an example of what goes wrong with a page size of 16 if it does not.
2. Implement an "optimistic" admission policy: reserve only half of `max_tokens`, and when KV runs short preempt the most recently admitted request (free its resources and put it back at the head of the waiting queue). What has to change? When the preempted request prefills again, how does the prefix cache help?
3. If `cache_req` forgot to free the `[old.cached_len, cached_len)` region, where would it be noticed?

??? success "Answers"
    1. A request's KV is allocated in pages, while `remain_len` is counted in tokens and the next allocation is a whole page. A request with `device_len=17` and `remain_len=1` still needs a whole new page of 16 tokens. Add up several such requests and a token-based estimate underestimates the pages actually needed, so an allocation can fail.
    2. Check the free pages before `allocate_paged` in `_prepare_batch` and, when short, pick a request from the decode set: `table_manager.free`, `cache_manager.cache_req(finished=True)`, and wrap it back into a `PendingReq` at the head of the queue. The tokens it has already generated have to be spliced into `input_ids`. Because `cache_req` handed its KV to the radix cache, most of the prefix hits directly when it prefills again (unless it was evicted), so recomputation is cheap. This is exactly what production SGLang does.
    3. Those pages would be neither in the free list nor owned by the prefix cache (which holds another copy of the same content), so nobody would hold them once the request ended. The next `check_integrity` while the scheduler is idle would find that free pages plus cached pages do not add up to the total and raise. Chapter 9's end-to-end test exercises this path with two requests sharing a prefix prefilling at once.

## Summary {#小结}

- [x] `free_slots` stores a start location per page; allocation slices from the front, eviction from the prefix cache covers a shortfall, and frees are batched inside `lazy_free_region`.
- [x] Each step allocates only the pages `[ceil(cached_len/ps), ceil(device_len/ps))`, so decode allocates once every page_size steps.
- [x] Admission control reserves for the worst case (the rest of the prompt plus max_tokens), so there is never any preemption, at the cost of lower concurrency.
- [x] `cache_req` handles a request's KV in four regions, so every page has exactly one owner, and the idle-time integrity check catches any leak.
