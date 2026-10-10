# The radix cache: reusing prefixes across requests

<p class="lead">In a multi-turn chat the N-th prompt contains everything from the first N−1 turns; many applications put the same system prompt in front of every request; an agent carries an ever longer history at every step. These requests share a long opening, and so does their KV. SGLang's RadixAttention remembers every prefix it has computed in a radix tree, and a new request first looks for the longest match and reuses whatever it hits. It is SGLang's most characteristic design, and mini-sglang implements it in 237 lines.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. How does a radix tree differ from an ordinary trie?
    2. When can a node be evicted? Why only leaves?
    3. Does matching a prefix modify the tree?
    4. With a page size of 16, how much of a 30-token prompt can be cached?

??? success "Answers (try it yourself first, then expand)"
    1. A trie stores one token per node; a radix tree compresses a run of tokens with no branching into a single node that holds a stretch of tokens and their KV locations, so matching compares a whole stretch at once and there are far fewer nodes.
    2. When it is not locked by a running request (`ref_count == 0`) and it is a leaf. An internal node is a prefix of other nodes, and evicting it would make its children unreachable; evicting only leaves keeps the tree a valid prefix structure (and a parent may become the new leaf once its child is gone).
    3. Yes: when the match ends in the middle of a node, that node is split in two so the handle points exactly at where the match ended.
    4. 16 tokens: insertion only caches whole pages, so only the first page of the 30 tokens goes into the cache.

**Files you will write**: `kvcache/radix_cache.py`, `fast_compare_key` in `kernel/torch_ops.py`, and the `radix` registration in `kvcache/__init__.py`.

@@tree@@

**This step's main**: `examples/ch09_radix.py` — it uses only the files above; `python tools/steps.py check` rebuilds this tree chapter by chapter and runs it.

@@video radix Animation: insertion, splitting, matching, locking and eviction in the radix cache (about 2 minutes, Chinese narration and subtitles)@@

## The shape of the tree {#树的结构}

Each node of the tree holds a stretch of tokens (`key`) and where those tokens' KV lives in the pool (`value`, token-level locations like the page table). Walking from the root to a node and concatenating the keys gives a cached prefix, and concatenating the values gives its KV locations.

Unlike a trie that creates a node per token, a radix tree compresses "a run with no branching" into one node. A 4000-token system prompt is 4000 nodes in a trie and one node in a radix tree, compared in a single stretch.

@@code python/minisgl/kvcache/radix_cache.py:RadixTreeNode@@

- `children` is indexed by **the first token** of a child's key (when page size is 1) or **the tuple of the first page's tokens**: under one node no two children can start the same way, so a single dictionary lookup finds the only child that could match;
- `ref_count`: how many running requests depend on this node. It cannot be evicted while this is above 0;
- `timestamp`: when it was last visited, for LRU eviction;
- `split_at(pos)`: cuts the node in two at `pos`, inserting the new first half between the parent and itself.

## Walking the tree {#在树上走}

Matching and insertion both start from the same function:

@@code python/minisgl/kvcache/radix_cache.py:RadixPrefixCache._tree_walk@@

Starting at the root, look up a child by the start of what is left of the input; if one is found, compare how long a prefix that child's key shares with the input (`fast_compare_key`, rounded down to a page). A complete match walks on; a partial match splits the node at the matching point and returns the first half.

So **matching does change the shape of the tree**. It adds and removes nothing from the cache, but it may split one node into two so the returned handle points exactly at "where the match ended". That matters for the locking that follows, because what gets locked is precisely what was hit, no more and no less.

`fast_compare_key` finds the position of the first difference; upstream implements it in C++ with `std::mismatch` (bound through tvm-ffi), and we use PyTorch:

@@code python/minisgl/kernel/torch_ops.py:fast_compare_key@@

## Matching, inserting, locking {#匹配插入加锁}

@@code python/minisgl/kvcache/radix_cache.py:RadixPrefixCache.match_prefix@@

@@code python/minisgl/kvcache/radix_cache.py:RadixPrefixCache.insert_prefix@@

Insertion only caches whole pages (`align_down`): with a page size of 16, only the first 16 tokens of a 30-token prompt enter the cache. After walking to the end of the existing prefix, the rest is hung on as a new leaf. The `cached_len` in the return value is "how much was already in the tree before this insertion", which the caller (`CacheManager.cache_req`) uses to free its own duplicate pages.

@@code python/minisgl/kvcache/radix_cache.py:RadixPrefixCache.lock_handle@@

Locking walks from the handle's node all the way to the root, adding one to each node's `ref_count` along the way, because a request depends on the whole prefix and not just its last stretch. When a node's `ref_count` goes from 0 to 1, its length moves from "evictable" to "protected". The two counts together are the total number of tokens in the cache, and `CacheManager.available_size` is exactly "free pages plus evictable".

## Eviction {#淘汰}

@@code python/minisgl/kvcache/radix_cache.py:RadixPrefixCache.evict@@

Only a **leaf** with `ref_count` 0 can be evicted: evicting an internal node would strip its children of their prefix and make them unreachable. Every evictable leaf goes into a min-heap by timestamp, and the least recently visited is popped each time; if its parent thereby becomes a leaf and is unlocked, that goes into the heap too. Eviction works a node at a time, so it may free more than was asked for.

## An example {#看一个例子}

@@diagram radix-tree insertion, matching (splitting and locking), LRU eviction@@

@@code examples/ch09_radix.py@@

@@output ch09_radix@@

- After inserting `[1,2,3,4]` and `[1,2,5,6]`, the common prefix `[1,2]` becomes a node with two branches below it;
- Matching `[1,2,3,9]` hits 3 tokens: `[3,4]` splits into `[3]` and `[4]`, and the handle points at `[3]`. After locking, `[1,2]` and `[3]` have `ref_count` 1, so 3 tokens are protected and 5 are evictable;
- Asked to evict 2 tokens: the evictable leaves are `[4]`, `[5,6]` and `[7,8]`. The match only refreshed the timestamps of `[1,2]` and `[3]`, so the split-off `[4]` kept its original insertion time and goes first; one is not enough, so the next oldest `[5,6]` follows, freeing 3 in all;
- Asked to evict 1 more: only `[7,8]` is left to evict. The locked `[1,2]` and `[3]` stay.

End to end: four requests share one system prompt, the first prefills it in full and the other three compute only their own questions, which brings the prefill from 231 tokens down to 73.

## Fitting into the request lifecycle {#与请求生命周期的配合}

Looking back at `cache_req` from chapter 8, all four of its regions now really exist:

1. At admission `match_req` finds the prefix that hit and locks it (matching up to the second-to-last token, so at least one token is computed);
2. At the end of prefill, `cache_req(finished=False)` inserts the whole prompt (page-aligned) into the tree, unlocks the old handle and locks the new one; if another request inserted the same prefix in the meantime, this request's duplicate pages are freed right away;
3. When the request finishes, `cache_req(finished=True)` inserts the generated text too (the next turn of the conversation will want it), unlocks, and frees the tail that does not fill a page.

A finished request's KV is not freed but stays in the tree as "evictable", and is only really evicted when the space is needed. So memory is always used to cache prefixes as much as it can be.

!!! upstream "The official implementation"
    - the whole file: @@upstream kvcache/radix_cache.py@@
    - walking the tree: @@upstream kvcache/radix_cache.py:RadixPrefixCache._tree_walk@@
    - eviction: @@upstream kvcache/radix_cache.py:RadixPrefixCache.evict@@
    - the C++ `fast_compare_key`: `kernel/csrc/src/radix.cpp`

!!! diff "Difference from upstream: the integrity check"
    The official `RadixPrefixCache.check_integrity` is empty (`pass`). Ours walks the whole tree, recounts the evictable and protected lengths, reconciles them against the counters, and checks the invariant that a child is never locked more times than its parent. The scheduler calls it while idle, so any miscount shows up immediately.

@@code python/minisgl/kvcache/radix_cache.py:RadixPrefixCache.check_integrity@@

## Tests {#测试}

@@code tests/test_ch09_radix.py:test_lock_protects_from_eviction_and_lru_order@@

@@code tests/test_ch09_radix.py:test_shared_prefixes_are_reused_and_outputs_unchanged@@

Running the same 5 prompts a second time, each request has to recompute only its last token, the output matches Hugging Face exactly, and the memory integrity check passes.

!!! interview "How to explain it"
    On prefix caching: a radix tree is a compressed trie whose node holds a stretch of tokens and their KV locations, with children indexed by the first token (or first page) so a match compares a whole stretch at once; a match can end mid-node, and the node is then split so the handle points exactly at the match, which is why matching also modifies the tree. Only whole pages are cached (with a page size of 16, at most 16 of a 30-token prompt). A running request locks the whole path by incrementing reference counts, and only unlocked leaves are evicted in LRU order; a finished request's KV stays in the tree and is evicted only when space is needed. Against vLLM's hashed-block prefix cache: a radix tree matches at any length and handles multi-turn chat naturally, while hashed blocks look up whole blocks in a table, which is simpler to implement and easier to make distributed or multi-tier.

## Exercises {#练习}

1. vLLM uses "hashed blocks" for prefix caching: each KV block's hash comes from the previous block's hash plus this block's tokens, and lookup is a dictionary. What are the pros and cons against a radix tree? (See the [prefix-caching chapter of the Inference Systems handbook](serving://engine/prefix-cache/).)
2. Eviction currently walks the whole tree each time to collect evictable leaves. With a large tree of hundreds of thousands of nodes that is a bottleneck. How would you maintain an always-valid "set of evictable leaves"?
3. Design a test that puts two requests with the same prefix in one prefill batch and verifies that `cache_req` frees the duplicate pages correctly.

??? success "Answers"
    1. Hashed blocks: simple to implement, lookup is O(blocks) dictionary accesses, naturally block-aligned, and easy to share across processes (KV events); but the hashing starts from the beginning every time and matching is only block-granular. A radix tree: compares a whole stretch at once, stores a shared prefix only once, and knows the dependencies when evicting; but it is more complex, and splitting and merging nodes needs careful bookkeeping.
    2. Keep a set (or a heap ordered by timestamp with lazy deletion): add a node when it becomes a leaf with `ref_count` 0, and remove it when it is locked, gains a child or is evicted. SGLang 0.5.20's `RadixCache` does exactly this, maintaining an `evictable_leaves` set updated by `_update_leaf_status` after every lock, unlock, insertion and deletion.
    3. Use "The capital of France is" and "The capital of France is a city that" from `PROMPTS`: neither hits in the same prefill batch, the first to finish inserts the prefix, and the second sees `cached_len > 0` on insertion and takes the "free the duplicate pages" branch. A passing `check_integrity()` afterwards shows there is no leak, and `test_shared_prefixes_are_reused_and_outputs_unchanged` already covers this.

## Summary {#小结}

- [x] A radix-tree node holds a stretch of tokens and their KV locations; children are indexed by the first token (or first page), and a match compares a whole stretch at once.
- [x] A match may split a node so the handle points exactly at where it ended; insertion caches only whole pages and hangs the new part on as a leaf.
- [x] Locking adds one to `ref_count` along the whole path; only unlocked leaves are evicted in LRU order, a node at a time.
- [x] A finished request's KV stays in the tree and is evicted only when space is needed; with a shared system prompt the prefill work drops sharply.
