# RadixAttention's first version: 220 lines of radix tree, line by line

<p class="lead">RadixAttention is SGLang's signature, and its first version is only 220 lines: a <code>TreeNode</code>, a character-by-character <code>match</code>, recursive matching and insertion, an <code>evict</code> that drops leaves by last access time, and a pair of reference-counting functions. This chapter reads <code>22085081bb</code>'s <code>radix_cache.py</code> through section by section, sees how it works with the scheduler, and then reads a matching bug fixed 8 days after the release — a fine example of how easily a tree operation that looks right can be wrong. It ends by lining the first version up against today's 820 lines with page alignment, hashes and several eviction policies.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What are the "key" and the "value" stored on a node? Why is the key a stretch of the token sequence rather than one token?
    2. What happens when a match lands in the middle of an edge? And on insertion?
    3. What is the eviction policy? Which nodes cannot be evicted? How is `evictable_size` maintained?
    4. When does the scheduler call `match_prefix`, `inc_ref_counter`, `insert` and `dec_ref_counter`?

??? success "Answers for the self-test (answer first, then open this)"
    1. The key is a stretch of token ids (the tokens on the edge from the parent to the node) and the value is those tokens' slot indices in the pool (a tensor). A radix tree (a compressed prefix tree) collapses a chain of single-child nodes into one edge, so there are far fewer nodes and far less memory than in a trie with one node per token, and matching compares stretch by stretch along the edges.
    2. On landing in the middle of an edge (the new sequence shares only part of it), the edge is split at the fork: a new intermediate node takes the first half and the original node keeps the second, and the match ends at the intermediate node. Insertion splits first in the same way and then hangs a new edge under the intermediate node.
    3. Collect every leaf into a min-heap ordered by `last_access_time` and delete the least recently accessed leaves until `num_tokens` have been freed; nodes with a reference count above 0 (in use by a running request) are skipped; and when deleting a leaf turns its parent into a leaf, the parent goes into the heap too. `evictable_size` gains a node's length when the node is created, loses it when the node becomes referenced (the count goes from 0 to 1), gains it back when the reference is released (1 to 0), and loses it when a leaf is deleted.
    4. When forming a batch, `match_prefix` runs for each waiting request; on admission, `inc_ref_counter(last_node)` locks the path; when a request finishes, `insert(input_ids + output_ids, slots)` puts the whole sequence into the tree, frees the pool slots duplicating what the tree already held, and `dec_ref_counter(last_node)` unlocks.

A six-panel strip first, then the chapter:

<!-- comic ../assets/comics/radix-v1.webp is in Chinese; put it back once the English version exists -->

## The node and matching {#节点与匹配}

```python title="python/sglang/srt/managers/router/radix_cache.py @ 22085081bb L10-28" linenums="10"
class TreeNode:
    def __init__(self):
        self.children = defaultdict(TreeNode)
        self.parent = None
        self.value = None
        self.ref_counter = 0
        self.last_access_time = time.time()

    def __lt__(self, other):
        return self.last_access_time < other.last_access_time


def match(key, seq):
    i = 0
    for k, w in zip(key, seq):
        if k != w:
            break
        i += 1
    return i
```

`TreeNode`'s `children` is a `defaultdict` whose key is a stretch of tokens (a tuple) and whose value is the child node; `value` is the slot indices for that stretch of tokens; `ref_counter` records how many running requests pass through the node; `last_access_time` is for the LRU; and `__lt__` lets a node go straight into `heapq`. `match(key, seq)` returns the two sequences' common prefix length and is the whole tree's only comparison operation.

There are only six public methods: `match_prefix`, `insert`, `evict`, `inc_ref_counter`, `dec_ref_counter` and `evictable_size` (plus `pretty_print` and `total_size` for debugging):

```python title="python/sglang/srt/managers/router/radix_cache.py @ 22085081bb L41-58" linenums="41"
    def match_prefix(self, key):
        if self.disable:
            return [], self.root_node

        value = []
        last_node = [self.root_node]
        self._match_prefix_helper(self.root_node, key, value, last_node)
        if value:
            value = torch.concat(value)
        return value, last_node[0]

    def insert(self, key, value=None):
        if self.disable:
            return len(key)

        if value is None:
            value = [x for x in key]
        return self._insert_helper(self.root_node, key, value)
```

`match_prefix` returns two things: the slot indices that hit (the tensors of each stretch's `value` concatenated) and `last_node` (the last node on the matched path). The scheduler uses `last_node` for the reference counting later. `disable` is how `--model-mode no-cache` is implemented: a match is always empty and an insertion is always "all of it already exists" (it returns `len(key)`, from which the caller frees every slot).

## Recursive matching, splitting and insertion {#递归匹配分裂与插入}

```python title="python/sglang/srt/managers/router/radix_cache.py @ 22085081bb L113-140" linenums="113"
    def _match_prefix_helper(self, node, key, value, last_node):
        node.last_access_time = time.time()

        for c_key, child in node.children.items():
            prefix_len = match(c_key, key)
            if prefix_len != 0:
                if prefix_len == len(key) and prefix_len != len(c_key):
                    new_node = self._split_node(c_key, child, prefix_len)
                    value.append(new_node.value)
                    last_node[0] = new_node
                else:
                    value.append(child.value[:prefix_len])
                    last_node[0] = child
                    self._match_prefix_helper(child, key[prefix_len:], value, last_node)
                break

    def _split_node(self, key, child, split_len):
        # new_node -> child
        new_node = TreeNode()
        new_node.children = {key[split_len:]: child}
        new_node.parent = child.parent
        new_node.ref_counter = child.ref_counter
        new_node.value = child.value[:split_len]
        child.parent = new_node
        child.value = child.value[split_len:]
        new_node.parent.children[key[:split_len]] = new_node
        del new_node.parent.children[key]
        return new_node
```

`_match_prefix_helper` looks among the current node's children for an edge sharing a prefix with `key` (at most one, since sibling edges start with different tokens) and then takes one of two paths: if the common prefix is shorter than the edge it splits and stops there; otherwise the whole edge hits and it recurses into the child with the rest of `key`. `_split_node` cuts an edge in two: a new `new_node` takes the first `split_len` tokens (the value is cut too) and inherits the original node's reference count (because a request passing through the original node must pass through it as well), and the original node becomes `new_node`'s child.

Insertion has the same shape:

```python title="python/sglang/srt/managers/router/radix_cache.py @ 22085081bb L142-168" linenums="142"
    def _insert_helper(self, node, key, value):
        node.last_access_time = time.time()

        for c_key, child in node.children.items():
            prefix_len = match(c_key, key)

            if prefix_len == len(c_key):
                if prefix_len == len(key):
                    return prefix_len
                else:
                    key = key[prefix_len:]
                    value = value[prefix_len:]
                    return prefix_len + self._insert_helper(child, key, value)

            if prefix_len:
                new_node = self._split_node(c_key, child, prefix_len)
                return prefix_len + self._insert_helper(
                    new_node, key[prefix_len:], value[prefix_len:]
                )

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.value = value
            node.children[key] = new_node
            self.evictable_size_ += len(value)
        return 0
```

The return value is "the length of the prefix the tree already holds": from it the caller knows how many of this sequence's leading tokens already have their KV stored, and can free its own copy of those slots (deduplication). On creating a node, `evictable_size_ += len(value)`, because a new node has no references and can be evicted.

![Figure: splitting a node when the match lands in the middle of an edge](../assets/figures/sgl-radix-split.svg){.aig-svg}

## Eviction and reference counting {#淘汰与引用计数}

```python title="python/sglang/srt/managers/router/radix_cache.py @ 22085081bb L67-110" linenums="67"
    def evict(self, num_tokens, evict_callback):
        if self.disable:
            raise RuntimeError()

        leaves = self._collect_leaves()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)

            if x == self.root_node:
                break
            if x.ref_counter > 0:
                continue

            num_evicted += evict_callback(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0:
                heapq.heappush(leaves, x.parent)

    def inc_ref_counter(self, node):
        delta = 0
        while node != self.root_node:
            if node.ref_counter == 0:
                self.evictable_size_ -= len(node.value)
                delta -= len(node.value)
            node.ref_counter += 1
            node = node.parent
        return delta

    def dec_ref_counter(self, node):
        delta = 0
        while node != self.root_node:
            if node.ref_counter == 1:
                self.evictable_size_ += len(node.value)
                delta += len(node.value)
            node.ref_counter -= 1
            node = node.parent
        return delta

    def evictable_size(self):
        return self.evictable_size_
```

`evict`'s policy is a **leaf-first LRU**: collect every leaf, build a min-heap by last access time, pop the least recently accessed leaf, skip it if its reference count is above 0, otherwise call `evict_callback` (the `token_to_kv_pool.free` the scheduler passed in) to free the slots and delete the leaf; and if the parent becomes a leaf as a result, it goes into the heap too. Evicting only leaves is inevitable: an internal node's KV is the prefix of all of its descendants, and deleting one would leave them incomplete.

Reference counts accumulate up the path: `inc_ref_counter(node)` walks from `node` to the root incrementing each node's count; a node whose count goes from 0 to 1 leaves the evictable set, so `evictable_size_` loses its length, and the function returns that change (a negative number) for the scheduler to correct its estimate of the space available ([the previous chapter](first-commit.md)'s admission logic). `dec_ref_counter` is the reverse.

## How the scheduler uses it {#调度器怎么用它}

The tree itself knows nothing about requests; all four call sites are in `model_rpc.py`. When forming a batch it matches first and locks after:

```python title="python/sglang/srt/managers/router/model_rpc.py @ 22085081bb L219-225,270-281"
        for req in self.forward_queue:
            prefix_indices, last_node = self.tree_cache.match_prefix(req.input_ids)
            if req.return_normalized_logprob:
                prefix_indices = prefix_indices[: req.normalized_logprob_start_len]
            req.adjust_input_len = len(req.input_ids) - len(prefix_indices)
            req.prefix_indices = prefix_indices
            req.last_node = last_node
...
                delta = self.tree_cache.inc_ref_counter(req.last_node)
                available_size += delta

                if not (
                    req.adjust_input_len + req.max_new_tokens() + new_batch_total_tokens
                    < available_size
                ):
                    delta = self.tree_cache.dec_ref_counter(req.last_node)
                    available_size += delta
                else:
                    self.token_to_kv_pool.add_refs(req.prefix_indices)
                    can_run_list.append(req)
```

When a request finishes it inserts and deduplicates:

```python title="python/sglang/srt/managers/router/model_rpc.py @ 22085081bb L401-415"
        # Remove finished reqs
        if finished_indices:
            # Update radix cache
            req_pool_indices_cpu = batch.req_pool_indices.cpu().tolist()
            for i in finished_indices:
                req = batch.reqs[i]
                req_pool_idx = req_pool_indices_cpu[i]
                token_ids = tuple(req.input_ids + req.output_ids)
                seq_len = len(token_ids) - 1
                indices = self.req_to_token_pool.req_to_token[req_pool_idx, :seq_len]
                prefix_len = self.tree_cache.insert(token_ids, indices.clone())

                self.token_to_kv_pool.free(indices[:prefix_len])
                self.req_to_token_pool.free(req_pool_idx)
                self.tree_cache.dec_ref_counter(req.last_node)
```

`token_ids` is the whole sequence, input plus output, and `seq_len = len - 1` because the last token's KV has not been computed yet (it is the one just sampled and not yet fed back into the model). `insert` returns the existing prefix's length `prefix_len`, those slots are duplicates and are freed, and the rest now belong to the tree. Then the request slot is freed and the lock released.

The cache-aware scheduling policy is in `scheduler.py`, only 73 lines:

```python title="python/sglang/srt/managers/router/scheduler.py @ 22085081bb L20-52" linenums="20"
    def new_token_estimation_ratio(self):
        return 0.4 if self.schedule_heuristic != "fcfs" else 0.5

    def get_priority_queue(self, forward_queue):
        if self.schedule_heuristic == "lpm":
            # longest prefix match
            forward_queue.sort(key=lambda x: -len(x.prefix_indices))
            return forward_queue
        elif self.schedule_heuristic == "random":
            random.shuffle(forward_queue)
            return forward_queue
        elif self.schedule_heuristic == "fcfs":
            return forward_queue
        elif self.schedule_heuristic == "weight":
            last_node_to_reqs = defaultdict(list)
            for req in forward_queue:
                last_node_to_reqs[req.last_node].append(req)
            for node in last_node_to_reqs:
                last_node_to_reqs[node].sort(key=lambda x: -len(x.prefix_indices))

            node_to_weight = defaultdict(int)
            self._calc_weight_recursive(
                self.tree_cache.root_node, last_node_to_reqs, node_to_weight
            )

            tmp_queue = []
            self._get_weight_priority_recursive(
                self.tree_cache.root_node, node_to_weight, last_node_to_reqs, tmp_queue
            )
            assert len(tmp_queue) == len(forward_queue)
            return tmp_queue
        else:
            raise ValueError(f"Unknown schedule_heuristic: {self.schedule_heuristic}")
```

The default is `lpm` (longest prefix match): descending by the length that hit. The `weight` policy implements the paper's theorem: group the waiting requests by `last_node`, then recurse from the root, where a node's weight is the number of requests hanging in its subtree, traversing depth-first into the heaviest child first — so requests in the same subtree are handled together and the prefix stays in the cache for the shortest time. `new_token_estimation_ratio` is 0.5 for `fcfs` and 0.4 for the others: a cache-aware policy hits more and adds less in practice, so it can estimate more optimistically.

## A bug 8 days after the release {#发布-8-天后的-bug}

`fix radix cache match (#7)` of 2024-01-16 is the repository's seventh PR and changes only two lines:

```bash title="radix-fix.sh"
git show --format='%ad  %h  %an  %s' --date=short 01ca82d765 -- python/sglang/srt/managers/router/radix_cache.py | sed -n '1p;/^@@/,$p'
```

```text title="output"
2024-01-16  01ca82d765  Liangsheng Yin  fix radix cache match (#7)
@@ -116,12 +116,12 @@ class RadixCache:
         for c_key, child in node.children.items():
             prefix_len = match(c_key, key)
             if prefix_len != 0:
-                if prefix_len == len(key) and prefix_len != len(c_key):
+                if prefix_len < len(c_key):
                     new_node = self._split_node(c_key, child, prefix_len)
                     value.append(new_node.value)
                     last_node[0] = new_node
                 else:
-                    value.append(child.value[:prefix_len])
+                    value.append(child.value)
                     last_node[0] = child
                     self._match_prefix_helper(child, key[prefix_len:], value, last_node)
                 break
```

The original condition was "split only if `key` is entirely contained in this edge and is shorter than it", and otherwise take the edge's first `prefix_len` tokens and **recurse into the child**. The mistake is that the second case includes "`key` is longer than the edge but matches only the edge's first half": the match has already forked in the middle of the edge and should go no further, but going further has the rest of `key` compared against grandchildren, and as soon as one happens to share a prefix it returns KV from a **completely different context** as a hit. A small pure-Python model reproduces it (the keys are strings and the values are the characters themselves, which makes the error easy to see):

```python title="radix-bug.py"
"""复现 #7 修掉的匹配 bug：树里有 "Hello_L.A.!" → "world"，查 "Hello_world"。"""
from collections import defaultdict


class Node:
    def __init__(self, value=""):
        self.children, self.value = {}, value


def match(a, b):
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i


def insert(node, key):
    for ck, child in node.children.items():
        n = match(ck, key)
        if n == len(ck):
            return insert(child, key[n:]) if n < len(key) else None
        if n:
            mid = Node(ck[:n]); mid.children[ck[n:]] = child; child.value = ck[n:]
            del node.children[ck]; node.children[ck[:n]] = mid
            return insert(mid, key[n:])
    if key:
        node.children[key] = Node(key)


def match_prefix(node, key, buggy):
    out = []
    for ck, child in node.children.items():
        n = match(ck, key)
        if n:
            if (n == len(key) and n != len(ck)) if buggy else (n < len(ck)):
                out.append(ck[:n])                      # split (only the first half is taken here, with the real split omitted)
            else:
                out.append(ck[:n] if buggy else ck)
                out += match_prefix(child, key[n:], buggy)
            break
    return out


root = Node()
insert(root, "Hello_L.A.!")
insert(root, "Hello_L.A.!world")
for buggy in (True, False):
    hit = "".join(match_prefix(root, "Hello_world", buggy))
    print(f"{'修复前' if buggy else '修复后'}：命中 {hit!r}（{len(hit)} 个 token）")
```

```text title="output"
修复前：命中 'Hello_world'（11 个 token）
修复后：命中 'Hello_'（6 个 token）
```

Before the fix, `Hello_world` "hits" 11 tokens: the first 6 are right (`Hello_`) and the last 5, `world`, come from the sequence `Hello_L.A.!world` — that is the "world" after "Hello_L.A.!", with entirely different KV. After the fix only `Hello_` hits. The two-line correction: split and stop as soon as the common prefix is shorter than the edge; append the whole edge's value (rather than a slice) when the whole edge hits, and then recurse. A unit test covers this kind of error badly (it takes a sequence constructed to fork and then happen to match), which is why every branch of a tree operation deserves to be walked through by hand like this.

## Design trade-offs {#设计取舍}

Several choices in the first radix tree, and why they make sense:

- **A leaf LRU rather than a global one.** An internal node cannot be deleted on its own, so eviction happens only at leaves; ordering leaves by `last_access_time` amounts to "the least recently used complete path first", and it takes only a heap.
- **Reference counts rather than a "running, do not evict" flag.** Several requests can share one path, and a count expresses "evictable once the last user has left" correctly.
- **Insertion when the request finishes, not when the prefill completes.** Simple, but at a price: a running request's prefix is invisible to other requests (identical prefixes within one batch are each computed separately). Only `in batch prefix caching by delay scheduling (#2442)` of 2024-12-11 and the later `cache_unfinished_req` made up for it.
- **Comparing Python tuples.** `match` is a pure Python loop and each match is O(the prefix's length). The paper measured 0.2 seconds of tree operations for 100 requests, which was good enough then; `Optimize radix tree matching (#364)` of 2024-04 and the C++ and Rust trees of 2025 all work on this point.

## What happened afterwards {#后来怎么样了}

This file's history can be followed directly with `--follow` (it moved from `managers/router/` to `mem_cache/` in #807):

```bash title="radix-evolution.sh"
REF=${REF:-29f6d408c0}
f=python/sglang/srt/mem_cache/radix_cache.py
echo "提交数：$(git log --follow --format=%h "$REF" -- $f | wc -l)"
for y in 2024 2025 2026; do echo "  $y：$(git log --follow --date=short --format=%ad "$REF" -- $f | grep -c "^$y")"; done
echo "今天的行数：$(git show "$REF:$f" | wc -l)"
echo "mem_cache/ 的 Python 文件数：$(git ls-tree -r --name-only "$REF" -- python/sglang/srt/mem_cache | grep -c '\.py$')"
```

```text title="output"
提交数：127
  2024：30
  2025：42
  2026：55
今天的行数：823
mem_cache/ 的 Python 文件数：156
```

A few milestones:

| Date | Commit | The change |
| --- | --- | --- |
| 2024-04-18 | `Optimize radix tree matching (#364)` | matching reimplemented faster, children indexed by first token |
| 2024-07-29 | `Code structure refactor (#807)` | moved into `mem_cache/`, next to `memory_pool.py` |
| 2025-02-23 | `Hierarchical Caching for SGLang (#2693)` | `hiradix_cache.py` inherits from it, with nodes movable between GPU and CPU |
| 2025-03-12 | `Support page size > 1 (#4356)` | the key is page-aligned and matching works a page at a time |
| 2025-08-11 | `HiCache Storage: generate hash when inserting new nodes (#9053)` | nodes carry a hash, used as the key in external storage |
| 2025-09 → 2026-03 | `#10190`, `#11506`, `#18843` | eviction policies made pluggable: LRU, LFU, SLRU… |
| 2026-01 → 2026-03 | the `[RadixTree][N/N Refactor]` series | unified insertion and eviction parameters and a locking interface; `SWARadixTree` for sliding-window attention |

Today's `TreeNode` looks like this:

```python title="python/sglang/srt/mem_cache/radix_cache.py @ 29f6d408c0 L249-277" linenums="249"
class TreeNode:
    counter = 0

    def __init__(self, id: Optional[int] = None, priority: int = 0):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: RadixKey = None
        self.value: Optional[torch.Tensor] = None
        self.lock_ref = 0
        self.last_access_time = time.monotonic()
        self.creation_time = time.monotonic()

        self.hit_count = 0
        # store hash values of each pages
        self.hash_value: Optional[List[str]] = None
        # Namespace-aware hashes used only for external KV events.
        self.event_hash_value: Optional[List[str]] = None
        # priority for priority-aware eviction
        self.priority = priority

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    @property
    def evicted(self):
        return self.value is None

    def __lt__(self, other: TreeNode):
        return self.last_access_time < other.last_access_time
```

Against the first version: it gains `id`, `priority`, `hit_count`, `host_value` (the copy on the CPU side) and `hash_value` (the storage key), `lock_ref` replaces `ref_counter`, and `last_access_time` uses a monotonic clock (#6211 of 2025-05-17 fixed the ordering anomalies a wall clock caused). The key also went from a tuple to a `RadixKey` supporting page alignment and a "pair view" prepared for sliding windows. But the names and the shape of the four functions `_match_prefix_helper`, `_split_node`, `_insert_helper` and `evict` have survived from January 2024 to now.

## Exercises {#练习}

**1. Why insertion returns the existing length.** Read `_insert_helper`'s three `return`s, explain what the value means in each case, and which slots the caller frees once it has it.

??? success "Answer"
    The whole edge hits and `key` ends too: it returns `prefix_len` (the whole sequence already exists). The whole edge hits but `key` has more left: it returns the edge's length plus the recursive result. A partial hit: after splitting it returns the part that hit plus the recursive result (the recursion under the new node usually creates an edge and returns 0). The caller's `indices[:prefix_len]` is the stretch of slots this request duplicates from the tree, and freeing it leaves one copy of the KV in the tree.

**2. Eviction's boundary.** Construct a case where `evict(num_tokens)` ends having freed fewer than `num_tokens` tokens. What does the scheduler do then?

??? success "Answer"
    When every leaf is referenced (a count above 0), or the tree is already empty, the heap empties without reaching the target. `Batch.init_extend_batch` calls `alloc` once more after `evict`, and on another failure prints "Prefill out of memory." and calls `exit()` — the first version simply exits the process. This is also why admission estimates future demand: it avoids this by admitting fewer requests rather than by preempting. Later versions retract some decoding requests in this situation.

**3. The price of deferred insertion.** Use `git show 9208618b3e` to see what #2442 "in batch prefix caching by delay scheduling" solved and how.

??? success "A way to approach it"
    When several requests in one batch share a prefix, the first version computes it separately for each (insertion happens only when a request finishes). #2442 identifies, while forming a batch, the requests that share a long prefix with a running request or with another request in this batch but whose prefix is not in the tree yet, and defers them to the next round so that the first request finishes and inserts and the rest can hit. This is the first correction to the "when to insert" trade-off.

!!! interview "How to answer in an interview"
    Present RadixAttention in four steps — the structure, the operations, the interface with the scheduler, the trade-offs: a radix tree whose node holds a stretch of tokens and the corresponding slots; matching and insertion both come down to "find the edge sharing a prefix and split it in the middle if necessary"; the scheduler matches and locks on admission and inserts and deduplicates on completion; eviction happens only at leaves, by LRU. Then raise a pitfall yourself (the matching bug fixed 8 days after the release, say, or the metadata overhead a page size of 1 brings) and how it evolved (page alignment, tiering, several eviction policies), which shows you have read the code rather than memorised the concept.

## Summary {#小结}

- [x] The first version is 220 lines: a `TreeNode` (a stretch of tokens as the key, slots as the value, a reference count and an access time), recursive matching and insertion, and a leaf LRU eviction.
- [x] The scheduler matches and locks the path on admission and inserts and deduplicates on completion; `evictable_size` moves with the reference counts and feeds the admission estimate.
- [x] The matching bug fixed 8 days after the release makes the point: once the match forks in the middle of an edge it must not recurse further, or KV from another context is returned as a hit.
- [x] The four core functions' shape survives to this day; the changes happened in the key (page alignment, hashes), the node (locks, a CPU copy) and the eviction policy (made pluggable).
