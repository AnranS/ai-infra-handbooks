# The compressed FSM and jump-forward decoding: three generations of structured-output backend

<p class="lead">The paper's second runtime technique is the compressed finite-state machine: the long deterministic stretches of text in JSON need not be generated token by token and can be "jumped" over whole. Its code history is interesting: the first version copied Outlines' FSM implementation and masked during sampling; three weeks later jump-forward was added; a month after that the copied code was swapped back for the Outlines library; xgrammar arrived in October 2024, llguidance in February 2025, and in March of that year jump-forward was deleted from the scheduler entirely — because the grammar libraries do it better. This chapter follows that line, watching an idea from the paper reshaped by engineering reality.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. How does regex-constrained decoding work at bottom? How much computation does each step take?
    2. What does jump-forward jump over? Why does it have to retokenize afterwards?
    3. Where does the first version compile the FSM? Why on a background thread?
    4. Why was jump-forward deleted from the scheduler in March 2025? Where does the feature live today?

??? success "Answers for the self-test (answer first, then open this)"
    1. Compile the regex into a deterministic finite-state machine, then precompute for each state which tokens are allowed (those whose string the FSM can accept from that state); at each step set the disallowed tokens' logits to negative infinity, and after sampling advance the state with the token chosen. The precomputation is one-off (proportional to the vocabulary size) and each step is just a mask.
    2. It jumps over the string corresponding to a run of states in the FSM with only one outgoing edge (such as `", "age": ` in JSON); those characters are determined and the model does not have to decide them. But appending them as new tokens directly would give a tokenization different from the one seen in training (the boundary between `"age` and `":`, for instance), so the preceding text and the jumped string are concatenated, retokenized and continued from there.
    3. In `FSMCache`: the compiled result is cached by regex, and on first seeing a regex a thread is started to compile it (`init_fsm_in_background`) while an arriving request waits for it to finish. Compiling a regex takes seconds to tens of seconds, which on the main thread would stall the whole scheduler.
    4. Jump-forward has to pull a request out of the batch inside the scheduling loop, change its input, retokenize and put it back, which interleaves with overlapped scheduling, speculative decoding, chunked prefill and the rest at a high maintenance cost; grammar libraries like xgrammar implement the same "find the deterministic string" (`find_jump_forward_string`) on the C++ side, and SGLang only has to call the interface. Today `BaseGrammarObject` still keeps `try_jump_forward` and its companions, and the backend decides whether to support them.

A six-panel strip first, then the chapter:

<!-- comic ../assets/comics/fsm-jump.webp is in Chinese; put it back once the English version exists -->

## The first version: copy Outlines, mask during sampling {#第一版复制-outlines采样时做掩码}

The first code release's `srt/constrained/` has four files, and the heads of `fsm.py` and `regex.py` state where they came from:

```python title="python/sglang/srt/constrained/fsm.py @ 22085081bb L1-2" linenums="1"
# Adapted from:
# https://github.com/outlines-dev/outlines/blob/0355ab4272a5d7e4d94c4a53a52593f885b81a61/outlines/fsm/fsm.py
```

What Outlines does: `interegular` parses the regex into an FSM and `create_fsm_index_tokenizer` computes the allowed token set for each state (`states_to_token_maps`). SGLang copied that code rather than depending on Outlines directly because Outlines' interface was changing fast and its dependencies were heavy at the time, and because SGLang wanted to do its own things on the FSM (jump-forward, below, needs access to the FSM's internal transition table).

Compiling a regex is slow, so there is a cache with a background thread:

```python title="python/sglang/srt/constrained/fsm_cache.py @ 22085081bb L1-41" linenums="1"
import threading

from sglang.srt.constrained.fsm import RegexFSM
from sglang.srt.constrained.tokenizer import TransformerTokenizer


def get_fsm(regex, tokenizer, fsm_cache_entry):
    outlines_tokenizer = TransformerTokenizer(tokenizer)
    fsm = RegexFSM(regex, outlines_tokenizer)
    fsm_cache_entry.fsm = fsm
    fsm_cache_entry.event.set()


class FSMCacheEntry:
    def __init__(self):
        self.fsm = None
        self.event = threading.Event()


class FSMCache:
    def __init__(self, tokenizer):
        self.cache = {}
        self.tokenizer = tokenizer

    def init_fsm_in_background(self, regex):
        if regex not in self.cache:
            self.cache[regex] = FSMCacheEntry()
            threading.Thread(
                target=get_fsm,
                args=(
                    regex,
                    self.tokenizer,
                    self.cache[regex],
                ),
            ).start()

    def get_fsm(self, regex):
        self.init_fsm_in_background(regex)
        entry = self.cache[regex]
        entry.event.wait()
        return entry.fsm
```

When a request arrives (`handle_generate_request`) carrying a `regex`, the scheduler calls `get_fsm(regex)`: a regex seen for the first time starts a compiling thread and `event.wait()` waits for it; afterwards the cache hits and returns directly. Note that `get_fsm` blocks — the scheduling loop stops here to wait for the compilation, an acceptable simplification in the first version (in the paper's experiments one regex is reused by a batch of requests). The masking during sampling is in `Batch.sample`:

```python title="python/sglang/srt/managers/router/infer_batch.py @ 22085081bb L279-294,307-313"
    def sample(self, logits: torch.Tensor):
        # Post process logits
        logits = logits.contiguous()
        logits.div_(self.temperatures)
        logits.add_(self.logit_bias)

        has_regex = any(req.regex_fsm is not None for req in self.reqs)
        if has_regex:
            allowed_mask = torch.empty_like(logits[0], dtype=torch.bool)
            for i, req in enumerate(self.reqs):
                if req.regex_fsm is not None:
                    allowed_mask.zero_()
                    allowed_mask[
                        req.regex_fsm.allowed_token_ids(req.regex_fsm_state)
                    ] = 1
                    logits[i].masked_fill_(~allowed_mask, float("-inf"))
...
        if has_regex:
            batch_next_token_ids_cpu = batch_next_token_ids.cpu().numpy()
            for i, req in enumerate(self.reqs):
                if req.regex_fsm is not None:
                    req.regex_fsm_state = req.regex_fsm.next_state(
                        req.regex_fsm_state, batch_next_token_ids_cpu[i]
                    )
```

It builds a boolean mask per request, `masked_fill_`s to negative infinity, and advances the state on the CPU after sampling. Each request with a regex costs one Python loop plus one GPU mask per step — simple, but a visible overhead at a large batch, later replaced by a batched `fill_vocab_mask_batched` and an `apply_vocab_mask` on the GPU.

## Three weeks later: fast forward, then renamed jump-forward {#三周后fast-forward然后改名-jump-forward}

The commit `fast regex decode` (`01ee0fbc05`) of 2024-01-25 added the paper's compressed FSM, called fast-forward at the time; `jump-forward rename (#144)` of 5 February settled the name (LMSYS published a blog post introducing jump-forward the same day). The commits along this line:

```bash title="jump-forward-timeline.sh"
REF=${REF:-29f6d408c0}
git log --reverse --date=short --format='%ad  %h  %s' "$REF" | grep -iE 'jump.?forward|fast regex' | cut -c1-100
```

```text title="output"
2024-01-25  01ee0fbc05  fast regex decode
2024-02-05  26f0bedc8f  jump-forward rename (#144)
2024-02-09  4d303c4fa3  Fix token usage with jump forward (#174)
2024-02-15  63ba630bbb  Refactor decoding logprob and add completion_tokens_wo_jump_forward (#189)
2024-06-16  ad5f04d6ce  Fix the Jump-Forward with Chinese (#551)
2024-07-19  7620cd37dd  Fix jump forward when streaming (#665)
2024-08-13  e205527cb1  Fix jump forward final state circular path bug. (#1084)
2024-10-12  9610fcd469  Fix the batch_is_full check for jump-forward decoding (#1654)
2025-03-03  935cda944b  Misc clean up; Remove the support of jump forward (#4032)
```

v0.1.12's (2024-02-11) `jump_forward.py` is a self-contained `JumpForwardMap`:

```python title="python/sglang/srt/constrained/jump_forward.py @ v0.1.12 L9-56" linenums="9"
    def __init__(self, regex_string):
        @disk_cache()
        def _init_state_to_jump_forward(regex_string):
            regex_pattern = interegular.parse_pattern(regex_string)
            regex_fsm, _ = make_deterministic_fsm(regex_pattern.to_fsm().reduce())

            fsm_info: FSMInfo = regex_fsm.fsm_info

            symbol_to_id = fsm_info.alphabet_symbol_mapping
            id_to_symbol = {}
            for symbol, id_ in symbol_to_id.items():
                id_to_symbol.setdefault(id_, []).append(symbol)

            transitions = fsm_info.transitions
            dirty_states = set()
            state_to_jump_forward = {}

            for (state, id_), next_state in transitions.items():
                if state in dirty_states:
                    continue
                if state in state_to_jump_forward:
                    dirty_states.add(state)
                    del state_to_jump_forward[state]
                    continue
                if len(id_to_symbol[id_]) > 1:
                    dirty_states.add(state)
                    continue

                state_to_jump_forward[state] = (id_to_symbol[id_][0], next_state)

            return state_to_jump_forward

        self.state_to_jump_forward = _init_state_to_jump_forward(regex_string)

    def valid_states(self):
        return self.state_to_jump_forward.keys()

    def jump_forward(self, state):
        if state not in self.state_to_jump_forward:
            return None

        jump_forward_str = ""
        next_state = None
        while state in self.state_to_jump_forward:
            symbol, next_state = self.state_to_jump_forward[state]
            jump_forward_str += symbol
            state = next_state
        return jump_forward_str, next_state
```

It reads the FSM's transition table (`fsm_info.transitions`) and finds every state with only one outgoing edge corresponding to exactly one character (`state_to_jump_forward`), marking states with several edges, or an edge covering several characters, as `dirty`. A query walks along those single edges from the current state and concatenates the characters, which is the string that can be jumped over. The compiled result is stored on disk with `disk_cache` (`Disk FSM cache (#63)`, added 2024-01-21, because the same regex is wanted again after a restart).

Before each decode step the scheduler checks which requests can jump:

```python title="python/sglang/srt/managers/router/infer_batch.py @ v0.1.12 L334-372" linenums="334"
    def check_for_jump_forward(self):
        jump_forward_reqs = []
        filter_indices = [i for i in range(len(self.reqs))]

        req_pool_indices_cpu = None

        for i, req in enumerate(self.reqs):
            if req.jump_forward_map is not None:
                res = req.jump_forward_map.jump_forward(req.regex_fsm_state)
                if res is not None:
                    jump_forward_str, next_state = res
                    if len(jump_forward_str) <= 1:
                        continue

                    # insert the old request into tree_cache
                    token_ids_in_memory = tuple(req.input_ids + req.output_ids)[:-1]
                    if req_pool_indices_cpu is None:
                        req_pool_indices_cpu = self.req_pool_indices.cpu().tolist()
                    req_pool_idx = req_pool_indices_cpu[i]
                    indices = self.req_to_token_pool.req_to_token[
                        req_pool_idx, : len(token_ids_in_memory)
                    ]
                    prefix_len = self.tree_cache.insert(
                        token_ids_in_memory, indices.clone()
                    )
                    self.token_to_kv_pool.free(indices[:prefix_len])
                    self.req_to_token_pool.free(req_pool_idx)
                    self.tree_cache.dec_ref_counter(req.last_node)

                    # jump-forward
                    req.jump_forward_and_retokenize(jump_forward_str, next_state)

                    jump_forward_reqs.append(req)
                    filter_indices.remove(i)

        if len(filter_indices) < len(self.reqs):
            self.filter_batch(filter_indices)

        return jump_forward_reqs
```

A request that can jump is **pulled out of the batch**: its existing KV goes into the radix tree first (as when a request finishes), the slots and the request slot are freed, and then `jump_forward_and_retokenize` concatenates the preceding text with the jumped string, retokenizes and produces new `input_ids` — and it goes back into the waiting queue to enter the next round as a new extend request, whose prefix naturally hits on the tree. The design is clever: **jump-forward is reduced to "one new request with a prefix cache"**, with no tensors inside the batch to change. The price is there too: a request goes through "pull out → insert into the tree → retokenize → be admitted again", deeply coupled to the batch's other machinery.

The blog post gives an example of why retokenizing is necessary: appending the jumped string character by character would split `"name": ` at token boundaries different from the ones the model was trained on and the output quality afterwards would drop; after retokenizing, the prefix's tokens may differ slightly from the original (the blog measured about 4% of token boundaries changing), which is why the tree's matching is used rather than assuming the prefix is identical. `Fix the Jump-Forward with Chinese (#551)` of 2024-06-16 fixed exactly the multi-byte character boundary problem on this path.

![Figure: a compressed FSM collapses a chain of single-edge states into one jump](../assets/figures/sgl-jump-forward.svg){.aig-svg}

## Back to a library, then to a new one {#换回库再换新库}

`import outlines (#168)` of 2024-02-09 deleted the copied FSM code and depended on Outlines directly (followed by a run of "Pin outlines version", "Support outlines > 0.0.31" and "Fix outlines-0.0.35 incompatibility" — the price of depending on a fast-moving library). In August, `JSON constrained support (#1125)` added JSON schemas. On 26 October, `Support both xgrammar and outlines for constrained decoding (#1752)` brought in a second backend, and v0.4's blog post claimed JSON decoding 10 times faster; in November, `Fix grammar backend for tensor parallelism (#2020)` factored out `base_grammar_backend.py` and the backends became pluggable; `Support llguidance (#3298)` of 2025-02-26 added a third; and `integrate Structural Tag in xgrammar backend for function calling (#3566)` of 2025-02-28 applied constrained decoding to function calling.

The directory at four points in time:

```bash title="constrained-dir.sh"
REF=${REF:-29f6d408c0}
for r in 22085081bb v0.1.12 v0.4.0 "$REF"; do
  echo "== $r"; git ls-tree --name-only "$r" python/sglang/srt/constrained/ | sed 's|python/sglang/srt/constrained/|   |'
done
```

```text title="output"
== 22085081bb
   fsm.py
   fsm_cache.py
   regex.py
   tokenizer.py
== v0.1.12
   __init__.py
   base_cache.py
   fsm_cache.py
   jump_forward.py
== v0.4.0
   __init__.py
   base_grammar_backend.py
   outlines_backend.py
   outlines_jump_forward.py
   xgrammar_backend.py
== 29f6d408c0
   base_grammar_backend.py
   grammar_manager.py
   json_schema_validation.py
   llguidance_backend.py
   outlines_backend.py
   outlines_jump_forward.py
   reasoner_grammar_backend.py
   torch_ops
   utils.py
   xgrammar_backend.py
```

Today's interface is `BaseGrammarObject`, with every backend implementing the same set of methods:

```python title="python/sglang/srt/constrained/base_grammar_backend.py @ 29f6d408c0 L58-76,119-145"
class BaseGrammarObject:
    def __init__(self):
        self._finished = False
        self.grammar_stats = None
        self.current_token = None

    def maybe_init_reasoning(self, reasoning: bool):
        pass

    def accept_token(self, token: int) -> None:
        """
        Accept a token in the grammar.
        """
        raise NotImplementedError()

    def rollback(self, k: int):
        raise NotImplementedError()

    def is_terminated(self):
...
    def try_jump_forward(self, tokenizer) -> Optional[Tuple[List[int], str]]:
        """
        Try to jump forward in the grammar.

        Returns:
            A jump forward helper which may be used in `jump_forward_str_state`.
            None if the jump forward is not possible.
        """
        raise NotImplementedError()

    def jump_forward_str_state(self, helper: Tuple[List[int], str]) -> Tuple[str, int]:
        """
        Jump forward for the grammar.

        Returns:
            A tuple of the jump forward string and the next state of the grammar
            (which can be used in `jump_and_retokenize` if needed).
        """
        raise NotImplementedError()

    def jump_and_retokenize(
        self, old_output_ids: List[int], new_output_ids: List[int], next_state: int
    ) -> None:
        """
        Jump forward occurs, and update the grammar state if needed.
        """
        raise NotImplementedError()
```

The three methods `try_jump_forward`, `jump_forward_str_state` and `jump_and_retokenize` are the interface jump-forward left behind; the xgrammar backend's implementation simply forwards to its matcher:

```python title="python/sglang/srt/constrained/xgrammar_backend.py @ 29f6d408c0 L167-180" linenums="167"
    def try_jump_forward(self, tokenizer) -> Optional[Tuple[List[int], str]]:
        s = self.matcher.find_jump_forward_string()
        if s:
            return [], s
        return None

    def jump_forward_str_state(self, helper: Tuple[List[int], str]) -> Tuple[str, int]:
        _, data = helper
        return data, -1

    def jump_and_retokenize(
        self, old_output_ids: List[int], new_output_ids: List[int], next_state: int
    ):
        k = 0
```

## Deleting jump-forward from the scheduler {#删掉调度器里的-jump-forward}

`Misc clean up; Remove the support of jump forward (#4032)` of 2025-03-03 changed 41 files and deleted 426 lines. Its commit message and what it removed:

```bash title="remove-jump-forward.sh"
git show --format='%ad  %h  %an%n%s%n%b' --date=short --stat=90 935cda944b | grep -v '^$' | grep -iE 'jump|^2025|Misc|constrained|schedule|infer|batch' | head -20
```

```text title="output"
2025-03-03  935cda944b  Lianmin Zheng
Misc clean up; Remove the support of jump forward (#4032)
 ...atch_inference.py => offline_batch_inference_eagle.py} |   2 +-
 python/sglang/srt/constrained/base_grammar_backend.py     |   1 -
 python/sglang/srt/constrained/outlines_backend.py         |   7 +-
 python/sglang/srt/layers/attention/flashinfer_backend.py  |   3 +-
 .../sglang/srt/layers/attention/flashinfer_mla_backend.py |   3 +-
 python/sglang/srt/managers/schedule_batch.py              | 126 +---------
 python/sglang/srt/managers/scheduler.py                   |  59 -----
 python/sglang/srt/model_executor/forward_batch_info.py    |   2 +-
 test/srt/test_ebnf_constrained.py                         |   8 -
 test/srt/test_json_constrained.py                         |  22 --
 test/srt/test_regex_constrained.py                        |  17 --
```

The reason for the removal can be inferred from the code of the time: by early 2025 the scheduler already had overlapped scheduling (the CPU's scheduling one step out of phase with the GPU's forward pass), EAGLE speculative decoding (several tokens verified in one step) and chunked prefill, each of which had to meet jump-forward's "pull the request out of the batch and put it back" path; meanwhile xgrammar implemented the same `find_jump_forward_string` in C++ and did the mask computation on the C++ side as well. So SGLang kept the interface and deleted its own implementation: a classic case of "build it first, then use the library". The paper's technique did not disappear — it became a grammar library's feature, and SGLang went back to its main business of scheduling and caching.

## Design trade-offs {#设计取舍}

| Stage | The choice | Gained | Lost |
| --- | --- | --- | --- |
| 2024-01 | copy Outlines' FSM code | access to the transition table, enabling jump-forward | bugs have to be fixed along with upstream |
| 2024-01-25 | reduce jump-forward to "a new request" | no tensors inside the batch to change, the prefix cache reused | the request is pulled out, retokenized and re-admitted, a deep coupling |
| 2024-02 | depend on the Outlines library | one copy of the code less to maintain | version compatibility broke repeatedly |
| 2024-10 | several backends (xgrammar) | speed, JSON schemas, EBNF, structural tags | a backend abstraction to maintain |
| 2025-03 | delete its own jump-forward | the scheduler simplified | the feature depends on whether a backend supports it |

## What happened afterwards {#后来怎么样了}

- Around June 2025 constrained decoding met "reasoning models": `reasoner_grammar_backend.py` lets a grammar take effect only after `</think>`.
- Applying the mask went from one `masked_fill_` per request to a batched `fill_vocab_mask_batched` plus one GPU kernel (`constrained/torch_ops/`).
- `grammar_manager.py` manages the compilation cache and the fallback on failure in one place (a grammar that fails to compile returns an `InvalidGrammarObject` rather than leaving the request stuck).
- Function calling ([chapter 20](../platform/entrypoints.md)) is parsed in `function_call/` and in the Rust gateway, and constrained decoding is only responsible for guaranteeing the format during generation.

## Exercises {#练习}

**1. The race in background compilation.** In the first version's `FSMCache.get_fsm`, what happens when two requests bring the same new regex at the same time? Does it compile twice?

??? success "Answer"
    No: `init_fsm_in_background` checks `regex not in self.cache` first, so the first call puts an empty `FSMCacheEntry` in and starts the thread, and the second sees the key already present and goes straight to `event.wait()`. But this relies on the scheduling loop being single-threaded (two calls never enter the `if` simultaneously); the first version's scheduling does run in one thread, so it holds.

**2. When jump-forward pays.** From `check_for_jump_forward`'s code, what kind of regex or JSON schema gains from jump-forward? Why skip when `len(jump_forward_str) <= 1`?

??? success "Answer"
    Patterns with long deterministic stretches (JSON with fixed key names, a sentence with a fixed prefix). Every jump costs a pull-out, a retokenization and a fresh extend, and when the jumped string is a single character the gain does not cover the overhead, so it is skipped.

**3. Verify the removal's effect yourself.** Count the lines containing `jump_forward` under `managers/` and `constrained/` at `935cda944b`'s parent and at the commit itself.

??? success "A way to approach it"
    Compare `git grep -c jump_forward 935cda944b^ -- python/sglang/srt/managers python/sglang/srt/constrained` with `git grep -c jump_forward 935cda944b -- ...`; the references on the scheduler side go to essentially zero and only the backend interface is left in `constrained/`.

!!! interview "How to answer in an interview"
    "How is structured output done, and how is it made faster?" — At bottom it is an FSM mask (precompute each state's allowed tokens and mask once per step); the speedup is jumping over the deterministic stretches (a compressed FSM, jump-forward), and the hard parts are retokenizing and the coupling with the scheduler. You can add a sentence on SGLang's evolution: it implemented jump-forward itself, later handed it to a grammar library like xgrammar, and the scheduler keeps only the interface — which shows you know which layer the feature belongs in.

## Summary {#小结}

- [x] The first version copied Outlines' FSM, compiled in the background through `FSMCache`, and masked per request in `Batch.sample`.
- [x] Jump-forward reduces "skip the deterministic string" to one new request with a prefix cache, and must retokenize.
- [x] The backend went from copied code → the Outlines library → pluggable xgrammar and llguidance; jump-forward left the scheduler in 2025-03 and the feature moved into the grammar libraries.
- [x] This is a complete specimen of "an idea from a paper → an implementation of one's own → handed to a dedicated library".
