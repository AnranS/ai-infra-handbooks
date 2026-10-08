# EAGLE speculative decoding in four parts: why it became a worker

<p class="lead">Speculative decoding was already a "prototype" in the Q3 2024 roadmap, and it entered the main line in four commits between 29 December 2024 and 2 January 2025 — titled part 1 through part 4. The last part's PR number, #2150, is smaller than the other three's: it was opened in November and merged once the three pieces of groundwork beneath it (the draft model, the CUDA graph and DP attention fixes, the scheduler's small changes) had landed. This chapter covers why EAGLE was implemented as a subclass of <code>TpModelWorker</code> rather than a branch in the scheduler, how it gets on with the KV pool, the radix tree and CUDA graphs, and how that "worker" abstraction later accommodated MTP, n-gram and a dozen new speculative methods.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. How many stages does one EAGLE decode step have in SGLang? How many times does the target model run, and how many the draft model?
    2. Why make speculative decoding an `EAGLEWorker(TpModelWorker)`? What does the scheduler have to change for it?
    3. How are the draft tokens' KV slots allocated? What happens to the tokens rejected at verification?
    4. What does "part 4's PR number being smaller than part 1's" say about the way it was developed?

??? success "Answers for the self-test (answer first, then open this)"
    1. `draft`: the draft model runs `speculative_num_steps` steps, taking the top-k at each and expanding them into a tree; `verify`: the target model runs **once**, verifying every candidate on the tree at the same time with a tree attention mask and giving each request its accepted token sequence; `forward_draft_extend_after_decode`: the draft model does one extend with the accepted tokens to prepare for the next round. The target model once, the draft model `num_steps + 1` times.
    2. The scheduler knows one interface: given a batch, return the next tokens and how many were accepted. `EAGLEWorker` holds the target `TpModelWorker` and the draft model's `ModelRunner` internally and still presents `forward_batch_generation` outside. What the scheduler has to change is that "one step may produce several tokens": a `Req` appends several ids, `seq_lens` grows by several at once, and streaming pushes by the accepted count (part 3's "small modifications").
    3. Before drafting, slots for `batch_size x topk x num_steps` candidates are allocated at once (`alloc_token_slots`, with the allocator's state backed up); after verification only the slots on the accepted path are kept and the rest freed; with a page size above 1 the last page also has to be copied. An accepted token enters the radix tree exactly as an ordinary decoded one does.
    4. Open one large PR to show the overall design, then split off the parts that can land independently as small PRs and merge them first (the draft model's file, the compatibility fixes for CUDA graphs and DP attention, the scheduler changes), and merge the large one last — spreading the risk over several reviews.

A six-panel strip first, then the chapter:

<!-- comic ../assets/comics/eagle.webp is in Chinese; put it back once the English version exists -->

## The four parts and what followed {#四部曲与它们的后续}

```bash title="eagle-commits.sh"
for h in 9c05c6898e b0524c3789 ad20b7957e 815dce0554 8c8779cd05 013021b6a1 f9905d59a8 862dd76c76 9fafa62db7 9fb48f951f e983e43248 b26bc86b36; do
  git log -1 --date=short --format='%ad  %h  %s' $h 2>/dev/null | cut -c1-96
done
```

```text title="output"
2024-12-29  9c05c6898e  Add llama_eagle.py (#2640)
2024-12-31  b0524c3789  Eagle speculative decoding part 2: Fix cuda graph + DP attention hanging
2025-01-02  ad20b7957e  Eagle speculative decoding part 3: small modifications to the general sc
2025-01-02  815dce0554  Eagle speculative decoding part 4: Add EAGLE2 worker (#2150)
2025-01-03  8c8779cd05  [Fix] fix retract error in eagle speculative decoding (#2711)
2025-02-03  013021b6a1  refactor EAGLE 2 (#3269)
2025-02-07  f9905d59a8  support speculative decoding kernel in sgl-kernel (#3373)
2025-02-15  862dd76c76  Support NextN (MTP) speculative decoding for DeepSeek-V3/R1 (#3582)
2025-03-04  9fafa62db7  Share target model embed and head weights for nextn (#4033)
2025-03-09  9fb48f951f  Support nextn for flashinfer mla attention backend (#4218)
2025-04-02  e983e43248  Add Eagle Speculative Decoding to FA3 Backend (#4951)
2025-03-30  b26bc86b36  Support page size > 1 + eagle (#4908)
```

The month after the four parts went into fixing edge cases: retraction (#2711), crashes with several concurrent requests (#2730), `max_tokens = 1` (#2939), CPU overhead and `flush_cache` (#3014); #3269 of 3 February was a wholesale restructuring, #3373 of 7 February moved the tree mask and other verification kernels into sgl-kernel, and #3582 of 15 February let DeepSeek-V3 and R1's own MTP head (NextN) take the same path.

## Making it a worker {#做成一个-worker}

v0.4.6's `speculative/` has only five files:

```bash title="spec-dir.sh"
REF=${REF:-29f6d408c0}
for t in v0.4.6 v0.5.0rc0 "$REF"; do
  printf '%-11s %3d 个文件：' "$t" "$(git ls-tree -r --name-only "$t" -- python/sglang/srt/speculative | grep -c '\.py$')"
  git ls-tree -r --name-only "$t" -- python/sglang/srt/speculative | grep '\.py$' | grep -c '_worker' | xargs printf '%s 个带 worker 的文件；'
  git ls-tree -r --name-only "$t" -- python/sglang/srt/speculative | grep '_worker.*\.py$' | sed 's|.*/||; s|\.py||' | tr '\n' ' ' | cut -c1-120; echo
done
```

```text title="output"
v0.4.6        5 个文件：1 个带 worker 的文件；eagle_worker 

v0.5.0rc0     6 个文件：1 个带 worker 的文件；eagle_worker 

29f6d408c0   62 个文件：11 个带 worker 的文件；base_spec_worker dflash_worker_v2 draft_worker_common dspark_worker_v2 eagle_worker_common eagle_worker_v2 frozen_kv_mtp
```

At its centre is `EAGLEWorker` in `eagle_worker.py`:

```python title="python/sglang/srt/speculative/eagle_worker.py @ v0.4.6 L53-60" linenums="53"
class EAGLEWorker(TpModelWorker):

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
```

It inherits from `TpModelWorker`, so it *is* the target model's worker, and holds an extra `ModelRunner` for the draft model. The entry point the scheduler calls:

```python title="python/sglang/srt/speculative/eagle_worker.py @ v0.4.6 L236-280" linenums="236"
    def forward_batch_speculative_generation(
        self, batch: ScheduleBatch
    ) -> Tuple[LogitsProcessorOutput, List[int], int, int]:
        """Run speculative decoding forward.

        NOTE: Many states of batch is modified as you go through. It is not guaranteed that
        the final output batch have the same state as the input.

        Args:
            batch: The batch to run forward. The state of the batch is modified as it runs.
        Returns:
            A tuple of the final logit output of the target model, next tokens accepeted,
            the batch id (used for overlap schedule), and number of accepeted tokens.
        """
        if batch.forward_mode.is_decode():
            with self.draft_tp_context(self.draft_model_runner.tp_group):
                spec_info = self.draft(batch)
            logits_output, verify_output, model_worker_batch = self.verify(
                batch, spec_info
            )

            # If it is None, it means all requests are finished
            if batch.spec_info.verified_id is not None:
                with self.draft_tp_context(self.draft_model_runner.tp_group):
                    self.forward_draft_extend_after_decode(batch)
            return (
                logits_output,
                verify_output.verified_id,
                model_worker_batch.bid,
                sum(verify_output.accept_length_per_req_cpu),
            )
        elif batch.forward_mode.is_idle():
            model_worker_batch = batch.get_model_worker_batch()
            logits_output, next_token_ids = self.target_worker.forward_batch_generation(
                model_worker_batch
            )

            return logits_output, next_token_ids, model_worker_batch.bid, 0
        else:
            logits_output, next_token_ids, bid = self.forward_target_extend(batch)
            with self.draft_tp_context(self.draft_model_runner.tp_group):
                self.forward_draft_extend(
                    batch, logits_output.hidden_states, next_token_ids
                )
            return logits_output, next_token_ids, bid, 0
```

Three branches for three forward modes: during decode it is "draft → verify → draft extend", returning the target model's logits, the accepted tokens and the total accepted; when idle it runs the target model only (keeping step under DP attention); during extend (prefill) the target model computes first and the draft model then does one extend on the target's hidden states (EAGLE's draft model takes the target model's penultimate layer's hidden states as input). To the scheduler this is a `forward_batch_generation` with one extra return value, the accepted count — and part 3's "small modifications to the general scheduler" changed exactly the few places that handle multi-token output.

![Figure: one decode step of the EAGLEWorker](../assets/figures/sgl-eagle-worker.svg){.aig-svg}

## Drafting, verification and slots {#草稿验证与槽位}

`draft` deals with memory first:

```python title="python/sglang/srt/speculative/eagle_worker.py @ v0.4.6 L304-330" linenums="304"
    def draft(self, batch: ScheduleBatch):
        # Parse args
        num_seqs = batch.batch_size()
        spec_info = batch.spec_info

        # Accumulate penalty
        if batch.sampling_info.penalizer_orchestrator.is_required:
            # This is a relaxed version of penalties for speculative decoding.
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                spec_info.verified_id.to(torch.int64)
            )

        # Allocate cache locations
        if self.page_size == 1:
            out_cache_loc, token_to_kv_pool_state_backup = batch.alloc_token_slots(
                num_seqs * self.topk * self.speculative_num_steps, backup_state=True
            )
        else:
            if self.topk == 1:
                prefix_lens = batch.seq_lens
                seq_lens = prefix_lens + self.speculative_num_steps
                extend_num_tokens = num_seqs * self.speculative_num_steps
            else:
                # In this case, the last partial page needs to be duplicated.
                # KV cache layout in batch.req_to_token_pool.req_to_token:
                #
                # | -------- | -- xxxx .. | -- xxxx .. | -- xxxx .. |
```

With a page size of 1 it allocates `num_seqs x topk x num_steps` slots at once and backs up the allocator's state (rolled back by the acceptance result after verification); with a page size above 1 it computes per request and the last partial page has to be copied — the subject of #4908 "Support page size > 1 + eagle" at the end of March. The draft's `num_steps` steps take the top-k each, `build_eagle_tree.py` expands the candidates into a tree and produces the attention mask and positions for verification, and `verify` runs one extend of the target model, with `EagleVerifyInput` carrying the tree's structure while the kernels (`verify_tree_greedy` and others in `sgl-kernel`) walk the tree to find each request's longest accepted path. The accepted tokens are written back into the `Req`, the rest of the tree's slots are freed, and the draft model does one extend with the accepted tokens. The draft model has CUDA graphs of its own (`eagle_draft_cuda_graph_runner.py`), because each drafting step's batch shape is fixed.

Its relationship with the radix tree is simple: an accepted token is no different from one produced by ordinary decoding, and they all go in when the request finishes. But retraction got more complicated (#2711): retracting a request mid-speculation has to free the tree's slots along with it.

## Design trade-offs {#设计取舍}

- **A worker rather than a branch in the scheduler.** All of speculative decoding's complexity (the tree, the mask, rolling back slots, two models' CUDA graphs) is sealed inside the worker, and the scheduler only has to understand "several tokens in one step". The price is a thick worker (655 lines to start with) and that some decisions that belong in the scheduling layer (adjusting the draft step count by load, say) have to take a detour.
- **Allocate first, roll back after.** Reserving slots for every candidate is simpler than allocating on demand and avoids running short of memory during verification; the price is a high peak, which `alloc_token_slots(..., backup_state=True)`'s backup and rollback pays for.
- **Tree verification in one forward pass.** Against verifying one candidate at a time, one forward pass costs what the batch size and the tree size make it; [the inference-systems handbook's speculative-decoding chapter](serving://topics/speculative/) discusses the trade-off in K.
- **Merged in four steps.** A large feature split into independently reviewable pieces, with the groundwork (the draft model, the compatibility fixes) merged first.

## What happened afterwards {#后来怎么样了}

- 2025-02: MTP / NextN (#3582); DeepSeek-V3's draft head shares the embedding and the LM head with the target model (#4033); March brought the FlashInfer MLA backend (#4218) and the FA3 backend (#4951).
- 2025-03: a page size above 1 (#4908); speculative decoding was reconciled with overlapped scheduling, DP attention and PD disaggregation one by one.
- H2 2025: `eagle_worker_v2.py` — "spec v2" brings drafting and verification into the overlapping flow too, while `base_spec_worker.py` and `spec_registry.py` make speculative methods registrable plugins; `ngram_worker.py` plus n-gram matching in C++ (no draft model needed) and `standalone_worker_v2.py` (a separate small model as the draft).
- The baseline commit's `speculative/` has over 60 files: besides EAGLE, multi-layer EAGLE, MTP and n-gram there are several new methods of 2026 (dflash, uno, dspark in the file names), all connected to the scheduler through the same worker interface.

## Exercises {#练习}

**1. Forward passes in one step.** With `num_steps = 3` and `topk = 4`, how many forward passes does each of the draft and target models make in one decode? How many candidate tokens are on the tree?

??? success "Answer"
    The draft runs 3 times (top-4 each step, continuing from the previous step's candidates from the second onwards, with the node count depending on whether the implementation prunes; v0.4.6's `build_eagle_tree` expands each step's top-k, for at most 4 + 4 + 4 = 12 candidates plus the root), verification is 1 target forward pass (over 1 + the candidate count tokens), and the draft extend is 1 more. The target model running only once is where the speedup comes from.

**2. Why part 2 had to fix DP attention's hang.** Read `git show b0524c3789`'s diff and explain the conflict between speculative decoding's batch shapes and DP attention's all-gather.

??? success "A way to approach it"
    A speculative decode step inputs several tokens, so a rank's token count is no longer its batch size; `prepare_dp_attn_batch` has to gather with the right token count, or some rank waits for a matching collective that never comes and hangs.

**3. Count today's methods.** List every `*_worker*.py` under `speculative/` at the baseline commit, guess which speculative method each is, and confirm the registered names in `spec_registry.py`.

??? success "A way to approach it"
    `git ls-tree --name-only 29f6d408c0 python/sglang/srt/speculative/ | grep worker`, then `git show 29f6d408c0:python/sglang/srt/speculative/spec_registry.py`.

!!! interview "How to explain it"
    "How is speculative decoding integrated into an inference engine?" — Give SGLang's approach: make it a worker (with the draft model inside) exposing the same interface to the scheduler, only returning several tokens per step; allocate the slots first and roll back after; verify once with a tree mask; and give the draft model its own CUDA graphs. Then say where the integration is genuinely hard: retraction, DP attention's gather shapes, the page size, overlapped scheduling — each with its own fixing commit.

## Summary {#小结}

- [x] The four parts merged EAGLE-2 between 2024-12-29 and 2025-01-02, with the large PR split into independently reviewable pieces.
- [x] `EAGLEWorker(TpModelWorker)`: draft → one verification → draft extend, leaving the scheduler to handle only "several tokens per step".
- [x] Slots are allocated first and rolled back after, the tree's kernels went into sgl-kernel, and MTP, n-gram, spec v2 and the later methods all connect through the same worker interface.
