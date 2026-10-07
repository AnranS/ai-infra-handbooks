# Becoming a service: the OpenAI-compatible interface, streaming and multimodal input

<p class="lead">The first version's HTTP service has three routes and a <code>/v1/completions</code> that is a 14-line shell. Half a year later it has a complete OpenAI-compatible layer, chat templates, incremental detokenizing and several vision models. This chapter covers the work of turning a research prototype into a usable service: none of it appears in the paper, yet it decides whether a user stays after their first <code>pip install</code>. Three things get the attention: how the OpenAI adapter turns chat messages into one <code>/generate</code>, why detokenizing has to be incremental, and how the multimodal models were brought in one after another.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. How does OpenAI's `chat/completions` become a native `/generate` request in SGLang? Where does the chat template come from?
    2. Why can streaming not decode every token afresh each time? What are `surr_offset` and `read_offset` in incremental detokenizing?
    3. The first version already supported LLaVA; what new problems did LLaVA-NeXT, video and OneVision each bring?
    4. The route count grew from 3 to 86. What kinds of interface are they mostly?

??? success "Answers for the self-test (answer first, then open this)"
    1. `v1_chat_completions` in `openai_api/adapter.py`: it joins the `messages` into one prompt string with a chat template (the conversation template `--chat-template` names, or the HF tokenizer's own `apply_chat_template`), collects the template's stop strings and images, builds a `GenerateReqInput` for the `TokenizerManager`, and finally wraps the result in OpenAI's response format (including streaming SSE chunks).
    2. Decoding everything is O(the length generated), so each token's cost grows linearly on a long answer; and many tokenizers have tokens that must be decoded together with the preceding one to get the spacing or the bytes right. The incremental scheme records two offsets: the tokens after `surr_offset` are "the window decoded with context" and those after `read_offset` are "the new tokens not yet confirmed as output"; each round decodes only the window and subtracts the confirmed part from the window's text to get the new text, then advances both offsets once it is confirmed.
    3. LLaVA-NeXT's (llava-hd) anyres cuts one image into several tiles and the token count varies with the resolution, so `pad_input_ids` has to pad by the actual tile count; video is many frames and still more tokens; OneVision changed the vision encoder (SigLIP) and the language model (Qwen2) and supports several images — the model files went from one each to a single composable file (#475).
    4. OpenAI compatibility (completions, chat, embeddings, models, batches, responses, rerank…), runtime control (flush cache, update weights, abort, profiler, server args), health and metrics (health, metrics), and the native generate / encode / classify.

A six-panel strip first, then the chapter:

<!-- comic ../assets/comics/api-multimodal.webp is in Chinese; put it back once the English version exists -->

## From a 14-line shell to an adapter layer {#从-14-行的壳到适配层}

The first version's `/v1/completions`:

```python title="python/sglang/srt/server.py @ 22085081bb L62-76" linenums="62"
@app.post("/v1/completions")
async def v1_completions(obj: CompletionRequest):
    assert obj.n == 1
    obj = GenerateReqInput(
        text=obj.prompt,
        sampling_params={
            "temperature": obj.temperature,
            "max_new_tokens": obj.max_tokens,
            "stop": obj.stop,
        },
    )
    ret = await generate_request(obj)
    return {
        "choices": [{"text": ret["text"]}],
    }
```

It moves `prompt`, `temperature`, `max_tokens` and `stop` into a `GenerateReqInput` and returns `choices[0].text`: no streaming, no `n > 1`, no logprobs. Becoming a service really began on 18 January: Cody Yu's #49 added `stream=True` to completions and #50 added `/v1/chat/completions`; #113 of 30 January let the chat interface accept image content; and #667 "Update OpenAI API" of 19 July tidied all of it into a separate `openai_api/adapter.py` (432 lines) and `openai_api/protocol.py` (pydantic models). How the route count grew:

```bash title="routes-per-tag.sh"
REF=${REF:-29f6d408c0}
for t in v0.1.5 v0.2.0 v0.4.0; do printf '%-8s %3d 个路由（srt/server.py）\n' $t "$(git show $t:python/sglang/srt/server.py | grep -c '^@app\.')"; done
printf '%-8s %3d 个路由（srt/entrypoints/http_server.py）\n' "$REF" "$(git show "$REF:python/sglang/srt/entrypoints/http_server.py" | grep -c '^@app\.')"
```

```text title="output"
v0.1.5     3 个路由（srt/server.py）
v0.2.0     7 个路由（srt/server.py）
v0.4.0    29 个路由（srt/server.py）
29f6d408c0  86 个路由（srt/entrypoints/http_server.py）
```

The heart of v0.2.0's chat interface is "messages → template → prompt":

```python title="python/sglang/srt/openai_api/adapter.py @ v0.2.0 L257-290" linenums="257"
async def v1_chat_completions(tokenizer_manager, raw_request: Request):
    request_json = await raw_request.json()
    request = ChatCompletionRequest(**request_json)

    # Prep the data needed for the underlying GenerateReqInput:
    #  - prompt: The full prompt string.
    #  - stop: Custom stop tokens.
    #  - image_data: None or a list of image strings (URLs or base64 strings).
    #    None skips any image processing in GenerateReqInput.
    if not isinstance(request.messages, str):
        # Apply chat template and its stop strings.
        if chat_template_name is None:
            prompt = tokenizer_manager.tokenizer.apply_chat_template(
                request.messages, tokenize=False, add_generation_prompt=True
            )
            stop = request.stop
            image_data = None
        else:
            conv = generate_chat_conv(request, chat_template_name)
            prompt = conv.get_prompt()
            image_data = conv.image_data
            stop = conv.stop_str or []
            if request.stop:
                if isinstance(request.stop, str):
                    stop.append(request.stop)
                else:
                    stop.extend(request.stop)
    else:
        # Use the raw prompt and stop strings if the messages is already a string.
        prompt = request.messages
        stop = request.stop
        image_data = None

    adapted_request = GenerateReqInput(
```

Two paths: without `--chat-template` it uses the HF tokenizer's own `apply_chat_template`; with one it uses SGLang's own conversation template (the `Conversation` object in `lang/chat_template.py`, reused from the frontend language), and the template also supplies the stop strings and extracts the images. After that it is an ordinary `GenerateReqInput` — **the OpenAI-compatible layer is a layer of translation above the runtime, and the runtime itself knows only the native interface**. That separation has held: the OpenAI layer was rewritten entirely as `entrypoints/openai/` in June 2025 (chapter 20) while the runtime's `/generate` barely changed.

## Streaming and incremental detokenizing {#流式输出与增量反分词}

Streaming looks like no more than sending the result out in chunks, but it involves all three processes. Several commits in January fix exactly this: #30 "Fix streaming", #95 "Improve Chinese character streaming when the last char is half Chinese word" (when a character's UTF-8 bytes are cut across two tokens, half a character must not be emitted) and #117 "Improve the control of streaming and improve the first token latency" (`--stream-interval` controls how many decode steps between pushes).

The more fundamental problem is in the detokenizer process. The first version decodes **all** of a request's output tokens afresh every time:

```python title="python/sglang/srt/managers/detokenizer_manager.py @ 22085081bb L37-59" linenums="37"
            if isinstance(recv_obj, BatchTokenIDOut):
                output_tokens = recv_obj.output_tokens

                # TODO(lmzheng): handle skip_special_tokens per request
                output_strs = self.tokenizer.batch_decode(
                    output_tokens,
                    skip_special_tokens=recv_obj.skip_special_tokens[0],
                )

                # Trim stop str
                # TODO(lmzheng): handle the case where multiple stop strs are hit
                for i in range(len(output_strs)):
                    if recv_obj.hit_stop_str[i] is not None:
                        pos = output_strs[i].find(recv_obj.hit_stop_str[i])
                        if pos != -1:
                            output_strs[i] = output_strs[i][:pos]

                    if len(output_tokens[i]) > 0:
                        first_token = self.tokenizer.convert_ids_to_tokens(
                            int(output_tokens[i][0])
                        )
                        if first_token.startswith("▁"):
                            output_strs[i] = " " + output_strs[i]
```

`batch_decode(output_tokens)`'s cost grows linearly with the output's length, so a 1000-token answer decodes 1000 tokens at every step towards the end; the `"▁"` handling at the end shows the author already wrestling with the tokenizer's boundary problems. #517 "Decode Incrementally" of 2024-06-12 replaced it with today's scheme:

```python title="python/sglang/srt/managers/detokenizer_manager.py @ v0.2.0 L57-106" linenums="57"
            # Initialize decode status
            read_ids, surr_ids = [], []
            for i in range(bs):
                rid = recv_obj.rids[i]
                vid = recv_obj.vids[i]
                if rid not in self.decode_status or self.decode_status[rid].vid != vid:
                    s = DecodeStatus(
                        vid=vid,
                        decoded_text=recv_obj.decoded_texts[i],
                        decode_ids=recv_obj.decode_ids[i],
                        surr_offset=0,
                        read_offset=recv_obj.read_offsets[i],
                    )
                    self.decode_status[rid] = s
                else:
                    s = self.decode_status[rid]
                    s.decode_ids = recv_obj.decode_ids[i]

                read_ids.append(s.decode_ids[s.surr_offset :])
                surr_ids.append(s.decode_ids[s.surr_offset : s.read_offset])

            # TODO(lmzheng): handle skip_special_tokens/spaces_between_special_tokens per request
            surr_texts = self.tokenizer.batch_decode(
                surr_ids,
                skip_special_tokens=recv_obj.skip_special_tokens[0],
                spaces_between_special_tokens=recv_obj.spaces_between_special_tokens[0],
            )
            read_texts = self.tokenizer.batch_decode(
                read_ids,
                skip_special_tokens=recv_obj.skip_special_tokens[0],
                spaces_between_special_tokens=recv_obj.spaces_between_special_tokens[0],
            )

            # Trim stop str
            # TODO(lmzheng): handle the case where multiple stop strs are hit
            output_strs = []
            for i in range(bs):
                s = self.decode_status[recv_obj.rids[i]]
                new_text = read_texts[i][len(surr_texts[i]) :]
                if recv_obj.finished_reason[i] is None:
                    # Streaming chunk: update the decode status
                    if len(new_text) > 0 and not new_text.endswith("�"):
                        s.decoded_text = s.decoded_text + new_text
                        s.surr_offset = s.read_offset
                        s.read_offset = len(s.decode_ids)
                        new_text = ""
                    else:
                        new_text = find_printable_text(new_text)

                output_strs.append(s.decoded_text + new_text)
```

Each request keeps a `DecodeStatus`: `decoded_text` is the confirmed text, `decode_ids` is every token, and the two offsets `surr_offset` and `read_offset` cut the token sequence into three parts — `[:surr_offset]` is confirmed and already emitted, `[surr_offset:read_offset]` is "the few surrounding tokens" (which have to be decoded with the new tokens to get the spacing right) and `[read_offset:]` is the new tokens. Each round decodes two windows, `surr_ids` (the surroundings) and `read_ids` (the surroundings plus the new), and the latter's text minus the former's is what is new this time; the new text is confirmed only if it does not end in the replacement character `�` (which would mean a cut inside a multi-byte character), and then both offsets advance. The algorithm shares its ancestry with HF's `TextIteratorStreamer` and vLLM's `detokenize_incrementally`; SGLang's version lives in its own process, and the scheduler only has to send the `read_offset` and a version number `vid` (which resets the state when a request is jump-forwarded or retried).

![Figure: incremental detokenizing's two offsets](../assets/figures/sgl-incremental-decode.svg){.aig-svg}

## Multimodal: from LLaVA to OneVision {#多模态从-llava-到-onevision}

Multimodal is not a feature added later: LLaVA is one of the first version's three models ([chapter two](../origins/first-commit.md) covered the trick of mapping an image onto a fixed run of tokens with a hash). The commits around it in the first half of 2024:

```bash title="multimodal-commits.sh"
REF=${REF:-29f6d408c0}
git log --reverse --date=short --format='%ad  %h  %s' "$REF" | awk '$1 <= "2024-08-31"' | grep -iE 'llava|yi-vl|vision|video' | cut -c1-96 | head -18
```

```text title="output"
2024-01-18  98a3e8ef78  Add a llava example (#47)
2024-01-22  e08bca2840  Support load fine-tuned LLaVA model (#80)
2024-01-24  c6576e820c  Llava-hd Support (#92)
2024-01-24  bef0b35902  Fix llava & Fix multiprocessing
2024-01-24  711d343530  add a batch llava example
2024-01-25  0147f940dd  fix batch error for llava-hd (#98)
2024-01-31  c7af9f7393  Fix a bug in llava-hd
2024-02-01  864425300f  Yi-VL Model (#112)
2024-02-01  03e04b2331  update docs for Yi-VL
2024-02-03  bb3a3b6675  Support Faster JSON decoding for llava (#137)
2024-02-11  c51020cf0c  Fix the chat template for llava-v1.6-34b & format code (#177)
2024-03-29  cb389c91bc  Fix llava parallelism/fork bug (#315)
2024-05-14  0992d85f92  support llava video (#426)
2024-05-14  664287b2a7  [Feat] Add llava qwen, llava mistral (#419)
2024-05-24  3167d8dabc  fix test bug in srt_llava_next_test.py (#470)
2024-05-24  44c998fcb5  Add the instruction link to the LLaVA-NeXT-Video at README (#463)
2024-05-27  2b605ab1d7  [Feat/Fix] Refactoring Llava models into single file (#475)
2024-06-01  7d1ebc2d71  update the script: examples/usage/llava_video/srt_example_llava_v.sh (#4
```

Every step challenges the assumption that an image is a fixed-length run of tokens:

- **llava-hd (#92, 2024-01-24)**: LLaVA-NeXT's anyres cuts a high-resolution image into several tiles, each with its own group of tokens, so the total varies with the resolution. `mm_utils.py` (adapted from LLaVA's official repository: `process_anyres_image`, `expand2square`, `select_best_resolution`) does the tiling, `pad_input_ids` pads by the actual tile count, and `image_offset` records where the image's tokens begin.
- **Yi-VL (#112) and images in the OpenAI chat interface (#113)**: different models have different image placeholders and templates, so the template system has to describe where an image goes and with which tokens.
- **Video (#426, 2024-05-14) and llava-qwen / llava-mistral (#419)**: frames times tokens per frame makes the prompt very long; the language model can be swapped while the vision tower stays — #475 merged the scattered LLaVA variants into one file that composes the vision encoder and the language model by configuration.
- **LLaVA-OneVision (2024-08-24)**: a SigLIP encoder plus a Qwen2 decoder, with several images as input (#1205 fixed the errors with several images). The v0.3 blog post lists it among the headline features.

The growth in the model count can be counted from the directory too:

```bash title="models-per-tag.sh"
REF=${REF:-29f6d408c0}
for t in v0.1.5 v0.2.0 v0.3.0 v0.4.0 v0.4.6 v0.5.0rc0 "$REF"; do printf '%-11s %4d 个模型文件\n' "$t" "$(git ls-tree -r --name-only "$t" -- python/sglang/srt/models | grep -c '\.py$')"; done
```

```text title="output"
v0.1.5         3 个模型文件
v0.2.0        22 个模型文件
v0.3.0        25 个模型文件
v0.4.0        41 个模型文件
v0.4.6        62 个模型文件
v0.5.0rc0     94 个模型文件
29f6d408c0   287 个模型文件
```

## Design trade-offs {#设计取舍}

- **A translation layer rather than two interfaces.** The OpenAI-compatible layer only converts formats, and every capability (streaming, logprobs, constrained decoding, images) is implemented on the native `/generate` first and then exposed through it. The benefit is one copy of the runtime logic; the drawback is that some notions in OpenAI's semantics (`n`, `best_of`, tool calls) have to be assembled in the translation layer.
- **A separate detokenizer process plus an incremental algorithm.** A tokenizer is pure CPU work and holds the GIL, so inside the scheduling process it fights the GPU's launches; once it has its own process, the incremental algorithm makes each step's work independent of the output's length.
- **Two sources of templates.** SGLang's own templates (controllable, able to describe image placeholders) and HF's (broad coverage) coexist, selected with `--chat-template`. That duality lasted a long time, until the HF templates became the norm.
- **Multimodal goes "preprocess first, a placeholder in the middle, the features last".** An image is preprocessed into a tensor and hashed in the `TokenizerManager`, the scheduler sees only the placeholder tokens (so the prefix cache applies), and the visual features are filled into the right positions during the model's forward pass. The later `multimodal/` directory (2025-07) split the preprocessors by model while keeping this pipeline.

## What happened afterwards {#后来怎么样了}

- 2024-11 → 2025-06: `openai_api/` kept gaining interfaces (embeddings, batch, parsing for function calling), and #7167 of 2025-06 rewrote it wholesale as `entrypoints/openai/serving_*.py`, one class per kind of interface.
- 2025-05: a `function_call/` directory with tool-call format parsers for a dozen or so models; a gRPC entry point added in 2025-09.
- 2025-07: `multimodal/processors/`, one processor per model family, with `multimodal_cache.py` caching visual features by hash.
- Detokenizing: `detokenizer_manager.py` is still its own process today, the incremental algorithm's core logic is almost exactly v0.2.0's, with per-request `skip_special_tokens` and support for several tokenizers added.

## Exercises {#练习}

**1. Streaming's first token.** Read #117's diff (`git show 6f560c761b`) and explain why `stream_interval`'s default changed from 2 to something else, and how "the first token's latency" improved.

??? success "A way to approach it"
    A streaming request pushes its first token right after the prefill rather than waiting several decode steps; `--stream-interval` only controls the interval between the pushes that follow. While reading the diff, watch the test on `req.stream` in `handle_finished_requests`.

**2. Incremental detokenizing's boundary.** Construct a case where a new token decodes to text ending in `�`. What happens to `new_text`? How does the next step recover?

??? success "Answer"
    It is not confirmed (neither `decoded_text` nor the offsets move) and `find_printable_text` emits the printable part first; when the next token arrives the window grows, the multi-byte character is complete, and decoding the whole window again gives the right text.

**3. Count the compatibility layer.** Count the `serving_*.py` files in `entrypoints/openai/` at the baseline commit and which kind of OpenAI interface each one covers.

??? success "A way to approach it"
    `git ls-tree --name-only 29f6d408c0 python/sglang/srt/entrypoints/openai/` lists serving_chat, serving_completions, serving_embedding, serving_rerank, serving_responses, serving_score and others, matching OpenAI's chat, completions, embeddings and the later responses API one for one.

!!! interview "How to answer in an interview"
    Asked how streaming is implemented, do not stop at SSE. Give three layers: the scheduler pushes token ids every `stream_interval`; the detokenizer process decodes incrementally with two offsets, handling multi-byte characters and spacing; the HTTP layer wraps the text in OpenAI's chunk format. Then add that "the OpenAI-compatible layer is only a translation above the native interface", which shows you know where the boundary is.

## Summary {#小结}

- [x] Becoming a service began in January 2024: streaming completions (#49), chat completions (#50), image input (#113); it was tidied into the `openai_api/` adapter in July, with the compatible layer doing translation only.
- [x] Incremental detokenizing (#517) uses the two offsets `surr_offset` and `read_offset` to make each step's decoding cost constant, and handles multi-byte characters correctly.
- [x] Multimodal started from LLaVA, and anyres, video and OneVision broke the assumption that an image is a fixed run of tokens step by step — but the pipeline of "preprocess first, a placeholder in the middle, the features last" survives to this day.
