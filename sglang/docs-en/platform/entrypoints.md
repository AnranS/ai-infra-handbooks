# Rewriting the entry layer: entrypoints, the OpenAI server's restructuring, function calling and gRPC

<p class="lead">Before 2025 SGLang's "entry" was one <code>server.py</code>: FastAPI, starting the processes and the offline <code>Runtime</code> all in one file. #2996 of 19 January 2025 split it into <code>entrypoints/engine.py</code> (the offline engine, and the only place the subprocesses are started) and <code>http_server.py</code>; #7167 of 16 June rewrote the OpenAI-compatible layer in 4400 lines as <code>serving_*</code> classes by interface; from May a <code>function_call/</code> directory collected tool-call parsers for a dozen or so models; and #10283 of 11 September added a standalone gRPC server so the Rust gateway could talk to the scheduler directly, bypassing Python's HTTP. This chapter covers why the entry layer went from one file to a directory, and whom it serves: HTTP clients, offline scripts, RL frameworks and the gateway.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. How do `Engine` and `http_server` relate? Through which entry does an RL framework come in, and through which the gateway?
    2. What structure did June 2025's OpenAI-layer restructuring change? Why was the old `adapter.py` not enough?
    3. Why does tool-call parsing need one parser per model family? What does it cover and what does constrained decoding ([chapter four](../origins/fsm-jump.md)) cover?
    4. What problem does the gRPC entry solve? With both entries present, who uses each?

??? success "Answers for the self-test (answer first, then open this)"
    1. `Engine` (`entrypoints/engine.py`) starts the scheduler and detokenizer subprocesses, holds the `TokenizerManager` and offers Python methods like `generate` and `update_weights_*`; `http_server.py` hangs FastAPI routes on top of an `Engine`. An RL framework calls `Engine` directly in its own process (or through a wrapper like `verl_engine.py`); the gateway comes in over HTTP, or over gRPC straight to the scheduler after 2025-09.
    2. From one 432-line `adapter.py` to classes by interface under `entrypoints/openai/`: `serving_base.py` defines the common flow (parse the request → convert to an internal request → call → wrap the response, with a streaming and a non-streaming path), and `serving_chat.py`, `serving_completions.py` and `serving_embedding.py` implement the details, with `protocol.py` the pydantic models and 1800 lines of tests. The old adapter crammed every interface into two functions of a few hundred lines, and each new feature (tool calls, a reasoning model's thinking segment, multimodal input, logprob formats) added a branch to the same function.
    3. Each model family emits tool calls in its own markup (a `<tool_call>` tag, a JSON array, special tokens…), so a parser has to be written per format, and it has to support streaming (what to do with half a JSON object). Constrained decoding guarantees that "only a valid format is generated"; a parser turns the generated text into a structured call. They can work together (the structural tag), but their duties differ.
    4. Python's HTTP layer (FastAPI plus pydantic plus the `TokenizerManager`) is a bottleneck under high concurrency, and the gateway has already parsed the request and rendered the chat template; the gRPC entry lets the gateway send an already-tokenized request straight to the scheduler, saving a layer. The HTTP entry goes on serving ordinary clients and OpenAI-compatible SDKs.

A six-panel strip first, then the chapter:

<!-- comic ../assets/comics/entrypoints.webp is in Chinese; put it back once the English version exists -->

## From server.py to entrypoints/ {#从-serverpy-到-entrypoints}

```bash title="entrypoints-dir.sh"
REF=${REF:-29f6d408c0}
git log --reverse --date=short --format='%ad  %h  %s' "$REF" -- python/sglang/srt/entrypoints | head -1 | cut -c1-96
for t in v0.4.6 v0.5.0rc0 "$REF"; do
  printf '%-11s %2d 个文件：' "$t" "$(git ls-tree -r --name-only "$t" -- python/sglang/srt/entrypoints | grep -c '\.py$')"
  git ls-tree -r --name-only "$t" -- python/sglang/srt/entrypoints | sed 's|python/sglang/srt/entrypoints/||' | awk -F/ '{print (NF > 1 ? $1 "/" : $1)}' | sort -u | tr '\n' ' ' | cut -c1-150; echo
done
```

```text title="output"
2025-01-19  03464890e0  Separate two entry points: Engine and HTTP server (#2996)
v0.4.6       5 个文件：EngineBase.py engine.py http_server.py http_server_engine.py verl_engine.py 

v0.5.0rc0   19 个文件：EngineBase.py context.py engine.py harmony_utils.py http_server.py http_server_engine.py openai/ tool.py 

29f6d408c0  66 个文件：EngineBase.py anthropic/ context.py elastic_ep.py engine.py engine_info_bootstrap_server.py engine_score_mixin.py grpc_bridge.py grpc_server.py harmon
```

#2996's title states the intent: "Separate two entry points: Engine and HTTP server". Before it, `Runtime` (the offline interface present since the first version, [chapter two](../origins/first-commit.md)) started an HTTP service in a subprocess and called itself over HTTP; after the split, `Engine` holds the `TokenizerManager` and calls in-process, with the HTTP service only a layer of routes around it. `EngineBase.py` defines the interface they share, `http_server_engine.py` implements the same interface over an HTTP client (for cases wanting a "remote Engine"), and `verl_engine.py` (added 2025-03, removed in June, [chapter 22](rl.md)) was once the wrapper for RL frameworks. v0.5.0rc0 adds an `openai/` directory, `context.py` (the request context) and `harmony_utils.py` (gpt-oss's harmony format); the baseline commit adds `anthropic/` (compatibility with the Anthropic Messages API), `grpc_*` and `elastic_ep.py`.

## Rewriting the OpenAI layer {#openai-层的重写}

```bash title="openai-refactor.sh"
git show --stat=100 --format='%ad  %an  %s' --date=short 70c471a868 | grep -v '^$' | cut -c1-96
```

```text title="output"
2025-06-16  Xinyuan Tong  [Refactor] OAI Server components (#7167)
 python/sglang/srt/entrypoints/openai/__init__.py            |   0
 python/sglang/srt/entrypoints/openai/protocol.py            | 539 ++++++++++++++++++
 python/sglang/srt/entrypoints/openai/serving_base.py        | 178 ++++++
 python/sglang/srt/entrypoints/openai/serving_chat.py        | 938 +++++++++++++++++++++++++++++
 python/sglang/srt/entrypoints/openai/serving_completions.py | 467 ++++++++++++++++
 python/sglang/srt/entrypoints/openai/serving_embedding.py   | 227 ++++++++
 python/sglang/srt/entrypoints/openai/utils.py               | 264 +++++++++
 test/pytest.ini                                             |   2 +
 test/srt/openai/test_protocol.py                            | 683 +++++++++++++++++++++++
 test/srt/openai/test_serving_chat.py                        | 634 +++++++++++++++++++++
 test/srt/openai/test_serving_completions.py                 | 176 ++++++
 test/srt/openai/test_serving_embedding.py                   | 316 +++++++++++
 12 files changed, 4424 insertions(+)
```

12 files, 4424 lines and no deletions — a restructuring by "build the new road alongside, then switch over": the new directory built, enough tests written (1800 lines of tests, over 40% of the implementation), the routes pointed at it, and the old `openai_api/` deleted afterwards. The structure:

```bash title="openai-serving.sh"
REF=${REF:-29f6d408c0}
for f in $(git ls-tree -r --name-only "$REF" -- python/sglang/srt/entrypoints/openai | grep '\.py$'); do printf '%5d  %s\n' "$(git show "$REF:$f" | wc -l)" "${f#python/sglang/srt/entrypoints/openai/}"; done
echo "-- serving_base.py 的方法："; git show "$REF:python/sglang/srt/entrypoints/openai/serving_base.py" | grep -E '^    (async )?def ' | sed 's/^ *//; s/(.*//' | tr '\n' ' '; echo
```

```text title="output"
    0  __init__.py
  102  audio_chunking.py
  289  chat_encoding.py
  469  encoding_dsv32.py
  884  encoding_dsv4.py
  705  encoding_dsv41.py
 2352  protocol.py
    5  realtime/__init__.py
  120  realtime/handler.py
   78  realtime/protocol.py
  741  realtime/session.py
  215  responses_adapters.py
  290  serving_base.py
 3251  serving_chat.py
  204  serving_classify.py
  694  serving_completions.py
  535  serving_decisions.py
  308  serving_embedding.py
  609  serving_rerank.py
 2774  serving_responses.py
  120  serving_score.py
  194  serving_tokenize.py
  809  serving_transcription.py
   99  sse_utils.py
  209  streaming_asr.py
  188  tool_server.py
   39  transcription_adapters/__init__.py
  198  transcription_adapters/base.py
   91  transcription_adapters/glmasr.py
   63  transcription_adapters/granite_speech.py
   46  transcription_adapters/mimo_v2_asr.py
   61  transcription_adapters/qwen2_audio.py
   69  transcription_adapters/qwen3_asr.py
  382  transcription_adapters/whisper.py
  126  usage_processor.py
  389  utils.py
-- serving_base.py 的方法：
def __init__ def _parse_model_parameter def _resolve_lora_path async def handle_request def _request_id_prefix def _generate_request_id_base def _convert_to_internal_request async def _handle_streaming_request async def _handle_non_streaming_request def _validate_request def create_error_response def create_streaming_error_response def extract_custom_labels def extract_routing_key def extract_routed_dp_rank_from_header 
```

`serving_base.py` defines the template method: validate the request → convert it into an internal `GenerateReqInput` → call the `TokenizerManager` → wrap the response for streaming or not; each interface class implements only "how to convert and how to wrap". The responses API, score, rerank and tokenize added later are each one more `serving_*.py`. [Chapter seven](../service/api-multimodal.md) stated the principle that "the compatible layer only translates", and this restructuring made it a class hierarchy.

## Tool calling: a parser per model family {#工具调用按模型族的解析器}

```bash title="function-call.sh"
REF=${REF:-29f6d408c0}
git log --reverse --date=short --format='%ad  %h  %s' "$REF" -- python/sglang/srt/function_call | head -1 | cut -c1-96
echo "今天 $(git ls-tree -r --name-only "$REF" -- python/sglang/srt/function_call | grep -c '\.py$') 个文件，其中 detector：$(git ls-tree -r --name-only "$REF" -- python/sglang/srt/function_call | grep -c '_detector\.py$') 个"
git ls-tree -r --name-only "$REF" -- python/sglang/srt/function_call | grep '_detector\.py$' | sed 's|.*/||; s|_detector\.py||' | tr '\n' ' ' | cut -c1-200; echo
```

```text title="output"
2025-05-23  ed0c3035cd  feat(Tool Calling): Support `required` and specific function mode (#6550
今天 47 个文件，其中 detector：37 个
apertus2509 base_format cohere_command4 deepseekv31 deepseekv32 deepseekv3 deepseekv41 deepseekv4 dots gemma4 gigachat35 gigachat3 glm47_moe glm4_moe gpt_oss hermes hunyuan inkling internlm iquest_q1 
```

Each detector covers one model family's tool-call format: find the call's start marker in the output text, parse the JSON arguments, and handle streaming (emitting only complete fragments as they arrive). `--tool-call-parser` picks the parser. The structural tag of #3566 (February 2025) lets constrained decoding guarantee that "the tool-call part is valid JSON", while the parser turns it into OpenAI-format `tool_calls` — the generation phase and the parsing phase each cover their own stretch. From August 2025 the gateway has its own parsers in Rust ([the next chapter](gateway.md)), and both coexist.

## gRPC: a second entry for the gateway {#grpc给网关的第二条入口}

#10283 of 11 September 2025, "Implement Standalone gRPC Server for SGLang Python Scheduler": a gRPC service connecting straight to the scheduler's ZMQ ports, with the requests defined in protobuf (`proto/` at the repository root) and the tokenizing and template rendering done on the gateway's side. The last stretch of the route count's growth, and the gRPC files:

```bash title="routes-and-grpc.sh"
REF=${REF:-29f6d408c0}
for t in v0.4.6 v0.5.0rc0 "$REF"; do printf '%-11s %2d 个 HTTP 路由\n' "$t" "$(git show "$t:python/sglang/srt/entrypoints/http_server.py" | grep -c '^@app\.')"; done
echo "gRPC 相关文件：$(git ls-tree -r --name-only "$REF" -- python/sglang/srt/entrypoints python/sglang/srt/grpc proto | grep -iE 'grpc|\.proto$' | sed 's|.*/||' | tr '\n' ' ')"
```

```text title="output"
v0.4.6      41 个 HTTP 路由
v0.5.0rc0   49 个 HTTP 路由
29f6d408c0  86 个 HTTP 路由
gRPC 相关文件：sglang.proto grpc_bridge.py grpc_server.py 
```

On the gateway's side are `rust/sglang-grpc` and `sglang-renderer` (rendering chat templates in Rust) — the tokenizing and the template rendering moved from Python's `TokenizerManager` into Rust, leaving Python with the scheduling and the execution. This is the start of "the control plane moving up": the HTTP entry serves ordinary users and the gRPC entry serves the gateway.

![Figure: the four kinds of client the entry layer serves](../assets/figures/sgl-entry-layers.svg){.aig-svg}

## Design trade-offs {#设计取舍}

- **The Engine is the only place subprocesses start.** The HTTP service, offline scripts, RL frameworks and the gRPC service all start their processes through `Engine`, so there is one copy of the startup logic; the price is that `engine.py` becomes a thousand-line compendium of startup.
- **Build the new road before switching.** The OpenAI layer's rewrite deleted nothing and came with tests, and the old directory was cleaned up after the switch — the safe way to do a large restructuring.
- **A parser per model family.** There is no common tool-call format, so they have to be written one at a time; the price is that the detector files grow with the model count.
- **Two entries coexisting.** gRPC is fast but only for the gateway; HTTP keeps the compatibility.

## What happened afterwards {#后来怎么样了}

- 2025-08: gpt-oss's harmony format and separating a reasoning model's thinking segment (`parser/reasoning_parser.py`).
- H2 2025: the responses API, compatibility with the Anthropic Messages API, and the score and rerank interfaces.
- The gRPC path became one of the gateway's default data planes, and `rust/sglang-server` (71 files) is a service layer implemented in Rust ([the next chapter](gateway.md)).
- `entrypoints/` has 66 files at the baseline commit, and `http_server.py` has 86 routes.

## Exercises {#练习}

**1. Classify the routes.** Use `git show 29f6d408c0:python/sglang/srt/entrypoints/http_server.py | grep -E '^@app\.(get|post|put|delete)' | sed 's/.*("\([^"]*\)".*/\1/'` to list every route, and count them in the groups "OpenAI-compatible / native generation / runtime control / health and metrics".

??? success "A way to approach it"
    `/v1/*` is the compatible layer; `/generate`, `/encode` and `/classify` are native; `/update_weights*`, `/flush_cache`, `/abort_request`, `/release_memory_occupation` and the like are runtime control (mostly for RL and operations); `/health*`, `/metrics` and `/get_server_info` are health and metrics.

**2. The template method.** Read the baseline commit's `serving_base.py` and draw the sequence of method calls one chat request goes through, marking which are overridden in a subclass.

??? success "A way to approach it"
    `handle_request` → `_validate_request` → `_convert_to_internal_request` → `_handle_streaming_request` / `_handle_non_streaming_request` → `_build_*_response`; the conversion and the response building are overridden in the subclasses.

**3. One detector.** Take `qwen25_detector.py` and say which start and end markers it recognises, and how `parse_streaming_increment` handles a JSON object that has only half arrived.

??? success "A way to approach it"
    Qwen2.5 wraps its JSON in `<tool_call>` … `</tool_call>`; while streaming it buffers the text and emits an increment only when a complete function name and some arguments can be parsed, finishing when `</tool_call>` arrives.

!!! interview "How to answer in an interview"
    "How should an inference service's API layer be designed?" — Answer with SGLang's entry layer: one Engine as the only startup point, with HTTP, gRPC and Python calls as shells around it; an OpenAI-compatible layer that only translates, with a template method fixing "validate → convert → call → wrap" and one class per interface; tool-call formats parsed per model family, with a division of labour against constrained decoding; and under high concurrency the control plane moving up into a Rust gateway while Python keeps the scheduling.

## Summary {#小结}

- [x] #2996 (2025-01-19): `Engine` separated from `http_server`, with `Engine` the only place subprocesses start; RL frameworks and offline scripts use `Engine` directly.
- [x] #7167 (2025-06-16): the OpenAI layer rewritten as a `serving_*` class hierarchy, 4424 lines with no deletions and 1800 lines of tests.
- [x] `function_call/` parses tool calls per model family; #10283 (2025-09-11)'s gRPC entry lets the gateway bypass Python's HTTP, moving the tokenizing and the template rendering up into Rust.
