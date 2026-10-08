# The Rust gateway: from router to sgl-model-gateway

<p class="lead">The Rust router of October 2024 had five source files and two policies; at the baseline commit <code>sgl-model-gateway/</code> has over 400 files and nearly 95 thousand lines of Rust, releases on its own (<code>gateway-v*</code> tags), does cache-aware and PD-aware routing, service discovery, tool-call and reasoning-segment parsing, metrics and tracing, and even has a WASM plugin layer; and beside it the <code>rust/</code> directory holds six more Rust crates the Python package compiles on demand (gRPC, multimodal, the processor, the radix tree, template rendering, the service layer). This chapter follows that line through its three renamings, <code>rust/</code> → <code>sgl-router/</code> → <code>sgl-model-gateway/</code>: how Rust went from "a router" to the control plane in SGLang.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What change of role does each of the router's three renamings mark?
    2. What problem did the dependency-injection restructuring of July 2025 (#7987) solve? How were the policy and the router separated?
    3. What extra work does PD-aware routing take? How does the gateway know which instances are prefill and which decode?
    4. Why did tokenizing, template rendering and tool-call parsing move from Python to Rust? How do the crates in `rust/` relate to the gateway?

??? success "Answers for the self-test (answer first, then open this)"
    1. `rust/` (2024-10): an experimental router living with the engine. `sgl-router` (2024-12): a separate PyPI package, positioned as "SGLang's router". `sgl-model-gateway` (2025-12): a model gateway — not only routing but protocol conversion (OpenAI / Anthropic / native), parsing, service discovery and observability, aimed at several engines and several backends.
    2. Each policy used to be a branch of a `Router` enum, with the policy logic mixed into the HTTP forwarding and the worker management; #7987 abstracted "which worker to pick" into a `LoadBalancingPolicy` trait (round robin, random, cache-aware, power-of-two and others, one implementation each), leaving `Router` to hold the worker list and forward, with the policy injected as a dependency and the PD router reusing the same set.
    3. It has to pick a prefill instance and a decode instance at once and attach bootstrap information to the request (the room id, prefill's address) — what [chapter 17](../scale/pd.md)'s `mini_lb` did — keeping load and cache state for both kinds separately. An instance's role is declared at registration (a startup argument or a service-discovery label).
    4. Under high concurrency Python's GIL and asyncio are a bottleneck, while this work (tokenizing, templates, parsing) is pure CPU and has nothing to do with scheduling; in Rust the gateway sends an already-tokenized request straight to the scheduler over gRPC. The crates under `rust/` are extensions the Python package compiles on demand (`srt/rust_extensions/`): the radix tree's core, multimodal preprocessing, template rendering, gRPC and the service layer — shared by the gateway and the engine.

A six-panel strip first, then the chapter:

<!-- comic ../assets/comics/gateway.webp is in Chinese; put it back once the English version exists -->

## Three renamings {#三次改名}

```bash title="gateway-timeline.sh"
for h in 3839be2913 530ff541cf cbedd1db1d 2e4a5907c9 e3b3acfa6f c8f31042a8 8c7bb39dfb 6e316588f8 9768c50d90 53ca15529a 49dfa1d891; do
  git log -1 --date=short --format='%ad  %h  %s' $h | cut -c1-96
done
```

```text title="output"
2024-10-28  3839be2913  [Router] Add a rust-based router (#1790)
2024-11-04  530ff541cf  [router] Impl radix tree and set up CI (#1893)
2024-11-23  cbedd1db1d  [router] cache-aware load-balancing router v1 (#2114)
2024-12-11  2e4a5907c9  [router] Release router 0.1.0 with dynamic scaling and fault tolerance (
2024-12-12  e3b3acfa6f  Rename rust folder to sgl-router (#2464)
2025-07-18  c8f31042a8  [router] Refactor router and policy traits with dependency injection (#7
2025-08-05  8c7bb39dfb  [router] PD Router Simplification and Reorganization (#8838)
2025-08-18  6e316588f8  [router] add reasoning parser base structure (#9310)
2025-08-27  9768c50d90  [router] restructure tool parser module folder (#9693)
2025-09-11  53ca15529a  Implement Standalone gRPC Server for SGLang Python Scheduler (#10283)
2025-12-05  49dfa1d891  [model-gateway] change sgl-router to sgl-model-gateway (#14312)
```

[Chapter 13](../perf/multi-gpu.md) covered 0.1.0 of 11 December 2024 (dynamic scaling, fault tolerance) and the renaming to `sgl-router` the next day. For the next half year it was mostly making the router steadier: retries, health checks, metrics and the Python binding's release process. Its role began changing in July 2025: #7987's dependency-injection restructuring, the PD router's reorganisation in August (#8838), the reasoning-segment parsers (#9310, #9353), the tool parsers moving in (#9693), the protocol definitions gathered into `spec.rs` (#9519), gRPC routing in September (to go with [#10283](entrypoints.md)), and the renaming to `sgl-model-gateway` on 5 December. How the size changed:

```bash title="gateway-size.sh"
REF=${REF:-29f6d408c0}
for spec in "v0.4.0 rust" "v0.4.6 sgl-router" "v0.5.0rc0 sgl-router" "$REF sgl-model-gateway" "$REF rust"; do
  set -- $spec
  printf '%-11s %-18s %4d 个文件，.rs %4d 个，%6d 行 Rust\n' "$1" "$2" "$(git ls-tree -r --name-only "$1" -- "$2" | wc -l)" "$(git ls-tree -r --name-only "$1" -- "$2" | grep -c '\.rs$')" "$(git ls-tree -r --name-only "$1" -- "$2" | grep '\.rs$' | while read -r f; do git show "$1:$f"; done | wc -l)"
done
echo "gateway 的 tag：$(git tag | grep -c '^gateway-v')，最早 $(git for-each-ref --sort=creatordate --format='%(creatordate:short) %(refname:short)' refs/tags | grep gateway | head -1)，最新 $(git for-each-ref --sort=creatordate --format='%(creatordate:short) %(refname:short)' refs/tags | grep gateway | tail -1)"
```

```text title="output"
v0.4.0      rust                 17 个文件，.rs    5 个，  2169 行 Rust
v0.4.6      sgl-router           18 个文件，.rs    4 个，  2660 行 Rust
v0.5.0rc0   sgl-router           50 个文件，.rs   34 个， 18918 行 Rust
29f6d408c0  sgl-model-gateway   403 个文件，.rs  252 个， 94747 行 Rust
29f6d408c0  rust                186 个文件，.rs  160 个， 89973 行 Rust
gateway 的 tag：12，最早 2025-07-06 gateway-v0.1.5，最新 2026-01-08 gateway-v0.3.1
```

## Today's structure {#今天的结构}

```bash title="gateway-tree.sh"
REF=${REF:-29f6d408c0}
echo "sgl-model-gateway/src/ 一级：$(git ls-tree --name-only "$REF" sgl-model-gateway/src/ | sed 's|.*/||' | tr '\n' ' ')"
echo "policies/：$(git ls-tree -r --name-only "$REF" -- sgl-model-gateway/src/policies | sed 's|.*/||; s|\.rs||' | tr '\n' ' ')"
echo "routers/ 一级：$(git ls-tree --name-only "$REF" sgl-model-gateway/src/routers/ | sed 's|.*/||' | tr '\n' ' ')"
echo "rust/ 下的 crate：$(git ls-tree --name-only "$REF" rust/ | sed 's|rust/||' | grep -v '\.' | tr '\n' ' ')"
```

```text title="output"
sgl-model-gateway/src/ 一级：app_context.rs config core lib.rs main.rs middleware.rs observability policies routers server.rs service_discovery.rs version.rs wasm 
policies/：bucket cache_aware consistent_hashing factory manual mod power_of_two prefix_hash random registry round_robin tree utils 
routers/ 一级：conversations error.rs factory.rs grpc header_utils.rs http mcp_utils.rs mesh mod.rs openai parse persistence_utils.rs router_manager.rs streaming_utils.rs tokenize 
rust/ 下的 crate：sglang-grpc sglang-mm sglang-processor sglang-radix-tree sglang-renderer sglang-server 
```

`policies/` is the set of policies after #7987: round robin, random, cache-aware ([chapter 13](../perf/multi-gpu.md)'s approximate radix tree), power-of-two (pick two at random and take the less loaded) and the combinations for the PD case; `routers/` is divided by data plane: ordinary HTTP routing, PD routing, gRPC routing and OpenAI-compatible routing; `core/` is the worker abstraction and registration, `service_discovery.rs` connects to Kubernetes, `observability/` is metrics and tracing, `wasm/` is the plugin layer, and `bindings/` is the Python binding (keeping the `sglang_router` package name). The gateway's README opens:

```markdown title="sgl-model-gateway/README.md @ 29f6d408c0 L1-16" linenums="1"
# SGLang Model Gateway

High-performance model routing control and data plane for large-scale LLM deployments. The gateway orchestrates fleets of workers, balances traffic across HTTP and gRPC backends, and exposes OpenAI-compatible APIs with pluggable history storage and tool integrations—while remaining deeply optimized for the SGLang serving runtime.

## Overview
- Unified control plane for registering, monitoring, and orchestrating prefill, decode, and regular workers across heterogeneous model fleets.
- Data plane that routes requests across HTTP, PD (prefill/decode), gRPC, and OpenAI-compatible backends with shared reliability features.
- Industry-first gRPC pipeline with native Rust tokenization, reasoning, and tool-call execution for high-throughput OpenAI-compatible serving.
- Multi-model inference gateway mode (`--enable-igw`) that runs several routers at once and applies per-model policies.
- Conversation, response, and chat-history connectors that centralize state at the router, enabling compliant sharing across models/MCP loops with in-memory, no-op, or Oracle ATP storage options.
- Built-in reliability primitives: retries with exponential backoff, circuit breakers, token-bucket rate limiting, and queuing.
- First-class observability with structured logging, OpenTelemetry trace and Prometheus metrics.

### Architecture at a Glance
**Control Plane**
- Worker Manager validates workers, discovers capabilities, and keeps the registry in sync.
```

## The six crates under rust/ {#rust-里的六个-crate}

The `rust/` directory disappeared for over a year after the renaming to `sgl-router` and came back in 2026 holding something else: extensions the Python package compiles on demand:

```bash title="rust-crates.sh"
REF=${REF:-29f6d408c0}
for d in $(git ls-tree --name-only "$REF" rust/ | grep -v '\.'); do printf '  %-24s %3d 个文件\n' "${d#rust/}" "$(git ls-tree -r --name-only "$REF" -- "$d" | wc -l)"; done
git log --reverse --date=short --format='%ad  %h  %s' "$REF" -- python/sglang/srt/rust_extensions | head -1 | cut -c1-96
git log --reverse --date=short --format='%ad  %h  %s' "$REF" -- rust/sglang-radix-tree | head -1 | cut -c1-96
```

```text title="output"
  sglang-grpc               12 个文件
  sglang-mm                 23 个文件
  sglang-processor          12 个文件
  sglang-radix-tree         22 个文件
  sglang-renderer           43 个文件
  sglang-server             71 个文件
2026-08-16  67e12131df  Build Rust extensions on demand in source checkouts (#34994)
2026-09-01  a77283fb02  [Rust] Rename mem-cache to sglang-radix-tree (#37290)
```

`sglang-radix-tree` is the radix tree's Rust core (#32710 of 2026-08-31, "Add Rust TreeCore backend with shared parity tests": the Python tree and the Rust tree run the same tests to guarantee identical behaviour — the final destination of [chapter three](../origins/radix-v1.md)'s 220 lines of Python); `sglang-renderer` renders chat templates; `sglang-mm` does multimodal preprocessing; `sglang-processor` is the processing pipeline; `sglang-grpc` implements the gRPC protocol; and `sglang-server` is a service layer in Rust (`srt/rust_server/` is its Python entry point). `srt/rust_extensions/loader.py` compiles them on demand in a source install (#34994). So Rust has two identities in SGLang: a separately deployed gateway, and hot-path components embedded in the Python package.

![Figure: where the gateway sits in the system](../assets/figures/sgl-gateway-layers.svg){.aig-svg}

## Design trade-offs {#设计取舍}

| Decision | The reason | The price |
| --- | --- | --- |
| The gateway separate from the engine (its own directory and releases) | it can front several engines and several kinds of backend; a different release pace | a version matrix: compatibility testing of gateway version x engine version (`e2e_test/`) |
| The policy separated from the router (DI) | policies become composable and testable, and the PD router reuses them | one more layer of abstraction |
| Control-plane work moved up (tokenizing, templates, parsing) | relief for the Python side, high concurrency | two sets of parsers (Python's `function_call/` and Rust's) to keep in step |
| Rust crates embedded in the Python package | the hot paths (the tree, the preprocessing) in Rust with the interface unchanged | a source install has to compile; two implementations need parity tests |

## What happened afterwards {#后来怎么样了}

- 2025-07 → 2026-01: twelve separate releases from `gateway-v0.1.5` to `gateway-v0.3.1` (it released separately even before the renaming); the later WASM plugins, Anthropic protocol and multimodal request handling ship with the main repository.
- `sglang-radix-tree`'s Rust tree became an optional backend for the engine's radix tree, used by `unified_cache/`.
- Kubernetes service discovery works with elastic EP ([chapter 18](../scale/large-ep.md)): an instance can join and leave on the fly.
- The inference-systems handbook's [vLLM Rust frontend chapter](serving://source/rust-frontend/) covers vLLM's equivalent, and the two divide the work along similar lines.

## Exercises {#练习}

**1. The policy interface.** Read the trait definition in the baseline commit's `sgl-model-gateway/src/policies/mod.rs`, list the methods a policy has to implement, and explain why a cache-aware policy also needs a "request finished" callback.

??? success "A way to approach it"
    There is at least `select_worker` (pick a worker for a request) and a hook for load updates; a cache-aware policy has to update the approximate tree and the load counts when a request finishes.

**2. PD routing's extra work.** Find the PD router under `routers/` and say in which function each of its three jobs for a request happens (pick a prefill, pick a decode, inject the bootstrap information).

??? success "A way to approach it"
    `grep -rn 'bootstrap' sgl-model-gateway/src/routers/` locates the injection; picking the instances calls the policy twice.

**3. The parity tests.** Read #32710's "shared parity tests": how do the Python tree and the Rust tree share one set of tests? Find where the test file is.

??? success "A way to approach it"
    `git show 9cf157c252 --stat` shows the parameterised tests added under `test/`, with each case run once against each backend.

!!! interview "How to explain it"
    "What should an inference gateway do?" — Use sgl-model-gateway's evolution: from routing (round robin → cache-aware → PD-aware) to protocol conversion, parsing, service discovery and observability; the policy separated from the router; and the control plane moving up from Python into Rust, talking to the scheduler directly over gRPC. Then mention Rust's second identity (hot-path crates embedded in the Python package) and the parity tests it demands.

## Summary {#小结}

- [x] `rust/` (2024-10) → `sgl-router` (2024-12) → `sgl-model-gateway` (2025-12): from a router to a model gateway, 95 thousand lines of Rust with its own releases.
- [x] #7987 abstracted the policies into a trait and left the router to forward; PD routing, gRPC routing, the parsers, service discovery and WASM plugins joined one after another.
- [x] `rust/` reappeared in 2026 with six crates compiled on demand (the radix tree's core, template rendering, multimodal, gRPC, the service layer): Rust is both the gateway and the engine's hot path.
