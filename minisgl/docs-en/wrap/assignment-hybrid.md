# Project: adding the hybrid model Qwen3.5

<p class="lead">The mini-sglang of this book only knows models where every layer is attention: a request's whole history lives in the KV pool, and prefix caching, chunked prefill and request-slot reuse all rest on that. Qwen3.5 breaks it, with 18 of its 24 layers being Gated DeltaNet, so each request carries a fixed-size state that is updated in place. This project asks you to bring it in: output matching Hugging Face token by token, and a clear account of which mechanisms in the engine have to be rebuilt because of it.</p>

The instructions and the check scripts are in the repository at [`assignments/a4-hybrid-model/`](https://github.com/AnranS/ai-infra-handbooks/tree/main/assignments/a4-hybrid-model), and only a CPU is needed. Before starting, read the Inference Systems handbook's chapters on [bringing up a new model and matching its numerics](serving://ops/new-model/) and on [linear attention and hybrid architectures](serving://frontier/linear-attn/).

## What to do {#要做什么}

| Part | Requirement |
| --- | --- |
| Config | `ModelConfig` reads out each layer's type and the linear-attention layers' shapes; RoPE rotates only the first quarter of each head's dimensions |
| Weights | take only the language part from a multimodal checkpoint, skipping the vision encoder and the MTP layers, with the keys matching the model exactly |
| Compute | gated full-attention layers and Gated DeltaNet layers, matching the reference implementation in both prefill and decode |
| State | one convolution cache and one recurrent state per request, kept by request slot; a new request starts from zero, and chunked prefill passes the state from chunk to chunk |
| Memory | the KV pool is allocated for the 6 full-attention layers only, and the state pool counts towards the memory plan |
| Prefix cache | the radix cache either refuses outright or hits only at positions where a state checkpoint was stored |

Four checks: config parsing, weight loading, greedy decoding matching Hugging Face token by token (a single request, variable-length batching, chunked prefill, request-slot reuse), and the prefix cache's handling.

## Why it is worth doing {#为什么值得做}

- **This is what bringing up a new model really costs**: read the config and the reference implementation, find every place it differs from the models you already have (two kinds of RMSNorm, a gate wedged inside q_proj, RoPE over only some of the dimensions), and match it layer by layer;
- **It forces you to think about state**: KV can be shared by block and truncated by position, while state can only be saved and copied whole. The state must be cleared when a request slot is reused, carried across chunks during chunked prefill, and both prefix caching and speculative decoding have to be redesigned around it. This is exactly what vLLM and SGLang went through supporting hybrid models like Qwen3-Next, Qwen3.5 and Kimi;
- **The numbers in the report go straight into an interview**: how large each request's state is, why a hybrid model uses *more* memory on short requests, and why decode's cost barely grows with context.

## What to deliver {#交付物}

Code that passes every check, a one-page report (the memory ledger, how decode's cost changes with context, prefill time for the chunked algorithm against token-by-token recurrence), and a paragraph on what your engine would still be missing to serve this model on a GPU.
