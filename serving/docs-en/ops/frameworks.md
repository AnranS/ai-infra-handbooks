# Choosing an inference framework: vLLM, SGLang, TensorRT-LLM, LMDeploy and on-device frameworks

<p class="lead">This book follows vLLM and SGLang, but in practice you are often asked "why choose it and not another". This chapter compares several mainstream frameworks by the questions that really matter when choosing: each one's design focus, the scenarios it excels at, and the costs it brings. Frameworks iterate quickly, so only their relatively stable positioning and ways to judge them are written here; for specific features and performance, always go by measurements on your own model, hardware and workload.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. How do vLLM's and SGLang's design focuses differ?
    2. What scenarios suit TensorRT-LLM? What does it cost?
    3. What three things should you confirm first when choosing?
    4. How do you run a fair comparison of frameworks?

??? success "Answers (try first, then expand to compare)"
    1. vLLM is general-purpose, with the broadest ecosystem and the widest hardware coverage (many kinds of GPUs and accelerators), valuing stability and compatibility; SGLang leans toward performance and cutting-edge features: prefix caching (RadixAttention), large-scale EP, structured output, and RL-related capabilities.
    2. Scenarios only on NVIDIA GPUs, chasing peak performance, with relatively fixed models. The costs: models must first be compiled into engines, so changing models or making modifications takes long cycles, new models are supported more slowly, and debugging and customization are harder.
    3. Model support (including quantization formats), hardware support, and the key features your scenario needs (such as multi-LoRA, structured output, PD disaggregation, long context).
    4. The same model, precision and request distribution; tune each framework's parameters per its official advice (parallelism, batch limits, chunk size, CUDA Graph buckets) rather than using defaults; compare goodput curves under the SLO rather than peak throughput; sample-check output quality to make sure nothing is "faster but worse".

## The positioning of several mainstream frameworks {#几个主流框架的定位}

| Framework | Design focus | Excels at | Watch out for |
| --- | --- | --- | --- |
| **vLLM** | general-purpose with the broadest ecosystem: the origin of paged KV; the V1 engine layers scheduling, KV management and execution; pluggable hardware backends | broad model coverage with fast support for new models; many kinds of hardware (NVIDIA, AMD, TPU and more); the KV connector interface makes PD disaggregation and KV offloading pluggable | many features and many options, needing tuning for the workload; some cutting-edge features may land later than in frameworks specialized for them |
| **SGLang** | performance and cutting-edge features: RadixAttention prefix caching, zero-overhead overlap scheduling, structured output, large-scale EP | multi-turn chat and agents (lots of prefix reuse), large-scale deployment of DeepSeek-style MoE (DP Attention, DeepEP, EPLB), RL rollout (used by verl, slime and others) | fast iteration with sizable changes between versions; support for some hardware less comprehensive than vLLM's |
| **TensorRT-LLM** | NVIDIA's official framework: highly optimized kernels and engines, with the earliest support for new hardware features (FP8, FP4, new attention kernels) | chasing peak performance on NVIDIA GPUs; integration with Triton Inference Server and Dynamo | NVIDIA only; the traditional "compile an engine" workflow is heavy (newer versions also offer a PyTorch workflow); customizing new models is costly |
| **LMDeploy** | the C++ TurboMind engine plus a PyTorch engine; a complete quantization toolchain | W4A16 (AWQ), INT8 / INT4 quantization of the KV Cache; adaptation of Chinese models | smaller community and ecosystem than the first two |
| **llama.cpp / Ollama, MLX, ExecuTorch** | on-device and local: CPUs, Apple Silicon, phones | single-user, offline and privacy scenarios (see [on-device inference](edge.md)) | unsuited to high-concurrency servers |

There are also many components built around these engines: NVIDIA Dynamo, llm-d and Mooncake handle scheduling and KV management for disaggregated architectures, LMCache provides a KV cache layer, and Triton Inference Server and KServe are general model-serving frameworks. They usually don't replace the engine, but combine on top of it.

## Questions to ask when choosing {#选型时要问的问题}

In order of priority:

1. **Can the model and hardware run**: is the target model (including newly released ones) supported promptly? Does your hardware (GPU model, AMD, domestic accelerators) have a well-maintained backend? If this fails, nothing else matters (how frameworks plug into different hardware is in [multi-hardware support](platforms.md));
2. **Are the key features there**: of the features your scenario needs (prefix caching, chunked prefill, PD disaggregation, large-scale EP, multi-LoRA, structured output, speculative decoding, quantization formats, long context), which are mature and usable, and which are merely "supported" but not yet stable;
3. **Performance on your workload**: load test with your real request distribution (input and output lengths, concurrency, prefix-sharing ratio), comparing the maximum throughput while meeting the SLO (goodput), not the vendor's benchmark numbers;
4. **Operability**: metrics and tracing, Kubernetes integration, how smooth upgrades are, how hard problems are to troubleshoot;
5. **Customizability and community**: does the team need to change code? Is the code easy to read and modify (mostly Python, or lots of C++)? Community activity, issue response speed, release cadence.

What many teams actually do is **primary + backup**: one framework as the mainstay, the other as a supplement for specific scenarios (say RL rollout, or a certain class of models), keeping the ability to switch between them (a unified OpenAI-compatible interface, a unified gateway).

## How to run a fair comparison {#怎样做一次公平的对比}

- **The same model files and precision**: the same checkpoint, the same quantization;
- **The same requests**: a dataset generated with a fixed random seed or a replay of real traffic, with the same input and output length distributions; control output length with `ignore_eos` or a fixed maximum, avoiding differences from early stops;
- **Tune each one**: comparing default configurations is unfair; tune each framework's parallelism, batch limits, chunk size and CUDA Graph buckets per its official advice;
- **Compare goodput**: requests completed per second under the TTFT and TPOT SLOs; also report P50 / P99 latency curves (how they change as load rises), not single points;
- **Check the outputs**: sample and compare output quality to make sure no optimization (such as aggressive quantization or wrong sampling parameters) has made things "faster but worse".

Load-testing methods are in [load testing, SLOs and capacity planning](../perf/benchmark.md), and locating gaps in [profiling inference engines](../perf/profiling.md).

!!! interview "In an interview"
    For "why vLLM / SGLang", don't just say "good performance, big community". Answer in the order of selection: model and hardware support → the key features the scenario needs → measured goodput on your own workload → operability → customizability. Then give a concrete trade-off, such as "multi-turn chat has lots of prefix reuse, and SGLang's RadixAttention hits more often; but on our AMD cluster vLLM's backend is more mature, so...". Being able to explain "how to run a fair comparison" is more convincing than quoting any benchmark number.

## Exercises {#练习}

**1. A vendor's benchmark says framework A is 2× faster than B. How would you verify it?**

??? success "Approach"
    First look at the benchmark's conditions: model, precision, hardware, request length distribution, concurrency, whether speculative decoding or special quantization was on, whether it compares throughput or latency, and whether framework B used its recommended configuration. Then reproduce on your own workload: a fixed request set, each tuned, plotting goodput curves as load varies. A common outcome is that it really is 2× faster at one particular load point, but the gap is small in the range you care about, or the other way around.

## Summary {#小结}

- [x] vLLM is general-purpose with a broad ecosystem and full hardware coverage; SGLang leans toward performance and cutting-edge features (prefix caching, large-scale EP, RL); TensorRT-LLM chases peak performance on NVIDIA; LMDeploy has a complete quantization toolchain; on-device has its own dedicated frameworks.
- [x] Order of selection: model and hardware support → key features → goodput on your own workload → operability → customizability and community.
- [x] Fair comparison: the same model and requests, each tuned, comparing goodput curves under the SLO and checking output quality.
