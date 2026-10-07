# Self-test questions

<p class="lead">Use these questions to find out which math you need to fill in: skip the ones you can answer; for the ones you cannot, open the reference answer first, then go and read the linked section. Every question comes from a real problem in inference work, and every number comes from the measurements on a real model in the chapters.</p>

!!! tip "How to use these questions"
    - First pass: 2 minutes per question, and only mark "can answer / unsure / don't know";
    - Second pass: only the "unsure" and "don't know" questions; read the section linked in the answer, then answer again **without looking at the answer**;
    - Questions on model architecture, training, how inference works, and questions spanning chapters are in [the LLM Internals self-test](llm://synthesis/quiz/).

## Linear algebra {#线性代数}

**1. Are pretrained weights low rank? What about the KV cache? What does that mean for compression?**

??? success "Reference answer"
    Look at how the energy of the singular values is distributed with an SVD. Keeping 128 directions of Qwen3-0.6B's `gate_proj` retains only 36.1% of the energy (25.7% for a random matrix): **the weights are not low rank**, so you cannot simply factor the whole model into low-rank pieces. K and V clearly are low rank: about 90–130 of 1024 directions cover 90% of the energy. That is exactly what MLA and low-rank KV compression rely on. See [rank and low-rank approximation](linear-algebra.md#秩与低秩近似).

**2. Why do rotation methods such as QuaRot improve W4A4 quantization? Why does the rotation not change the model's output?**

??? success "Reference answer"
    An orthogonal matrix satisfies $QQ^\top = I$, so $xW = (xQ)(Q^\top W)$: mathematically equivalent to the original computation, and $Q^\top$ can be multiplied into the weights ahead of time. The rotation "spreads" outliers concentrated in a few channels across all channels: measured, the max / median ratio drops from 47.2 to 9.0, and the relative error of W4A4 from 0.421 to 0.148 (with a random orthogonal rotation). Rotating with a Hadamard matrix lets the multiplication run as a fast transform in $O(d \log d)$. See [orthogonal matrices and rotations](linear-algebra.md#正交矩阵与旋转).

## Probability and information theory {#概率与信息论}

**3. In speculative decoding, what is the probability that a draft token is accepted?**

??? success "Reference answer"
    Accepting with probability $\min(1, p/q)$ gives an acceptance rate of $\sum_x \min(p(x), q(x)) = 1 - \text{TV}(p, q)$, one minus the total variation distance between the two distributions; on rejection you resample from $\max(0, p - q)$ renormalized, and the final output distribution is exactly the target distribution. With the INT4 version as the draft, the measured mean acceptance rate is 0.70, exactly 1 − TV. See [rejection sampling and speculative decoding](probability.md#拒绝采样与投机解码).

**4. When evaluating a quantized model, why look at KL divergence and top-1 agreement in addition to perplexity?**

??? success "Reference answer"
    Perplexity only looks at the probability of the correct answer: two models with similar perplexity do not necessarily produce similar distributions at every position. KL divergence measures directly how far the quantized model's distribution moves from the original, and top-1 agreement shows how much the greedy output will differ. Measured, INT8 has a KL of only 0.005 and 96% top-1 agreement; group-wise INT4 has a KL of 0.46 and only 64% agreement. KL is also directly tied to the acceptance rate of speculative decoding. See [KL divergence: how much quantization changes](information-theory.md#kl-散度量化改变了多少).

## Calculus and backpropagation {#微积分与反向传播}

**5. Why is GPTQ better than rounding each weight to nearest (RTN)?**

??? success "Reference answer"
    The per-layer goal is to keep the quantized output $XW$ unchanged, and the second-order approximation of the error is governed by $H = X^\top X$. RTN amounts to assuming $H$ is diagonal and ignores the correlation between input dimensions; GPTQ quantizes column by column and uses $H^{-1}$ to push each column's error onto the columns not yet quantized. Measured with per-channel INT4: the original model's perplexity is 25.94, RTN 55.02, GPTQ 36.85. See [second-order information: GPTQ](calculus.md#二阶信息gptq).

## Floating point and numerical computing {#浮点与数值计算}

**6. Why do Tensor Cores multiply in BF16/FP8 but accumulate in FP32?**

??? success "Reference answer"
    Floating-point addition rounds, and the larger the sum, the larger the gap between neighboring representable numbers. BF16 has only 8 significant bits: once the sum reaches 256 the gap is 2, so adding a number smaller than 1 leaves it unchanged. Measured, accumulating 20,000 numbers one by one in BF16 gives 256, while the right answer is about 10034; with an FP32 accumulator there is almost no error. See [accumulation: errors add up](floating-point.md#累加误差会积累).

## The math of performance and serving {#性能与服务中的数学}

**7. If a service's utilization rises from 90% to 95%, what happens to mean latency? Why must a service not be planned near full load?**

??? success "Reference answer"
    In the M/M/1 queueing model, the mean time in the system is $\frac{1}{\mu - \lambda} = \frac{1}{\mu}\cdot\frac{1}{1-\rho}$; as utilization ρ goes from 0.9 to 0.95, $\frac{1}{1-\rho}$ goes from 10 to 20, so mean latency doubles and tail latency grows even more. Inference services can batch, but the pattern is the same: past a certain request rate, TTFT suddenly explodes. So capacity planning leaves headroom and keeps utilization in the flat part of the latency curve. Also, Little's law $L = \lambda W$ lets you work out latency from concurrency and throughput. See [queueing theory](performance-math.md#排队论为什么接近满载时延迟爆炸).

**8. A load test of 200 requests gives a P99 of 233 ms. Can you trust that number?**

??? success "Reference answer"
    Only 2 of 200 samples lie above the P99, so the estimate has a large variance. Estimate a confidence interval by bootstrap resampling: with 200 samples the 95% confidence interval is [204, 313] ms, and it narrows to [310, 341] ms only at 5000 samples. Report tail latency with enough samples, preferably with a confidence interval. See [the statistics of performance measurement](performance-math.md#性能测量的统计学).
