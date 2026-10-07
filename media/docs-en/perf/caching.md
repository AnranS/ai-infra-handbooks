# Feature caching: two neighbouring steps look too much alike

<p class="lead">Over the dozens of denoising steps, two neighbouring steps' inputs differ by a little noise and the network's outputs by a little too, so why compute each one from scratch? Feature caching reuses the previous step's result: DeepCache reuses a UNet's deep features, TeaCache and FBCache use a cheap signal to decide whether a step can be skipped entirely, and PAB updates different attentions at different rates. None changes the model or retrains it, and video models see 1.5 to 2.5 times across the board. This chapter measures how alike neighbouring steps are with a small DiT, implements a TeaCache-style skipper, and sees how much it saves, how much it gets wrong, and when it fails.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What does feature caching presuppose? What quantity verifies that presupposition?
    2. What do DeepCache and TeaCache each cache? Which suits a UNet and which a DiT?
    3. How does TeaCache decide a step can be skipped without computing it?
    4. What error does caching introduce? How does it accumulate? How is the threshold set?
    5. Why can a few-step distilled model barely use caching? Do guidance's two paths need separate caches?

??? success "Answers for the self-test (answer first, then open this)"
    1. That neighbouring timesteps' network outputs are highly similar (the denoising trajectory is smooth). It can be measured directly: each step's relative L1 distance from the previous step's output is usually a few percent over the long middle stretch.
    2. DeepCache caches a UNet's deep (low-resolution) blocks' features and recomputes only the shallow (high-resolution) branch every few steps, exploiting the UNet's skip connections, so it suits a UNet alone. TeaCache caches the whole denoising network's output residual and, when it skips, adds the previous step's residual to get this step's output, which depends on no architecture and works for both DiTs and UNets.
    3. It computes only the very front of the network: the input after the timestep embedding's AdaLN modulation (or the first block's output), takes its relative L1 distance from the same quantity at the previous step, accumulates that distance, and really computes a step and resets when it exceeds a threshold, skipping otherwise. That front piece is a few percent of a step's computation.
    4. A skipped step uses an old residual, which differs from the true output, and the difference propagates through the steps that follow; the more consecutive skips, the larger the accumulation. So the rule is "recompute when the accumulated distance exceeds a threshold" rather than "recompute every k steps", so that fast-changing stretches compute more and slow-changing ones skip more. The threshold is swept against a quality metric (PSNR, or the eye) versus the speedup, and differs per model.
    5. A few-step model's every step changes the latents greatly (the whole path in 4 steps), so neighbouring steps are not alike and there is nothing to reuse. Guidance's two paths have different inputs (different conditions), so each needs its own cache, or the caching is done only on the unconditional path.

A six-panel strip first, then the chapter:

<!-- comic ../assets/comics/feature-cache.webp is in Chinese; put it back once the English version exists -->

## The presupposition: how alike two neighbouring steps are {#前提相邻两步有多像}

Running a denoising loop with a small DiT and recording three quantities at each step: the modulated input (the tensor entering the first block after AdaLN), the first block's output residual, and the whole network's output. Here is their relative change between neighbouring steps:

```python
import torch
from diffusers import DiTTransformer2DModel, DDIMScheduler

torch.manual_seed(0)
dit = DiTTransformer2DModel(num_attention_heads=4, attention_head_dim=16, in_channels=4, out_channels=8,
                            num_layers=4, sample_size=16, patch_size=2, num_embeds_ada_norm=1000).eval()
sched = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear", clip_sample=False)
sched.set_timesteps(20)

captured = {}
def grab(name):
    def hook(m, i, o):
        captured[name] = (o[0] if isinstance(o, tuple) else o).detach()
    return hook
dit.transformer_blocks[0].register_forward_hook(grab("block0_out"))

def rel_l1(a, b):
    return ((a - b).abs().mean() / b.abs().mean()).item()

x = torch.randn(1, 4, 16, 16)
cls = torch.tensor([3])
prev = {}
rows = []
with torch.no_grad():
    for i, t in enumerate(sched.timesteps):
        out = dit(x, timestep=t[None], class_labels=cls).sample[:, :4]
        cur = {"输出": out, "第一块输出": captured["block0_out"]}
        if prev:
            rows.append((i, int(t), rel_l1(cur["第一块输出"], prev["第一块输出"]), rel_l1(cur["输出"], prev["输出"])))
        prev = cur
        x = sched.step(out, t, x).prev_sample

print(f"{'步':>3} {'t':>4} {'第一块输出的变化':>14} {'整网输出的变化':>13}")
for i, t, d1, d2 in rows:
    print(f"{i:>3} {t:>4} {d1:>14.3f} {d2:>13.3f}")
```

```text title="output"
  步    t       第一块输出的变化       整网输出的变化
  1  900          0.422         0.280
  2  850          0.499         0.237
  3  800          0.511         0.150
  4  750          0.489         0.102
  5  700          0.454         0.081
  6  650          0.416         0.074
  7  600          0.378         0.063
  8  550          0.342         0.054
  9  500          0.308         0.051
 10  450          0.274         0.046
 11  400          0.242         0.053
 12  350          0.211         0.058
 13  300          0.181         0.062
 14  250          0.151         0.059
 15  200          0.123         0.039
 16  150          0.095         0.061
 17  100          0.068         0.052
 18   50          0.041         0.063
 19    0          0.016         0.072
```

This is a small model with random weights, so the numbers mean nothing in themselves, but two things match a real model: **the change between neighbouring steps is continuous and smooth** (no jumps), and **the front of the network's change tracks the whole network's output change**. The latter is TeaCache's entire basis: predict the expensive whole-network change from the cheap front-end one (a real implementation also fits a polynomial to map one to the other).

A real model's curve (FLUX, Wan, HunyuanVideo) has this shape: large changes over the first few steps (the composition is settling), a few percent over a long middle stretch, and a slight rise again at the end (the detail is settling). So what can be skipped is that long middle.

Using the change curve measured above, dial the threshold and see which steps get skipped:

<div class="aig-widget" data-widget="cache-skip"></div>

## Implementing a TeaCache-style skipper {#实现一个-teacache-风格的跳步器}

The rule: at each step, first compute the cheap probe (the first block's output here), take its relative L1 distance from the probe at the last genuinely computed step and accumulate it; while the accumulation is below the threshold, skip, taking the **residual** (output minus input) from the last genuine computation and adding it to the current input as the output; when it exceeds the threshold, really compute a step and reset the accumulation.

```python
def denoise(threshold, probe_blocks=1):
    """threshold=0 等于每步都算；越大跳得越多。返回最终潜变量、真算的次数。"""
    torch.manual_seed(0)
    x = torch.randn(1, 4, 16, 16)
    acc, last_probe, last_resid, computed = 0.0, None, None, 0
    with torch.no_grad():
        for t in sched.timesteps:
            # the probe: run only as far as the first block (here a hook takes its output; a real implementation runs the first few layers separately)
            full = dit(x, timestep=t[None], class_labels=cls).sample[:, :4]
            probe = captured["block0_out"]
            skip = False
            if last_probe is not None:
                acc += rel_l1(probe, last_probe)
                skip = acc < threshold
            if skip:
                out = x + last_resid                           # reuse the residual from the last genuine computation
            else:
                out = full
                computed += 1
                acc, last_probe, last_resid = 0.0, probe, full - x
            x = sched.step(out, t, x).prev_sample
    return x, computed

ref, n_ref = denoise(0.0)
print(f"{'阈值':>5} {'真算步数':>7} {'加速比':>6} {'最终潜变量相对误差':>12}")
for th in (0.0, 0.05, 0.1, 0.2, 0.4):
    x, n = denoise(th)
    err = ((x - ref).norm() / ref.norm()).item()
    print(f"{th:>5.2f} {n:>7} {len(sched.timesteps) / n:>6.2f}× {err:>12.3f}")
```

```text title="output"
   阈值    真算步数    加速比    最终潜变量相对误差
 0.00      20   1.00×        0.000
 0.05      19   1.05×        0.007
 0.10      17   1.18×        0.021
 0.20      16   1.25×        0.068
 0.40      12   1.67×        0.218
```

(This implementation still runs the whole network to get the probe, so it saves no real time; it only measures the skipping's logic and error. A real TeaCache runs the first few layers separately, and the probe costs a few percent of a step.)

How to read the table: the larger the threshold, the fewer the genuinely computed steps and the larger the error, so **the speedup and the error trace a curve** and deployment picks a point on it against a quality metric. The empirical values on real models:

| Method | Model | Typical speedup | Quality cost | Notes |
| --- | --- | --- | --- | --- |
| DeepCache | SD 1.5 / SDXL (UNet) | 2 to 3x | little visible PSNR drop | the deep features recompute every N steps and the shallow ones every step |
| TeaCache | FLUX, HunyuanVideo, Wan, CogVideoX | 1.5 to 2.5x | indistinguishable at a threshold of 0.1 to 0.2, blurring above 0.3 | the probe is the input after the timestep embedding's modulation, with a fitted polynomial correction |
| FBCache (first-block cache) | FLUX, Wan and other DiTs | 1.5 to 2x | the same | the probe is the first block's output residual, with no fitted coefficients |
| PAB (pyramid attention broadcast) | video DiTs | 1.3 to 1.5x | slight | spatial, temporal and cross attention update at different rates, stackable with sequence parallelism |
| ToCa / per-token caching | DiTs | 1.5 to 2x | slight | only the tokens that changed a lot are recomputed |
| FORA | DiTs | about 2x | slight | caches the attention and MLP outputs every N steps |

What they have in common: **no weight changes, no retraining, and they take effect simply by being turned on at inference**, so they stack with quantization, parallelism and compilation. That is the fundamental reason they are popular in deployment.

## How the error accumulates and how the threshold is set {#误差怎么累积阈值怎么定}

A skipped step uses an old residual, so its output differs from the true value; the scheduler carries that difference into the next step's input, affecting the later probes and outputs. Here is the error's growth under consecutive skips:

```python
def denoise_fixed(skip_pattern):
    """按固定模式跳步：skip_pattern[i] 为 True 表示第 i 步复用残差。返回最终潜变量。"""
    torch.manual_seed(0)
    x = torch.randn(1, 4, 16, 16)
    last_resid = None
    with torch.no_grad():
        for i, t in enumerate(sched.timesteps):
            if skip_pattern[i] and last_resid is not None:
                out = x + last_resid
            else:
                out = dit(x, timestep=t[None], class_labels=cls).sample[:, :4]
                last_resid = out - x
            x = sched.step(out, t, x).prev_sample
    return x

n = len(sched.timesteps)
patterns = {
    "每隔一步跳一步（跳 10 步）": [i % 2 == 1 for i in range(n)],
    "中间连跳 10 步":            [5 <= i < 15 for i in range(n)],
    "开头连跳 10 步":            [1 <= i < 11 for i in range(n)],
    "结尾连跳 10 步":            [i >= 10 for i in range(n)],
}
for name, pat in patterns.items():
    err = ((denoise_fixed(pat) - ref).norm() / ref.norm()).item()
    print(f"{name:<22} 跳 {sum(pat):>2} 步  相对误差 {err:.3f}")
```

```text title="output"
每隔一步跳一步（跳 10 步）        跳 10 步  相对误差 0.440
中间连跳 10 步              跳 10 步  相对误差 0.755
开头连跳 10 步              跳 10 步  相对误差 0.878
结尾连跳 10 步              跳 10 步  相对误差 0.506
```

For the same 10 skipped steps, **spreading the skips out is far better than skipping consecutively**, and **skipping consecutively at the start is the worst** (an error in the composition stage is amplified by every step that follows). That is why every good caching policy recomputes when the accumulated change exceeds a threshold rather than every k steps: fast-changing stretches automatically compute more and slow-changing ones automatically skip more. It is also why every implementation forces the first and last few steps never to skip.

How to set the threshold: fix a set of prompts and seeds, sweep the threshold, plot the speedup against quality (PSNR or SSIM against the uncached result, or a human score), and take the point just before the quality starts dropping visibly. The curve differs for each model, resolution and step count, so **a threshold cannot be copied across models**.

## When it fails {#什么时候失效}

- **Few-step models**: a 4-step model changes the latents greatly at every step, so neighbouring steps are not alike and there is nothing to reuse. Caching is for complete models at 20 steps and above.
- **Guidance's two paths**: the conditional and unconditional inputs differ and so do their residuals, so each needs its own cache; one cache mixes the two up.
- **Requests out of sync within a batch**: if different requests in one batch are at different timesteps, or some skip while others do not, skipping the whole network is impossible. Caching naturally prefers many steps of one request, which is one of the things to consider when scheduling a generation service (see [Scheduling a generation service](../serving/scheduling.md)).
- **A changed resolution or step count**: the threshold has to be swept again.
- **Stacked with sequence parallelism**: the probe's distance has to be computed consistently across cards (all-reduce a scalar), or the cards' skip decisions diverge and the communication deadlocks.

!!! interview "How to answer in an interview"
    Asked what caching acceleration is for a diffusion model, give the presupposition and the evidence: the relative change in neighbouring steps' outputs is a few percent over the middle stretch, and it can be measured directly. Then the two families: DeepCache reuses a UNet's deep features through its skip connections; TeaCache and FBCache use the first few layers' change as a probe, really computing a step only when the accumulation exceeds a threshold and otherwise reusing the previous residual. The probe is a few percent of a step and the whole thing gives 1.5 to 2.5 times, with no weight changes and no retraining, stackable with quantization and parallelism. Finish with the boundaries: the error accumulates along the steps, so spread the skips out and never skip the first or last; a few-step model has nothing to cache; guidance's two paths need separate caches; and across cards the skip decisions have to be synchronised.

## Exercises {#练习}

1. Change the probe from the first block's output to the modulated input (registering a forward pre-hook on `transformer_blocks[0]` to take its input) and repeat the first experiment. Which probe tracks the whole network's output change better? Why does a real TeaCache still apply a polynomial correction to the probe's distance?

??? success "Answer"
    The modulated input has only been scaled and shifted by the timestep embedding, so its change mostly reflects the timestep's own change, with the same trend as the whole network's output but a different proportion; the first block's output has been through an attention and an MLP and is closer to the whole network. TeaCache fits a polynomial mapping the input's distance to an estimate of the output's, which is correcting exactly that proportion. The coefficients differ per model, so each one has to be fitted separately.

2. Add a rule to `denoise` forcing the first 3 and last 2 steps to be computed. How do the computed step count and the error change at a threshold of 0.2?

??? success "Answer"
    The computed count rises slightly (at most 5 more steps) and the error falls noticeably, because an error at the start is amplified and the end decides the detail. Nearly every implementation carries this rule, and it is the best value of any.

3. A video service has sequence parallelism (4 cards) and TeaCache on together. One inference hangs, and caching is suspected. What could be wrong? How would you verify it?

??? success "Answer"
    The cards compute slightly different probe distances (each holding different tokens), so one card decides to skip while another decides to compute, and the computing card waits in attention for an all-to-all or all-gather that the skipping card never issues, which deadlocks. To verify: print each rank's `acc` and `skip` at the decision point and see whether they agree. To fix: all-reduce the distance (averaging or taking the maximum) so that every rank uses the same decision.

## Summary {#小结}

- [x] Caching presupposes that neighbouring steps' outputs are highly similar, which can be measured directly: a few percent of relative change over the middle stretch, with the first few layers' change predicting the whole network's.
- [x] DeepCache reuses a UNet's deep features; TeaCache and FBCache accumulate a probe's distance and really compute only past a threshold, otherwise reusing the previous residual. They change no weights, need no retraining, give 1.5 to 2.5 times, and stack with quantization and parallelism.
- [x] The error accumulates along the steps: spreading the skips out beats skipping consecutively, and skipping consecutively at the start is the worst; use the accumulated change rather than a fixed interval, and force the first and last steps never to skip; the threshold has to be swept against a quality curve per model.
- [x] Where it fails: few-step models, guidance's two paths needing separate caches, requests out of sync within a batch, and skip decisions needing synchronisation across cards.
