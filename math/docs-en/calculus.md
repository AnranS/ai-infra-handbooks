# Calculus and backpropagation

<p class="lead">Inference itself needs no derivatives, but people who work on inference cannot avoid them: training and fine-tuning (LoRA, RL) need backpropagation; "training costs about 3 times as much compute as inference" comes from the cost of backpropagation; quantization methods such as GPTQ use the loss's second-order information. This chapter starts from derivatives and the chain rule, writes a minimal automatic differentiation engine and checks it against PyTorch, derives the gradients of a linear layer and of softmax cross-entropy, and finally implements a simplified GPTQ with the Hessian that wins back a large part of INT4 quantization's accuracy on a real model.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. How does reverse-mode automatic differentiation work? Why does it suit the case of "many parameters, one loss"?
    2. In the backward pass of $Y = XW^\top$, what are $\partial L/\partial X$ and $\partial L/\partial W$? Why does training cost about $6N$ per token?
    3. Why is the gradient of softmax plus cross-entropy $p - y$?
    4. When checking gradients with finite differences, why must the step $h$ not be too small?
    5. What second-order information does GPTQ use? Why is it better than rounding each weight to nearest?

??? success "Answers (try first, then expand to compare)"
    1. The forward pass records the computation graph; the backward pass starts from the loss and, by the chain rule, passes "the derivative of the loss with respect to each intermediate" (the adjoint) all the way back. One backward pass gives the gradient of the loss with respect to every parameter at about twice the cost of the forward pass, independent of the number of parameters, so it suits the case of "many parameters, a single loss as output".
    2. $\partial L / \partial X = (\partial L / \partial Y)\,W$ and $\partial L / \partial W = (\partial L / \partial Y)^\top X$, each a matrix multiplication as large as the forward one. Forward $2N$ plus two backward matrix multiplications, $4N$, comes to about $6N$ per token.
    3. Cross-entropy is $L = -\log p_y$; substituting softmax and differentiating gives $\partial L / \partial z_i = p_i - \mathbb{1}[i = y]$, that is $p - y$ ($y$ is one-hot).
    4. Finite-difference error has two parts: truncation error shrinks with the step, while rounding error is about machine precision ÷ step and grows as the step shrinks. With too small a step, $f(x+h) - f(x-h)$ suffers catastrophic cancellation, so use float64 and a moderate step.
    5. GPTQ uses the second-order information of the layer input, $H = X^\top X$ (the Hessian of the reconstruction error with respect to the weights). After quantizing one weight, it spreads that weight's error over the not-yet-quantized weights of the same row according to the inverse of $H$, instead of rounding each one independently, so the overall output error is much smaller.

## Derivatives, gradients and the chain rule {#导数梯度与链式法则}

A derivative measures how sensitive a function's value is to its input: $f'(x) = \lim_{h\to 0} \frac{f(x+h) - f(x)}{h}$. With several inputs, the partial derivatives with respect to each input form the **gradient** $\nabla f$, which points in the direction of fastest increase; gradient descent updates the parameters in the opposite direction.

The derivative of a composite function $y = f(g(x))$ is given by the **chain rule**: $\frac{dy}{dx} = \frac{dy}{dg}\cdot\frac{dg}{dx}$. A neural network is a composition of many functions, and computing its gradient means applying the chain rule over and over.

### Finite differences: the most naive numerical derivative {#有限差分最朴素的数值导数}

Approximate the derivative with $\frac{f(x+h) - f(x-h)}{2h}$ (the central difference), whose truncation error is of order $h^2$. But $h$ cannot be too small either: the numerator subtracts two very close numbers, and the floating-point rounding error is amplified by $1/h$. The two errors trade off against each other:

```python
import math
import torch

x = torch.tensor(1.0, dtype=torch.float64)
exact = math.cos(1.0)                                              # derivative of sin
print("步长 h      float32 误差      float64 误差")
for h in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8):
    errs = []
    for dtype in (torch.float32, torch.float64):
        xx, hh = x.to(dtype), torch.tensor(h, dtype=dtype)
        approx = ((torch.sin(xx + hh) - torch.sin(xx - hh)) / (2 * hh)).item()
        errs.append(abs(approx - exact))
    print(f"{h:7.0e}   {errs[0]:12.2e}   {errs[1]:12.2e}")
```

```text title="输出"
步长 h      float32 误差      float64 误差
  1e-01       9.00e-04       9.00e-04
  1e-02       7.06e-06       9.00e-06
  1e-03       1.38e-05       9.01e-08
  1e-04       1.38e-05       9.00e-10
  1e-05       8.80e-04       1.11e-11
  1e-06       3.86e-03       2.77e-11
  1e-08       5.40e-01       2.58e-09
```

In float32 the best step is around $10^{-2}$, with an error of order $10^{-5}$; with smaller steps the error grows instead, and at $10^{-8}$ it has reached 0.5, completely useless. float64 is far more precise: the best step is around $10^{-5}$ with an error of about $10^{-11}$. So **check gradients with finite differences in float64** (`torch.autograd.gradcheck` requires float64 inputs by default). The sources of numerical error are covered in the next chapter.

### Walking down the gradient: learning rate, momentum and the shape of the surface {#沿着梯度往下走学习率动量与曲面形状}

The gradient points in the direction of fastest ascent, so updates go the opposite way. But "how far to step" is a separate question: too large a step oscillates back and forth in steep directions or even diverges, too small a step takes forever. What really decides this is **the shape of the surface**: how many times the loss's curvature differs between directions (the condition number).

Drag the learning rate, momentum and condition number, and press "play" to see the path:

<div class="aig-widget" data-widget="graddesc"></div>

Three conclusions you can read off directly: (1) a learning rate above $2 / L$ ($L$ is the largest curvature) always diverges; (2) the flatter the ellipse, the more it oscillates along the steep direction and the slower it moves along the flat one; (3) momentum cancels the back-and-forth oscillation and effectively speeds up the flat direction. These are exactly the problems normalization (making the surface rounder), momentum, and Adam's per-dimension step sizes address (see [optimizers](train://algo/optimizer/)).

## Reverse-mode automatic differentiation {#反向模式自动微分}

![Figure: the forward pass records, the backward pass applies the chain rule](assets/figures/backprop-graph.svg){.aig-svg}

A neural network has billions of parameters and one scalar loss. **Reverse-mode** automatic differentiation starts from the loss and propagates "the derivative of the loss with respect to each intermediate" (the adjoint) back along the computation graph; one backward pass gives the gradients of all parameters at a cost comparable to one forward pass. Below is a minimal implementation (the same idea as Karpathy's micrograd): every node records its value, its gradient, its parents, and the rule for passing the gradient to its parents.

```python
class Value:
    def __init__(self, data, parents=(), backward=lambda: None):
        self.data, self.grad, self.parents, self._backward = data, 0.0, parents, backward

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def backward():                                            # d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += out.grad
            other.grad += out.grad
        out._backward = backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def backward():                                            # d(ab)/da = b, d(ab)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,))
        def backward():
            self.grad += out.data * out.grad                       # (e^x)' = e^x
        out._backward = backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,))
        def backward():
            self.grad += out.grad / self.data                      # (ln x)' = 1/x
        out._backward = backward
        return out

    def backward(self):
        order, seen = [], set()
        def visit(v):                                              # topological sort: parents first
            if v not in seen:
                seen.add(v)
                for p in v.parents:
                    visit(p)
                order.append(v)
        visit(self)
        self.grad = 1.0
        for v in reversed(order):                                  # start from the loss and propagate in reverse order
            v._backward()

# a small example: softmax cross-entropy over 3 logits, true class 1
logits = [Value(2.0), Value(0.5), Value(-1.0)]
exps = [z.exp() for z in logits]
total = exps[0] + exps[1] + exps[2]
loss = (total.log() + logits[1] * -1.0)                            # -log softmax(z)[1] = log Σe^z - z_1
loss.backward()

z = torch.tensor([2.0, 0.5, -1.0], requires_grad=True)
torch.nn.functional.cross_entropy(z[None], torch.tensor([1])).backward()
print("手写自动微分：", [round(v.grad, 6) for v in logits])
print("PyTorch：     ", [round(g, 6) for g in z.grad.tolist()])
print("softmax - onehot：", [round(p - (i == 1), 6) for i, p in enumerate(z.softmax(0).tolist())])
```

```text title="输出"
手写自动微分： [0.785597, -0.82471, 0.039113]
PyTorch：      [0.785597, -0.82471, 0.039113]
softmax - onehot： [0.785597, -0.82471, 0.039113]
```

All three agree. The last line shows a very important result: **the gradient of softmax cross-entropy with respect to the logits is $p - y$** (the predicted distribution minus the one-hot label). Derivation: $L = \log\sum_j e^{z_j} - z_y$, and differentiating with respect to $z_i$ gives $\frac{e^{z_i}}{\sum_j e^{z_j}} - [i = y] = p_i - y_i$. This gradient is simple and numerically stable, which is why frameworks always compute softmax and cross-entropy together.

## Backpropagation through a linear layer {#线性层的反向传播}

For $Y = XW^\top$ ($X$ is $[T, d_\text{in}]$, $W$ is $[d_\text{out}, d_\text{in}]$), given $G = \partial L/\partial Y$,

$$
\frac{\partial L}{\partial X} = G W, \qquad \frac{\partial L}{\partial W} = G^\top X
$$

The shapes help you remember: a gradient has the same shape as its variable, and only one combination of matrix multiplications produces the right shape.

```python
torch.manual_seed(0)
X = torch.randn(8, 16, requires_grad=True)
W = torch.randn(32, 16, requires_grad=True)
Y = X @ W.T
G = torch.randn_like(Y)                                            # the gradient coming from upstream
Y.backward(G)
print("dX = G W：  ", torch.allclose(X.grad, G @ W, atol=1e-5))
print("dW = Gᵀ X： ", torch.allclose(W.grad, G.T @ X, atol=1e-5))
```

```text title="输出"
dX = G W：   True
dW = Gᵀ X：  True
```

The forward pass is one matrix multiplication ($2 T d_\text{in} d_\text{out}$ operations), and the backward pass is **two** matrix multiplications of the same size (one computes $\partial L/\partial X$ to pass to the previous layer, the other $\partial L/\partial W$ for the update). So training costs about 3 times the forward compute per token: $2N + 4N = 6N$, which is where the training compute formula $C \approx 6ND$ comes from (see [pretraining](llm://training/pretraining/#训练需要多少算力)). Inference only does the forward pass, about $2N$ per token.

## Second-order information: GPTQ {#二阶信息gptq}

The gradient is the first derivative; the **Hessian** (the matrix of second derivatives) describes the function's curvature. Near an optimum, the loss can be approximated by a quadratic: $\Delta L \approx \frac{1}{2}\Delta w^\top H \Delta w$.

When quantizing a linear layer, we want the quantized output to change as little as possible: minimize $\|XW^\top - X\hat{W}^\top\|^2$. For each row $w$ of the weights, this objective is quadratic:

$$
\|X(w - \hat w)^\top\|^2 = (w - \hat w)\, H\, (w - \hat w)^\top, \qquad H = X^\top X
$$

$H$ is determined by the activations (that is the second-order information). Rounding each weight to nearest (RTN) amounts to assuming $H$ is diagonal and ignores the correlation between input dimensions. **GPTQ** (building on the earlier OBS/OBQ) quantizes column by column, and after each column uses $H^{-1}$ to "compensate" the error that column introduced on the columns not yet quantized, nudging them in the direction that cancels the error. A Cholesky decomposition of $H^{-1}$ makes this efficient and stable.

Implement it on a real model: first collect $H = X^\top X$ of every linear layer's input with a calibration text (from this book's tokenization chapter, different from the evaluation text), then quantize all linear layers to per-channel INT4 and compare the perplexity of RTN and GPTQ:

```python
import copy
import re
from transformers import AutoTokenizer
from mini_llm import Transformer
from quant import fake_quant_int

torch.set_num_threads(16)
path = "models/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(path)
model = Transformer.from_pretrained(path)
text = ("大语言模型的推理过程分为两个阶段。在预填充阶段，模型一次性处理用户输入的全部提示词，计算每个位置的键和值并写入缓存。"
        "在解码阶段，模型每次只生成一个新的词元，需要读取全部的模型权重和已经缓存的键值。由于每一步的计算量很小而读取的数据量很大，"
        "解码阶段通常受限于显存带宽。为了提高吞吐量，推理系统会把多个请求合并成一个批次，让它们共享同一次权重读取。"
        "量化通过降低权重和激活的数值精度来减少需要读取的字节数，是加速解码的常用方法。")
eval_ids = tok(text, return_tensors="pt").input_ids                  # the same evaluation text as the quantization chapter of the LLM book
raw = open("docs/assets/calib-passage.txt").read()         # frozen calibration text: a snapshot of the tokenization chapter
calib_ids = tok(re.sub(r"[#*`>|\-\[\]()!]", "", raw)).input_ids[:512]    # calibration text

def perplexity(m):
    with torch.no_grad():
        logits = m(eval_ids)
    return math.exp(torch.nn.functional.cross_entropy(logits[0, :-1], eval_ids[0, 1:]).item())

def linears(m):
    for layer in m.layers:
        a, f = layer.self_attn, layer.mlp
        yield from (a.q_proj, a.k_proj, a.v_proj, a.o_proj, f.gate_proj, f.up_proj, f.down_proj)

hessians, hooks = {}, []
for lin in linears(model):                                         # collect H = XᵀX of every linear layer's input
    def hook(mod, inp, out, lin=lin):
        x = inp[0].reshape(-1, inp[0].shape[-1])
        hessians[lin] = x.T @ x
    hooks.append(lin.register_forward_hook(hook))
with torch.no_grad():
    model(torch.tensor([calib_ids]))
for h in hooks:
    h.remove()

def gptq(W, H, bits=4, block=128):
    """按列量化，把每列的量化误差用 H⁻¹ 补偿到后面的列上（按块延迟更新，与原论文一致）。"""
    W, n = W.clone(), W.shape[1]
    H = H + 0.01 * H.diag().mean() * torch.eye(n)                 # damping, to keep it invertible
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    qmax = 2 ** (bits - 1) - 1
    scale = W.abs().amax(1) / qmax                                  # per-channel scales
    Q = torch.zeros_like(W)
    for a in range(0, n, block):
        b = min(a + block, n)
        W1, errors = W[:, a:b].clone(), torch.zeros(W.shape[0], b - a)
        for i in range(b - a):
            q = (W1[:, i] / scale).round().clamp(-qmax - 1, qmax) * scale
            Q[:, a + i] = q
            err = (W1[:, i] - q) / Hinv[a + i, a + i]
            W1[:, i:] -= err[:, None] * Hinv[a + i, a + i:b][None, :]    # compensate on the remaining columns of this block
            errors[:, i] = err
        W[:, b:] -= errors @ Hinv[a:b, b:]                                # compensate on all later columns
    return Q

rtn, gq = copy.deepcopy(model), copy.deepcopy(model)
for lin_rtn, lin_gptq, lin in zip(linears(rtn), linears(gq), linears(model)):
    lin_rtn.weight.data = fake_quant_int(lin.weight.data, 4, "channel")
    lin_gptq.weight.data = gptq(lin.weight.data, hessians[lin])
print(f"FP32 原始 {perplexity(model):.2f}；INT4 按通道：四舍五入（RTN）{perplexity(rtn):.2f}，GPTQ {perplexity(gq):.2f}")
```

```text title="输出"
FP32 原始 25.94；INT4 按通道：四舍五入（RTN）55.02，GPTQ 36.85
```

The same per-channel INT4 format and the same scales, just a different way of "deciding where each weight rounds to", cut the perplexity loss by about sixty percent (the gap to the original model shrinks from 29.1 to 10.9). GPTQ does no training; it uses only a calibration text and some linear algebra. In practice GPTQ is usually combined with group-wise scales (for example one scale per 128 numbers), which brings the accuracy closer to the original model (see [how quantization works](llm://inference/quantization/#gptq-与-awq)).

!!! interview "In an interview"
    When asked about backpropagation, make three points: reverse-mode automatic differentiation starts from the loss, and one backward pass gives the gradients of all parameters, which suits "many parameters, one loss"; the backward pass of a linear layer $Y = XW^\top$ is two matrix multiplications ($dX = dY\,W$, $dW = dY^\top X$), so training costs about 6N per token and inference 2N; the gradient of softmax plus cross-entropy with respect to the logits is $p - y$, which is why a fused cross-entropy kernel does not need to store the full softmax. Check gradients with finite differences in float64: too small a step drowns in rounding error.

## Exercises {#练习}

**1. Why reverse mode?** A network has $n$ parameters and one scalar loss. How many passes does forward-mode automatic differentiation (propagating derivatives along one input direction at a time) need to get all the gradients? And reverse mode? When is forward mode the better deal?

??? success "Answer"
    Forward mode gives the derivative of the loss along one input direction per pass, so all $n$ parameter gradients take $n$ passes; reverse mode gets the gradients of all parameters in one backward pass. So "many inputs, few outputs" (training a neural network) calls for reverse mode, and "few inputs, many outputs" (for example the sensitivity of a function to a single hyperparameter, or Jacobian-vector products) is cheaper in forward mode. The price of reverse mode is keeping the forward intermediates (activations), which is why training needs so much memory and why activation recomputation exists.

**2. LoRA's gradients.** For $Y = X(W + BA)^\top$ with $W$ frozen, train only $A$ ($r \times d_\text{in}$) and $B$ ($d_\text{out} \times r$). Write down $\partial L/\partial A$ and $\partial L/\partial B$, and explain why LoRA saves memory.

??? success "Answer"
    Let $G = \partial L/\partial Y$ and $Z = XA^\top$ ($[T, r]$). The LoRA part of $Y$ is $ZB^\top$, so $\partial L/\partial B = G^\top Z$, $\partial L/\partial Z = GB$, and $\partial L/\partial A = (GB)^\top X$. Only $A$ and $B$ need optimizer state (about a hundredth of the original matrix's parameters), and $W$ needs neither gradients nor optimizer state, so memory drops a lot; but $\partial L/\partial X$ still has to pass through $W$ to the previous layer, so the forward and backward compute does not shrink much.

## Summary {#小结}

- [x] Finite differences are squeezed between truncation error and rounding error; check gradients in float64.
- [x] Reverse-mode automatic differentiation propagates adjoints back from the loss, and one backward pass gives the gradients of all parameters; the minimal hand-written implementation agrees with PyTorch.
- [x] The gradient of softmax cross-entropy with respect to the logits is $p - y$; the backward pass of a linear layer is two matrix multiplications, so training costs about $6N$ per token and inference about $2N$.
- [x] GPTQ compensates errors using the activations' second-order information $H = X^\top X$, and in this book's experiment it cut the perplexity loss of per-channel INT4 quantization by about 60%.
