# A tensor's memory model: storage, strides and views

<p class="lead">Writing a custom kernel or reading an inference framework's code, the first thing to get straight is what a <code>torch.Tensor</code> actually looks like in memory. The answer is "one block of storage plus metadata describing how to look at it". A transpose, a slice and an <code>expand</code> copy nothing and merely change the view; while kernels usually assume the data is contiguous. This chapter makes that relationship clear.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. After transposing a contiguous `(3, 4)` tensor, what is its `stride`? Is it still contiguous?
    2. How do `view` and `reshape` differ? When does `view` fail?
    3. How do `torch.repeat` and `torch.expand` differ in memory?
    4. What happens when you write one element into a tensor produced by `expand`?
    5. A custom kernel receives a non-contiguous input. What are the two ways to handle it?

??? success "Answers (try it yourself first, then expand)"
    1. The stride goes from `(4, 1)` to `(1, 4)` (with the shape becoming `(4, 3)`) and it is no longer contiguous.
    2. `view` only changes metadata and requires the new shape to be describable with one set of strides over the existing memory, raising otherwise; `reshape` returns a view when it can and copies when it cannot. `view` fails when the tensor is non-contiguous and the new shape would merge modes that do not join (flattening after a transpose, say).
    3. `expand` sets the stride of a size-1 dimension to 0 so every index points at the same data, copying nothing; `repeat` really allocates new memory.
    4. Several indices point at the same location, so writing one element changes many "elements" at once; PyTorch raises outright for most in-place operations, so `clone()` or `contiguous()` first.
    5. Either call `.contiguous()` first and work on the copy (simple, at the cost of one copy), or pass the strides into the kernel and compute addresses from them (no copy, but possibly losing coalescing and vectorization).

## Storage plus metadata {#storage--元数据}

A tensor is two things:

- **storage**: one one-dimensional contiguous block of raw memory (host or device) that only knows how many bytes it has;
- **metadata**: `dtype`, `shape`, `stride` (how many **elements** a step along each dimension skips) and `storage_offset` (which element of the storage it starts at).

The address of `x[i, j]` is `storage_offset + i * stride[0] + j * stride[1]`. A **view** is another tensor sharing the same storage with different metadata:

```python title="storage_views.py"
import torch

x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
print("原张量：shape", tuple(x.shape), "stride", x.stride(), "连续", x.is_contiguous())

t = x.t()                                  # a transpose: just the shape and the stride swapped
print("转置：shape", tuple(t.shape), "stride", t.stride(), "连续", t.is_contiguous())
print("共享存储：", t.untyped_storage().data_ptr() == x.untyped_storage().data_ptr())

s = x[:, 1:3]                              # a slice: a different offset and shape
print("切片：shape", tuple(s.shape), "stride", s.stride(), "offset", s.storage_offset(), "连续", s.is_contiguous())

t[0, 1] = 100.0                            # writing through the view changes the original too
print("通过视图修改后 x[1, 0] =", x[1, 0].item())

try:
    t.view(12)
except RuntimeError as e:
    print("view 失败：", str(e).split(" (")[0])
r = t.reshape(12)                          # reshape: a view when it can, a copy when it cannot
print("reshape 拷贝了：", r.untyped_storage().data_ptr() != x.untyped_storage().data_ptr())
print("contiguous() 之后 stride", t.contiguous().stride())
```

```text title="output"
原张量：shape (3, 4) stride (4, 1) 连续 True
转置：shape (4, 3) stride (1, 4) 连续 False
共享存储： True
切片：shape (3, 2) stride (4, 1) offset 1 连续 False
通过视图修改后 x[1, 0] = 100.0
view 失败： view size is not compatible with input tensor's size and stride
reshape 拷贝了： True
contiguous() 之后 stride (3, 1)
```

- **contiguous** means laid out row-major with no gaps: the last dimension's stride is 1 and each dimension's stride is the product of the sizes after it;
- `view` only changes metadata and requires the new shape to be describable with one set of strides over the existing memory, raising otherwise; `reshape` returns a view when it can and copies otherwise, so after a `reshape` you **do not know** whether it shares memory with the original;
- `contiguous()` returns the tensor itself when it is already contiguous and copies into a new contiguous one otherwise.

Switch between operations and watch the sizes, the strides and which elements of the storage are read:

<div class="aig-widget" data-widget="stride-view"></div>

## expand and repeat: broadcasting with a zero stride {#expand-与-repeat零-stride-的广播}

`expand` "broadcasts" a dimension of size 1 to a larger size by setting that dimension's stride to **0**: however many steps you take along it you stay in the same place, so nothing is allocated. `repeat` really copies the data several times:

```python title="expand_repeat.py"
import torch

bias = torch.tensor([1.0, 2.0, 3.0])
e = bias.expand(4, 3)              # all 4 rows point at the same memory
r = bias.repeat(4, 1)              # really copied into 4 rows
print("expand：stride", e.stride(), "存储字节", e.untyped_storage().nbytes())
print("repeat：stride", r.stride(), "存储字节", r.untyped_storage().nbytes())

e[0, 0] = 5.0                      # all 4 rows share this element: changing one changes a column
print("写 e[0, 0] 之后 bias =", bias.tolist(), "e[3, 0] =", e[3, 0].item())
try:
    e.add_(1)                      # an in-place operation would write one location several times, so it raises
except RuntimeError as err:
    print("原地 add_ 失败：", str(err).split(":")[0])

win = torch.arange(6).unfold(0, 3, 1)   # a sliding window is a view too: neighbouring windows are 1 apart
print("滑动窗口：stride", win.stride(), "内容", win.tolist())
```

```text title="output"
expand：stride (0, 1) 存储字节 12
repeat：stride (3, 1) 存储字节 48
写 e[0, 0] 之后 bias = [5.0, 2.0, 3.0] e[3, 0] = 5.0
原地 add_ 失败： unsupported operation
滑动窗口：stride (1, 1) 内容 [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]
```

"How do `torch.repeat` and `torch.expand` differ" is a common interview question: `expand` copies nothing and its result cannot be written in place safely, while `repeat` allocates. A sliding-window view like `unfold` is often used for n-gram matching (drafting with fragments already in the prompt for speculative decoding), giving every window without a copy.

### An example: whether GQA's KV heads need copying {#例子gqa-的-kv-头要不要复制}

In GQA, 8 query heads share 2 KV heads. The most direct approach copies the KV heads into 8 and runs ordinary multi-head attention; another reshapes the queries by group and lets the KV broadcast along the "group" dimension:

```python title="gqa_broadcast.py"
import torch

torch.manual_seed(0)
b, hq, hkv, s, d = 1, 8, 2, 16, 64
g = hq // hkv
q = torch.randn(b, hq, 1, d)        # decode: one query per head
k = torch.randn(b, hkv, s, d)

k_rep = k.repeat_interleave(g, dim=1)                    # version one: the KV replicated into 8 heads
s1 = q @ k_rep.transpose(-1, -2)
s2 = (q.view(b, hkv, g, 1, d) @ k.view(b, hkv, 1, s, d).transpose(-1, -2)).view(b, hq, 1, s)   # version two: broadcast by group
print("两种写法结果一致：", torch.allclose(s1, s2, atol=1e-5))
print("复制出来的 K 是原来的", k_rep.untyped_storage().nbytes() // k.untyped_storage().nbytes(), "倍")
```

```text title="output"
两种写法结果一致： True
复制出来的 K 是原来的 4 倍
```

At the PyTorch level, the `matmul` in the second version may still expand the broadcast operand internally to reach a batched matrix multiply; what really saves memory is an attention kernel with **native GQA support**: inside, `kv_head = q_head / group` says which KV head to read, and the KV is never replicated to the query head count. FlashAttention and FlashInfer both do this, and for decode, a pure bandwidth bottleneck, it means reading `g` times less KV.

## Non-contiguous tensors and kernels {#非连续的张量与-kernel}

Nearly every custom kernel accesses data as "a pointer plus contiguous indices", so a non-contiguous input leaves two options:

1. **call `.contiguous()` first**: simple, at the cost of an extra copy (one read and one write);
2. **pass the strides into the kernel**: the kernel accesses `base + i * stride0 + j * stride1` with no copy. vLLM's `reshape_and_cache` passes `key.stride(0)` into the kernel precisely because `key` is often a view sliced out of a fused QKV tensor (see the C++ handbook's [reading an inference library's C++](cpp://engineering/reading-code/)).

Two more pieces of metadata that are easy to overlook:

- **`dtype` and `element_size()`**: bf16 / fp16 are 2 bytes per element and fp8 is 1; a kernel computing addresses in bytes has to multiply by it;
- **alignment of the storage**: `untyped_storage().data_ptr()` is usually aligned to 64 bytes or more, but a view with a non-zero `storage_offset` may start aligned only to the element size, so check the alignment before a 16-byte vectorized read.

!!! interview "Answering in an interview"
    On a tensor's memory model: a tensor is one block of storage plus metadata (dtype, shape, stride, offset), and a view shares the storage with different metadata; a transpose merely swaps the shape and the stride and is no longer contiguous; `view` only changes metadata and may fail while `reshape` views when it can and copies otherwise (so you do not know whether it shares memory); `expand` broadcasts with a zero stride and cannot be written in place safely, while `repeat` really copies. A custom kernel receiving a non-contiguous input either calls `contiguous()` or takes the strides, and it checks the alignment before a vectorized read.

## Exercises {#练习}

1. For a contiguous tensor of shape `(2, 3, 4)`, what is the `stride` after `permute(2, 0, 1)`? Is it contiguous? What happens if you then `view(4, 6)`?

??? success "Answer"
    The original stride is `(12, 4, 1)`; `permute(2, 0, 1)` reorders the dimensions and the stride follows into `(1, 12, 4)` with shape `(4, 2, 3)`, which is not contiguous. `view(4, 6)` would merge the last two dimensions, which requires `stride[1] == shape[2] * stride[2]`, that is `12 == 3 * 4`, which happens to hold, so the view succeeds.

    ```python title="permute_stride.py"
    import torch

    x = torch.arange(24).reshape(2, 3, 4)
    p = x.permute(2, 0, 1)
    print(tuple(p.shape), p.stride(), p.is_contiguous())
    v = p.view(4, 6)
    print(tuple(v.shape), v.stride(), v.untyped_storage().data_ptr() == x.untyped_storage().data_ptr())
    ```

    ```text title="output"
    (4, 2, 3) (1, 12, 4) False
    (4, 6) (1, 4) True
    ```

2. A decode step has `batch` requests, each needing its own blocks from the KV cache. Why does an inference engine pass the block table to the kernel rather than "stitching" each request's KV into one large contiguous tensor and calling attention?

??? success "Answer"
    Stitching is a complete copy: every step would read and write all the requests' KV, and since decode is already bandwidth-bound that multiplies attention's memory traffic.
    Passing the block table has the kernel address indirectly, "logical position to block number to physical address", with the data staying where it is. That is the heart of paged attention, and an extension of this chapter's "pass strides or indices rather than copying" (see the Inference Systems handbook's [paged KV cache](serving://engine/paged-kv/)).

## Summary {#小结}

- [x] A tensor is storage (one contiguous block) plus metadata (dtype, shape, stride, offset); a view shares the storage with different metadata.
- [x] `view` only changes metadata and may fail; `reshape` views when it can and copies otherwise; `contiguous()` copies when it must.
- [x] `expand` broadcasts with a zero stride, copies nothing and cannot be written in place safely; `repeat` really copies.
- [x] Native GQA in an attention kernel and a block table for paged KV both exist to avoid copying data.
- [x] A kernel handling a non-contiguous input either calls `contiguous()` or takes the strides; check the alignment before a vectorized read.
