# 前缀和与其他并行模式

<p class="lead">前缀和（scan）看起来是个"天生串行"的问题：第 i 个结果依赖前面所有元素。它的并行解法是 GPU 算法设计的经典范例，也是流压缩、基数排序、MoE 的 token 分发、top-p 采样等大量算法的基础构件。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 包含式（inclusive）和排除式（exclusive）前缀和有什么区别？
    2. Hillis-Steele 和 Blelloch 两种扫描算法，各自的工作量是多少？
    3. 怎么用 warp shuffle 完成 32 个数的前缀和？一个 block 的呢？
    4. 数组大到需要多个 block 时，怎么把各 block 的结果拼起来？
    5. 流压缩（筛选出满足条件的元素）和前缀和有什么关系？

## 定义

输入 `[3, 1, 7, 0, 4, 1, 6, 3]`：

- **包含式前缀和**：`[3, 4, 11, 11, 15, 16, 22, 25]`，第 i 项包含 `x[i]` 本身；
- **排除式前缀和**：`[0, 3, 4, 11, 11, 15, 16, 22]`，第 i 项是 `x[0..i-1]` 之和。

排除式前缀和最常用的解释是**"我前面有多少个"**，也就是"我的数据应该写到哪个位置"。这就是它在并行算法里无处不在的原因。

## 两种经典的并行算法

**Hillis-Steele**：进行 log n 轮，第 d 轮每个元素加上它前面距离 2^d 的元素。步数少（log n），但总工作量是 O(n log n)，比串行的 O(n) 多。

**Blelloch**：分两个阶段，先像归约一样自底向上构建部分和树（up-sweep），再自顶向下分发（down-sweep）。总工作量 O(n)，步数 2 log n。

在 GPU 上的实际做法是分层组合：**warp 内用 Hillis-Steele 风格的 shuffle**（32 个元素，5 步，没有同步开销，多出来的工作量无所谓），**warp 之间和 block 之间再往上套一层**。

## warp 扫描与 block 扫描

warp 内扫描在[上一章的练习](../basics/sync-warp.md#练习)里已经出现过：

```cuda
__device__ int warp_inclusive_scan(int v) {
  const int lane = threadIdx.x % 32;
  for (int offset = 1; offset < 32; offset *= 2) {
    int n = __shfl_up_sync(0xffffffff, v, offset);
    if (lane >= offset) v += n;
  }
  return v;
}
```

block 扫描在它上面再套一层：

1. 每个 warp 做 warp 内扫描；
2. 每个 warp 的最后一个 lane 把本 warp 的总和写入共享内存；
3. 第 0 个 warp 对这些 warp 总和做一次扫描（最多 32 个）；
4. 每个线程加上"前面所有 warp 的总和"。

## 设备级扫描：scan-then-propagate

数组大于一个 block 能处理的范围时，经典的做法分三步：

1. **分块扫描**：每个 block 扫描自己的一段，并把这一段的总和写到 `block_sums[blockIdx.x]`；
2. **扫描块总和**：对 `block_sums` 做排除式扫描，得到每个 block 的起始偏移；
3. **传播**：每个 block 给自己段内的所有元素加上起始偏移。

这需要读两遍、写两遍数据。CUB 的 `DeviceScan` 使用更先进的 **decoupled look-back** 单遍算法（Merrill & Garland, 2016）：每个 block 算完自己的局部和后，向前查看前面 block 公布的状态（"只有局部和"或"已有包含前缀"），一旦拿到前一个 block 的包含前缀就能确定自己的偏移。它只需读一遍、写一遍，性能接近拷贝。

```cuda title="scan.cu"
// scan.cu —— warp 扫描 → block 扫描 → 三步法设备级扫描（排除式前缀和）
// 编译：nvcc -O3 -arch=sm_75 scan.cu -o scan
#include "common.cuh"

constexpr int kThreads = 256;
constexpr int kItems = 4;                       // 每个线程处理 4 个元素
constexpr int kTile = kThreads * kItems;        // 每个 block 处理 1024 个元素

__device__ __forceinline__ int warp_inclusive_scan(int v) {
  const int lane = threadIdx.x % 32;
#pragma unroll
  for (int offset = 1; offset < 32; offset *= 2) {
    int n = __shfl_up_sync(0xffffffff, v, offset);
    if (lane >= offset) v += n;
  }
  return v;
}

// block 内排除式扫描：返回本线程之前（不含本线程）所有线程的 v 之和，total 为整个 block 的和
__device__ int block_exclusive_scan(int v, int& total) {
  __shared__ int warp_totals[32];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, nwarps = blockDim.x / 32;
  int inclusive = warp_inclusive_scan(v);
  if (lane == 31) warp_totals[warp] = inclusive;
  __syncthreads();
  if (warp == 0) {
    int t = lane < nwarps ? warp_totals[lane] : 0;
    int s = warp_inclusive_scan(t);
    if (lane < nwarps) warp_totals[lane] = s - t;   // 每个 warp 的起始偏移（排除式）
    if (lane == nwarps - 1) warp_totals[31] = s;     // 暂存 block 总和
  }
  __syncthreads();
  total = warp_totals[31];
  int result = warp_totals[warp] + inclusive - v;
  __syncthreads();   // 允许同一个 kernel 里再次调用
  return result;
}

// 第 1 步：每个 block 扫描自己的 kTile 个元素，写出块内的排除式前缀和与块总和
__global__ void scan_tiles(const int* __restrict__ in, int* __restrict__ out, int* __restrict__ block_sums, int n) {
  const int base = blockIdx.x * kTile + threadIdx.x * kItems;
  int items[kItems];
  int thread_sum = 0;
#pragma unroll
  for (int k = 0; k < kItems; ++k) {
    items[k] = base + k < n ? in[base + k] : 0;
    thread_sum += items[k];
  }
  int total;
  int running = block_exclusive_scan(thread_sum, total);   // 本线程之前的所有元素之和
#pragma unroll
  for (int k = 0; k < kItems; ++k) {
    if (base + k < n) out[base + k] = running;
    running += items[k];
  }
  if (threadIdx.x == 0) block_sums[blockIdx.x] = total;
}

// 第 2 步：用一个 block 对 block_sums 做排除式扫描（按 kThreads 一段一段处理任意数量的块）
__global__ void scan_block_sums(int* block_sums, int num_blocks) {
  __shared__ int carry;
  if (threadIdx.x == 0) carry = 0;
  __syncthreads();
  for (int start = 0; start < num_blocks; start += blockDim.x) {
    int i = start + threadIdx.x;
    int v = i < num_blocks ? block_sums[i] : 0;
    int total;
    int ex = block_exclusive_scan(v, total);
    if (i < num_blocks) block_sums[i] = ex + carry;
    __syncthreads();
    if (threadIdx.x == 0) carry += total;
    __syncthreads();
  }
}

// 第 3 步：每个元素加上所在块的起始偏移
__global__ void add_offsets(int* __restrict__ out, const int* __restrict__ block_sums, int n) {
  const int base = blockIdx.x * kTile + threadIdx.x * kItems;
  const int offset = block_sums[blockIdx.x];
#pragma unroll
  for (int k = 0; k < kItems; ++k)
    if (base + k < n) out[base + k] += offset;
}

int main() {
  const int n = 10'000'000 + 7;
  std::vector<int> h(n), ref(n), got(n);
  std::mt19937 gen(1);
  std::uniform_int_distribution<int> dist(0, 9);
  for (auto& v : h) v = dist(gen);
  int acc = 0;
  for (int i = 0; i < n; ++i) { ref[i] = acc; acc += h[i]; }

  const int num_blocks = (n + kTile - 1) / kTile;
  int *d_in, *d_out, *d_sums;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_sums, num_blocks * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_in, h.data(), n * sizeof(int), cudaMemcpyHostToDevice));

  auto run = [&] {
    scan_tiles<<<num_blocks, kThreads>>>(d_in, d_out, d_sums, n);
    scan_block_sums<<<1, kThreads>>>(d_sums, num_blocks);
    add_offsets<<<num_blocks, kThreads>>>(d_out, d_sums, n);
  };
  run();
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(got.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
  size_t bad = 0;
  for (int i = 0; i < n; ++i) bad += got[i] != ref[i];
  std::printf("exclusive scan of %d ints: %s (%zu mismatches)\n", n, bad ? "FAIL" : "PASS", bad);

  float ms = time_ms(run);
  // 读 in、写 out、再读写一次 out（第 3 步）
  std::printf("time %.3f ms, %.1f GB/s effective (4 passes over the data)\n", ms, gbps(4.0 * n * sizeof(int), ms));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_sums));
  return bad ? 1 : 0;
}
```

工程中直接用 `cub::DeviceScan::ExclusiveSum`，它的接口和 `DeviceReduce` 一样是两次调用（先查询临时空间，再执行）。

## 流压缩

**流压缩（stream compaction）**：从数组里挑出满足条件的元素，紧凑地写到输出里。比如去掉 0、筛选出被选中的 token。关键问题是每个被选中的元素应该写到哪里：答案是"排在我前面的被选中元素的个数"，也就是对 0/1 标志做排除式前缀和。

在 warp 内，这个前缀和可以用投票指令一步算出：

```cuda
unsigned mask = __ballot_sync(0xffffffff, keep);            // 哪些 lane 要保留
int rank = __popc(mask & ((1u << lane) - 1));               // 我前面有几个要保留的
int count = __popc(mask);                                   // 这个 warp 一共保留几个
```

如果**不要求保持原来的顺序**，还可以更简单：每个 warp 由 lane 0 做一次 `atomicAdd` 申请 `count` 个位置，再用 shuffle 把起始位置广播给其他 lane。这叫 **warp 聚合原子操作**，全局原子操作的次数降到了 1/32：

```cuda title="compact.cu"
// compact.cu —— 用 ballot + popc + warp 聚合原子操作做流压缩（不保序）
// 编译：nvcc -O3 -arch=sm_75 compact.cu -o compact
#include "common.cuh"

__global__ void compact_positive(const float* __restrict__ in, float* __restrict__ out,
                                 int* __restrict__ count, int n) {
  const int lane = threadIdx.x % 32;
  // 整个 warp 一起循环，保证 __ballot_sync 时 warp 内线程都在
  for (int base = blockIdx.x * blockDim.x; base < n; base += blockDim.x * gridDim.x) {
    const int i = base + threadIdx.x;
    const float v = i < n ? in[i] : 0.f;
    const bool keep = i < n && v > 0.f;
    const unsigned mask = __ballot_sync(0xffffffff, keep);
    const int rank = __popc(mask & ((1u << lane) - 1));
    int start = 0;
    if (lane == 0 && mask) start = atomicAdd(count, __popc(mask));
    start = __shfl_sync(0xffffffff, start, 0);
    if (keep) out[start + rank] = v;
  }
}

int main() {
  const int n = 1 << 22;
  std::vector<float> h(n);
  fill_random(h, 3);
  std::vector<float> ref;
  for (float v : h) if (v > 0.f) ref.push_back(v);

  float *d_in, *d_out;
  int* d_count;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_in, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));
  compact_positive<<<sm_count() * 8, 256>>>(d_in, d_out, d_count, n);
  CUDA_CHECK_LAST();
  int count = 0;
  CUDA_CHECK(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
  std::vector<float> got(count);
  CUDA_CHECK(cudaMemcpy(got.data(), d_out, count * sizeof(float), cudaMemcpyDeviceToHost));
  // 顺序不保证，排序后比较
  std::sort(got.begin(), got.end());
  std::sort(ref.begin(), ref.end());
  bool ok = got == ref;
  std::printf("kept %d of %d: %s\n", count, n, ok ? "PASS" : "FAIL");
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_count));
  return ok ? 0 : 1;
}
```

注意循环的写法：不是每个线程各自判断 `i < n` 后跳出，而是整个 warp 一起按 `base` 循环，越界的线程也参与 `__ballot_sync`，只是 `keep` 为假。这样保证 warp 级原语执行时 mask 里的线程都在。

## 在大模型里的应用：MoE 的 token 分发

混合专家（MoE）模型的每一层，每个 token 被路由器分配给 top-k 个专家。为了让每个专家用一次 GEMM 处理分给它的所有 token，需要把 token 按专家重新排列：

1. **计数**：统计每个专家分到多少个 token（直方图）；
2. **排除式前缀和**：得到每个专家在重排后数组里的起始位置；
3. **分散（scatter）**：每个 token 按"专家起始位置 + 在该专家内的序号"写到对应位置（序号可以用原子操作获得）；
4. 分组 GEMM 计算后，再按原顺序收集（gather）回来并按路由权重加权求和。

vLLM 的 `moe_align_block_size`、SGLang 和 DeepSeek 开源的 DeepEP 等代码里都能看到这套"直方图 + 前缀和 + 分散"的结构。top-p 采样里对排序后的概率求累积和，也是前缀和。

## 练习

**1. 保序的流压缩。** 用 `scan.cu` 里的三步法实现保持原顺序的流压缩：先把"是否保留"转换成 0/1 标志，对标志做排除式扫描得到每个元素的输出位置，再写出。

??? success "参考思路"
    最直接的实现是三个 kernel：`flags[i] = in[i] > 0`；对 `flags` 做排除式扫描得到 `pos`；`if (flags[i]) out[pos[i]] = in[i]`，保留的总数是 `pos[n-1] + flags[n-1]`。更高效的做法是把标志计算融合进扫描的第一步、把写出融合进第三步，避免把 `flags` 和 `pos` 写回显存。CUB 的 `DeviceSelect::If` 就是单遍完成的。

**2. MoE 计数与偏移。** 给定 `num_tokens × top_k` 的专家编号数组 `topk_ids`，专家数为 E，输出每个专家的 token 数 `counts[E]` 和起始偏移 `offsets[E]`。写出 kernel 的结构。

??? success "参考思路"
    - 计数用共享内存直方图（E 一般是几十到几百，放得进共享内存），每个 block 处理一段，最后原子加到全局 `counts`，参见[直方图](../basics/sync-warp.md#直方图共享内存私有化)；
    - E 不大时，偏移可以由一个 block 用 `block_exclusive_scan` 直接算出；
    - 实际实现中常把偏移对齐到 GEMM 分块大小的整数倍（比如 64），方便后面的分组 GEMM 按块处理，这正是 vLLM 里 `moe_align_block_size` 名字的由来。

## 小结

- [x] 排除式前缀和 = "我前面有多少"= "我该写到哪里"。
- [x] warp 内用 shuffle 做 Hillis-Steele；block 内在 warp 扫描上再套一层；设备级用三步法或 decoupled look-back。
- [x] 流压缩用 ballot + popc 算出 warp 内偏移，warp 聚合原子操作减少争用。
- [x] MoE 的 token 分发是"直方图 + 前缀和 + 分散"的典型应用。
