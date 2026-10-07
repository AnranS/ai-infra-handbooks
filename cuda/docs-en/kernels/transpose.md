# Matrix transpose

<p class="lead">A transpose computes nothing, yet it is the best example there is for understanding memory optimization: the read and the write cannot both be contiguous along rows, so a naive implementation necessarily leaves one side uncoalesced. The answer is to stage through shared memory, which in turn brings up bank conflicts. The same idea recurs in GEMM and attention: change the data's layout in shared memory so that both the global reads and the global writes coalesce.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. In the naive transpose `out[x][y] = in[y][x]`, which side coalesces, the read or the write?
    2. Why do both coalesce once you stage through shared memory?
    3. Why is the shared-memory tile declared `[32][33]`?
    4. What is a transpose kernel's performance ceiling? What do you compare against?
    5. Why is a 32×8 block so often used for a 32×32 tile?

??? success "Answers (try it yourself first, then expand)"
    1. The read coalesces: a warp's threads read consecutive elements of one row. The write does not: they write one column of the output, a whole row apart each.
    2. The tile is read row-wise from the input into shared memory (a coalesced read), then taken out column-wise and written row-wise to the output (a coalesced write): the non-contiguous step happens in shared memory, which is randomly accessible.
    3. Reading a `[32][32]` tile by column puts 32 elements in one bank, a 32-way conflict; one extra column spreads them over 32 different banks.
    4. The ceiling is the bandwidth of a plain copy kernel of the same size: a transpose reads and writes exactly as many bytes as a copy, only in a different order.
    5. 32×8 = 256 threads, 4 elements each: the block is not too large (which keeps occupancy flexible), each thread has several accesses in flight, and a warp still reads and writes exactly one row of 32 elements.

## What is wrong with the naive version {#朴素实现的问题}

The input `in` is a `rows × cols` row-major matrix and the output `out` is `cols × rows`. The naive version gives thread `(x, y)` one element:

```cuda
int x = blockIdx.x * 32 + threadIdx.x;   // column
int y = blockIdx.y * 32 + threadIdx.y;   // row
out[x * rows + y] = in[y * cols + x];
```

A warp's threads have consecutive `threadIdx.x`: reading `in[y][x]` touches consecutive addresses and **coalesces**; writing `out[x][y]` puts `rows` elements between neighbouring threads, which **does not coalesce at all**, with each thread claiming its own 32-byte sector and write bandwidth reduced to an eighth. Swapping the roles of the read and the write only moves the problem to the other side.

## Staging through shared memory {#共享内存中转}

![Figure: a matrix transpose, reading rows into a shared-memory tile and taking columns out to write, with both sides coalesced](../assets/figures/transpose-tile.svg){.aig-svg}

The idea: one block owns one 32×32 tile.

1. read the input tile row-wise and **coalesced** into shared memory as `tile[threadIdx.y][threadIdx.x]`;
2. `__syncthreads()`;
3. read it back column-wise as `tile[threadIdx.x][threadIdx.y]` and write it row-wise and **coalesced** to the output tile (whose position is the input tile's coordinates swapped).

Both the global read and the global write now coalesce, and the "strided access" has moved into shared memory, where a stride only costs bank conflicts, far less than in global memory. Declare the tile `[32][33]` and even the column read's conflicts disappear (the reasoning is in [the memory hierarchy](../basics/memory.md#bank-冲突)).

## Several elements per thread {#每个线程处理多个元素}

Using 32×32 = 1024 threads per tile with one element each makes the thread startup and index arithmetic a large share of the cost, and a 1024-thread block schedules coarsely. The common choice is a 32×8 block with each thread looping over 4 elements in y. That is what NVIDIA's own blog post *An Efficient Matrix Transpose in CUDA C/C++* does.

## The complete code {#完整代码}

The program implements four versions and uses a **copy kernel** of the same shape as the performance ceiling: a transpose moves the same bytes as a copy and should ideally be just as fast.

```cuda title="transpose.cu"
// transpose.cu - matrix transpose: naive, then shared memory, then conflict-free, compared against a copy
// build: nvcc -O3 -arch=sm_75 transpose.cu -o transpose
#include "common.cuh"

constexpr int TILE = 32;
constexpr int ROWS_PER_ITER = 8;   // a 32 x 8 block, so each thread handles TILE / 8 = 4 elements

// the ceiling to compare against: the same access shape, only without transposing
__global__ void copy_tile(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += ROWS_PER_ITER)
    if (x < cols && y + j < rows) out[(y + j) * cols + x] = in[(y + j) * cols + x];
}

// naive: the read coalesces, the write does not
__global__ void transpose_naive(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += ROWS_PER_ITER)
    if (x < cols && y + j < rows) out[x * rows + (y + j)] = in[(y + j) * cols + x];
}

// staged through shared memory with a tile 32 wide: reading it by column is a 32-way bank conflict
// staged through shared memory with a tile 33 wide: one padding column removes the conflict
template <int PAD>
__global__ void transpose_smem(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
  __shared__ float tile[TILE][TILE + PAD];
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += ROWS_PER_ITER)
    if (x < cols && y + j < rows) tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * cols + x];
  __syncthreads();

  // where the output tile goes: the tile coordinates swapped
  x = blockIdx.y * TILE + threadIdx.x;   // the output's column is the input's row
  y = blockIdx.x * TILE + threadIdx.y;   // the output's row is the input's column
  for (int j = 0; j < TILE; j += ROWS_PER_ITER)
    if (x < rows && y + j < cols) out[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
}

int main() {
  const int rows = 4096 + 17, cols = 8192 + 5;   // deliberately not a multiple of 32, to exercise the bounds checks
  const size_t n = static_cast<size_t>(rows) * cols, bytes = n * sizeof(float);
  std::vector<float> h(n), ref(n), got(n);
  fill_random(h, 9);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) ref[static_cast<size_t>(c) * rows + r] = h[static_cast<size_t>(r) * cols + c];

  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemcpy(d_in, h.data(), bytes, cudaMemcpyHostToDevice));

  dim3 block(TILE, ROWS_PER_ITER);
  dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
  bool ok = true;

  auto bench = [&](const char* name, auto kernel, bool check) {
    CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    kernel<<<grid, block>>>(d_in, d_out, rows, cols);
    CUDA_CHECK_LAST();
    std::printf("%-22s ", name);
    if (check) {
      CUDA_CHECK(cudaMemcpy(got.data(), d_out, bytes, cudaMemcpyDeviceToHost));
      ok &= check_close(got.data(), ref.data(), n, 0.f, 0.f);
      std::printf("%-22s ", "");
    }
    float ms = time_ms([&] { kernel<<<grid, block>>>(d_in, d_out, rows, cols); });
    std::printf("%.3f ms, %.1f GB/s\n", ms, gbps(2.0 * bytes, ms));
  };

  bench("copy (upper bound)", copy_tile, false);
  bench("naive", transpose_naive, true);
  bench("smem [32][32]", transpose_smem<0>, true);
  bench("smem [32][33]", transpose_smem<1>, true);

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return ok ? 0 : 1;
}
```

What to expect: the naive version is clearly the slowest; the shared-memory version is much faster; removing the bank conflicts gains another step and comes close to the copy. If `[32][32]` and `[32][33]` are close on your GPU, check the shared-memory conflict count in Nsight Compute to confirm the conflicts really went away.

## Discussion {#讨论}

- **Boundaries**: both the read and the write phase need bounds checks, and the dimensions they check are swapped, which is the easiest thing to get wrong. The test above deliberately uses a size not divisible by 32.
- **Vectorization**: each thread can read and write `float2`/`float4` to cut the instruction count further, but the shared-memory layout has to change with it or new bank conflicts appear.
- **Batched transposes and permutes**: deep learning more often permutes a high-dimensional tensor, turning `[B, S, H, D]` into `[B, H, S, D]`. The idea is the same: keep the innermost dimension contiguous for both the read and the write, going through shared memory to change the layout when necessary. Often the better answer is **not to transpose at all**, and have the next kernel read the new layout directly (GEMM accepting an `A^T` input, say), or fuse the transpose into the surrounding kernels.
- **TMA on Hopper** moves whole tiles between global and shared memory and can apply some layout transformations on the way; see [Hopper](../advanced/async-hopper.md).

!!! interview "Answering in an interview"
    A transpose question is about coalescing: in the naive version one side is always non-contiguous; staging through shared memory makes both sides row-contiguous, and the bank conflict of reading shared memory by column is removed by padding to `[32][33]`; a 32×8 block with 4 elements per thread is the usual configuration; and the performance ceiling is a copy kernel of the same size. Finish with the engineering judgement: avoid transposing when you can, by fusing it into the surrounding kernels or having the consumer read the new layout directly.

## Exercises {#练习}

**1. Transpose a square matrix in place.** Transpose an N×N matrix in place with no extra global memory. Hint: only the tiles above the diagonal need handling, each block loads both the `(i, j)` and the `(j, i)` tile and writes them back swapped, and the diagonal tiles are handled separately.

??? success "Approach"
    - have the grid cover only the upper triangle (including the diagonal), for instance with a one-dimensional grid whose linear index maps to `(i, j)` with `i ≤ j`;
    - off-diagonal tiles: read `(i, j)` into `tileA` and `(j, i)` into `tileB`, synchronize, then write `tileB` transposed to `(i, j)` and `tileA` transposed to `(j, i)`;
    - diagonal tiles: read one tile, transpose it and write it back in place;
    - pad both tiles to `[32][33]`. Since each block has both tiles in shared memory before writing anything back, and different blocks own disjoint tiles, there is no race.

**2. An analysis question.** If the tile holds `double` (8 bytes), does the `[32][33]` padding still work?

??? success "Answer"
    Not quite for the same reason. 33 doubles per row is 66 words, so reading by column puts row k at word `66k`, bank `66k % 32 = 2k % 32`, while a 64-bit access is split into half-warp phases: each phase's 16 threads touch banks 0, 2, 4 … 30 (a double occupying two neighbouring banks), which lines up exactly with no conflict. So `[32][33]` still removes the conflict for doubles, for a different reason. The safer approach is to verify in Nsight Compute, or to compute the index with a swizzle.

## Summary {#小结}

- [x] A transpose's read and write cannot both be row-contiguous; stage through shared memory so both global sides coalesce.
- [x] Reading shared memory by column conflicts in banks; padding to `[32][33]` removes it.
- [x] Use a copy kernel of the same shape as the performance ceiling.
- [x] A 32×8 block with 4 elements per thread is the usual configuration for a transpose.
- [x] Avoid transposing when you can: fuse it into the surrounding kernels, or have the consumer read the new layout directly.
