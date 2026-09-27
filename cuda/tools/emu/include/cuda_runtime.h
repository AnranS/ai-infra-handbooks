// A tiny CUDA-on-CPU emulator used only to check the *logic* of the handbook's kernels on a
// machine without a GPU. Every CUDA thread is a ucontext coroutine; the blocks of a grid run one
// after another. __syncthreads() and warp-level primitives are real barriers, so indexing bugs,
// wrong shuffle partners and barriers inside divergent code (deadlock) are detected.
// It says nothing about performance or about races (execution is sequential).
#pragma once
#include <ucontext.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

using std::max;
using std::min;

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __noinline__
#define __restrict__ __restrict
#define __launch_bounds__(...)
#define __align__(n) __attribute__((aligned(n)))
#define __constant__

struct dim3 {
  unsigned x, y, z;
  constexpr dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};
using uint3 = dim3;
// No over-alignment on purpose: kernels reinterpret register arrays as float4, which is fine on
// the GPU but would let the host compiler emit aligned SSE moves on misaligned stack data.
struct float4 { float x, y, z, w; };
struct float2 { float x, y; };
struct int4 { int x, y, z, w; };
struct uint4 { unsigned x, y, z, w; };
inline float4 make_float4(float x, float y, float z, float w) { return {x, y, z, w}; }
inline float2 make_float2(float x, float y) { return {x, y}; }

enum cudaError_t { cudaSuccess = 0, cudaErrorInvalidConfiguration = 9, cudaErrorInvalidValue = 1 };
enum cudaMemcpyKind { cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, cudaMemcpyDefault };
enum cudaFuncAttribute { cudaFuncAttributeMaxDynamicSharedMemorySize, cudaFuncAttributePreferredSharedMemoryCarveout };
enum cudaDeviceAttr { cudaDevAttrMultiProcessorCount, cudaDevAttrMemoryClockRate, cudaDevAttrGlobalMemoryBusWidth, cudaDevAttrMaxSharedMemoryPerBlockOptin, cudaDevAttrConcurrentManagedAccess };
using cudaStream_t = void*;
struct EmuEvent { std::chrono::steady_clock::time_point t; };
using cudaEvent_t = EmuEvent*;

struct cudaDeviceProp {
  char name[256] = "CPU emulator";
  int major = 8, minor = 0, multiProcessorCount = 2, l2CacheSize = 4 << 20, regsPerMultiprocessor = 65536,
      regsPerBlock = 65536, maxThreadsPerMultiProcessor = 2048, maxThreadsPerBlock = 1024,
      maxBlocksPerMultiProcessor = 32, warpSize = 32;
  size_t totalGlobalMem = size_t(8) << 30, sharedMemPerMultiprocessor = 164 * 1024, sharedMemPerBlock = 48 * 1024,
         sharedMemPerBlockOptin = 163 * 1024;
};

namespace emu {
enum State { RUN, BLOCK_BAR, WARP_BAR, DONE };
struct Fiber {
  ucontext_t ctx;
  std::vector<char> stack;
  dim3 tid;
  unsigned lin = 0;
  State st = RUN;
};
inline ucontext_t sched_ctx;
inline std::vector<Fiber> fibers;
inline Fiber* cur = nullptr;
inline std::function<void()> body;
inline dim3 g_threadIdx, g_blockIdx, g_blockDim, g_gridDim;
inline std::vector<std::array<unsigned char, 16>> slots;
inline std::vector<unsigned char> dyn_smem_buf;
inline cudaError_t last_error = cudaSuccess;
inline void* dyn_smem() { return dyn_smem_buf.data(); }

inline void trampoline() {
  body();
  cur->st = DONE;
  swapcontext(&cur->ctx, &sched_ctx);
}
inline void yield_with(State s) {
  cur->st = s;
  swapcontext(&cur->ctx, &sched_ctx);
  g_threadIdx = cur->tid;
}

inline void run_block(unsigned nthreads) {
  constexpr size_t kStack = 256 * 1024;
  if (fibers.size() < nthreads) fibers.resize(nthreads);
  slots.assign(nthreads + 32, {});
  for (unsigned i = 0; i < nthreads; ++i) {
    Fiber& f = fibers[i];
    if (f.stack.size() != kStack) f.stack.resize(kStack);
    f.lin = i;
    f.tid = dim3(i % g_blockDim.x, (i / g_blockDim.x) % g_blockDim.y, i / (g_blockDim.x * g_blockDim.y));
    f.st = RUN;
    getcontext(&f.ctx);
    f.ctx.uc_stack.ss_sp = f.stack.data();
    f.ctx.uc_stack.ss_size = kStack;
    f.ctx.uc_link = nullptr;
    makecontext(&f.ctx, trampoline, 0);
  }
  for (;;) {
    bool progressed = false;
    for (unsigned i = 0; i < nthreads; ++i) {
      Fiber& f = fibers[i];
      if (f.st != RUN) continue;
      cur = &f;
      g_threadIdx = f.tid;
      swapcontext(&sched_ctx, &f.ctx);
      progressed = true;
    }
    unsigned done = 0, at_block = 0;
    for (unsigned i = 0; i < nthreads; ++i) {
      done += fibers[i].st == DONE;
      at_block += fibers[i].st == BLOCK_BAR;
    }
    if (done == nthreads) return;
    bool released = false;
    if (at_block && at_block + done == nthreads) {
      for (unsigned i = 0; i < nthreads; ++i)
        if (fibers[i].st == BLOCK_BAR) fibers[i].st = RUN;
      released = true;
    }
    for (unsigned w = 0; w * 32 < nthreads; ++w) {
      unsigned lo = w * 32, hi = std::min(nthreads, lo + 32), waiting = 0, alive = 0;
      for (unsigned i = lo; i < hi; ++i) {
        alive += fibers[i].st != DONE;
        waiting += fibers[i].st == WARP_BAR;
      }
      if (waiting && waiting == alive) {
        for (unsigned i = lo; i < hi; ++i)
          if (fibers[i].st == WARP_BAR) fibers[i].st = RUN;
        released = true;
      }
    }
    if (!progressed && !released) {
      std::fprintf(stderr, "EMU DEADLOCK in block (%u,%u,%u): threads wait at barriers that others never reach\n",
                   g_blockIdx.x, g_blockIdx.y, g_blockIdx.z);
      std::exit(3);
    }
  }
}

inline bool capturing = false;
inline std::vector<std::function<void()>> captured;
inline void launch_now(dim3 grid, dim3 block, size_t smem, std::function<void()> fn);
inline void launch(dim3 grid, dim3 block, size_t smem, std::function<void()> fn) {
  if (capturing) {
    captured.push_back([=] { launch_now(grid, block, smem, fn); });
    return;
  }
  launch_now(grid, block, smem, std::move(fn));
}
inline void launch_now(dim3 grid, dim3 block, size_t smem, std::function<void()> fn) {
  const unsigned nthreads = block.x * block.y * block.z;
  if (nthreads == 0 || nthreads > 1024 || block.z > 64 || grid.y > 65535 || grid.z > 65535 || smem > 227 * 1024) {
    last_error = cudaErrorInvalidConfiguration;
    return;
  }
  body = std::move(fn);
  dyn_smem_buf.assign(smem + 16, 0);
  g_gridDim = grid;
  g_blockDim = block;
  for (unsigned z = 0; z < grid.z; ++z)
    for (unsigned y = 0; y < grid.y; ++y)
      for (unsigned x = 0; x < grid.x; ++x) {
        g_blockIdx = dim3(x, y, z);
        run_block(nthreads);
      }
}
inline void launch(dim3 grid, dim3 block, std::function<void()> fn) { launch(grid, block, 0, std::move(fn)); }
inline void launch(dim3 grid, dim3 block, size_t smem, cudaStream_t, std::function<void()> fn) {
  launch(grid, block, smem, std::move(fn));
}

inline unsigned lane() { return cur->lin % 32; }
template <class T>
T exchange(T v, int src_lane) {
  std::memcpy(slots[cur->lin].data(), &v, sizeof(T));
  yield_with(WARP_BAR);
  T r;
  std::memcpy(&r, slots[cur->lin - lane() + src_lane].data(), sizeof(T));
  yield_with(WARP_BAR);
  return r;
}
}  // namespace emu

#define threadIdx (::emu::g_threadIdx)
#define blockIdx (::emu::g_blockIdx)
#define blockDim (::emu::g_blockDim)
#define gridDim (::emu::g_gridDim)
constexpr int warpSize = 32;

inline void __syncthreads() { emu::yield_with(emu::BLOCK_BAR); }
inline void __syncwarp(unsigned = 0xffffffffu) { emu::yield_with(emu::WARP_BAR); }
inline void __threadfence() {}
inline void __threadfence_block() {}
inline unsigned __activemask() { return 0xffffffffu; }

template <class T>
T __shfl_sync(unsigned, T v, int src, int width = 32) {
  int l = emu::lane();
  return emu::exchange(v, (l / width) * width + (src % width));
}
template <class T>
T __shfl_down_sync(unsigned, T v, unsigned d, int width = 32) {
  int l = emu::lane(), seg = (l / width) * width;
  int s = l + int(d);
  return emu::exchange(v, s < seg + width ? s : l);
}
template <class T>
T __shfl_up_sync(unsigned, T v, unsigned d, int width = 32) {
  int l = emu::lane(), seg = (l / width) * width;
  int s = l - int(d);
  return emu::exchange(v, s >= seg ? s : l);
}
template <class T>
T __shfl_xor_sync(unsigned, T v, int m, int width = 32) {
  int l = emu::lane(), seg = (l / width) * width;
  int s = l ^ m;
  return emu::exchange(v, (s >= seg && s < seg + width) ? s : l);
}
inline unsigned __ballot_sync(unsigned, int pred) {
  unsigned bits = 0;
  unsigned base = emu::cur->lin - emu::lane();
  std::memcpy(emu::slots[emu::cur->lin].data(), &pred, sizeof(int));
  emu::yield_with(emu::WARP_BAR);
  for (unsigned l = 0; l < 32; ++l) {
    if (base + l >= emu::fibers.size() || base + l >= emu::g_blockDim.x * emu::g_blockDim.y * emu::g_blockDim.z) break;
    int p;
    std::memcpy(&p, emu::slots[base + l].data(), sizeof(int));
    if (p) bits |= 1u << l;
  }
  emu::yield_with(emu::WARP_BAR);
  return bits;
}
inline int __any_sync(unsigned m, int p) { return __ballot_sync(m, p) != 0; }
inline int __all_sync(unsigned m, int p) {
  unsigned n = std::min<unsigned>(32, emu::g_blockDim.x * emu::g_blockDim.y * emu::g_blockDim.z - (emu::cur->lin - emu::lane()));
  unsigned full = n == 32 ? 0xffffffffu : ((1u << n) - 1);
  return __ballot_sync(m, p) == full;
}
inline int __popc(unsigned x) { return __builtin_popcount(x); }
inline int __clz(int x) { return x ? __builtin_clz(unsigned(x)) : 32; }
inline int __ffs(int x) { return __builtin_ffs(x); }

template <class T> T atomicAdd(T* a, T v) { T o = *a; *a = o + v; return o; }
template <class T> T atomicSub(T* a, T v) { T o = *a; *a = o - v; return o; }
template <class T> T atomicMax(T* a, T v) { T o = *a; *a = std::max(o, v); return o; }
template <class T> T atomicMin(T* a, T v) { T o = *a; *a = std::min(o, v); return o; }
template <class T> T atomicExch(T* a, T v) { T o = *a; *a = v; return o; }
template <class T> T atomicCAS(T* a, T cmp, T v) { T o = *a; if (o == cmp) *a = v; return o; }

inline float __int_as_float(int i) { float f; std::memcpy(&f, &i, 4); return f; }
inline int __float_as_int(float f) { int i; std::memcpy(&i, &f, 4); return i; }
inline unsigned __float_as_uint(float f) { unsigned i; std::memcpy(&i, &f, 4); return i; }
inline float __uint_as_float(unsigned i) { float f; std::memcpy(&f, &i, 4); return f; }
inline float __expf(float x) { return std::exp(x); }
inline float __logf(float x) { return std::log(x); }
inline float __fdividef(float a, float b) { return a / b; }
inline float rsqrtf(float x) { return 1.f / std::sqrt(x); }
inline float __frcp_rn(float x) { return 1.f / x; }
template <class T> T __ldg(const T* p) { return *p; }

// ---------------- runtime API shims ----------------
inline const char* cudaGetErrorString(cudaError_t e) { return e == cudaSuccess ? "no error" : "emulated error"; }
inline const char* cudaGetErrorName(cudaError_t e) { return e == cudaSuccess ? "cudaSuccess" : "cudaErrorEmulated"; }
inline cudaError_t cudaGetLastError() { cudaError_t e = emu::last_error; emu::last_error = cudaSuccess; return e; }
inline cudaError_t cudaPeekAtLastError() { return emu::last_error; }
template <class T> cudaError_t cudaMalloc(T** p, size_t n) { *p = static_cast<T*>(std::aligned_alloc(256, (n + 255) / 256 * 256)); return cudaSuccess; }
inline cudaError_t cudaMallocHost(void** p, size_t n) { *p = std::aligned_alloc(256, (n + 255) / 256 * 256); return cudaSuccess; }
template <class T> cudaError_t cudaMallocHost(T** p, size_t n) { return cudaMallocHost(reinterpret_cast<void**>(p), n); }
inline cudaError_t cudaFree(void* p) { std::free(p); return cudaSuccess; }
inline cudaError_t cudaFreeHost(void* p) { std::free(p); return cudaSuccess; }
inline cudaError_t cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind) { std::memcpy(d, s, n); return cudaSuccess; }
inline cudaError_t cudaMemcpyAsync(void* d, const void* s, size_t n, cudaMemcpyKind, cudaStream_t = nullptr) { std::memcpy(d, s, n); return cudaSuccess; }
inline cudaError_t cudaMemset(void* d, int v, size_t n) { std::memset(d, v, n); return cudaSuccess; }
inline cudaError_t cudaMemsetAsync(void* d, int v, size_t n, cudaStream_t = nullptr) { std::memset(d, v, n); return cudaSuccess; }
template <class T> cudaError_t cudaMemcpyToSymbol(T& sym, const void* src, size_t n, size_t off = 0, cudaMemcpyKind = cudaMemcpyHostToDevice) {
  std::memcpy(reinterpret_cast<char*>(&sym) + off, src, n); return cudaSuccess;
}
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaGetDevice(int* d) { *d = 0; return cudaSuccess; }
inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
inline cudaError_t cudaGetDeviceCount(int* n) { *n = 1; return cudaSuccess; }
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int) { *p = cudaDeviceProp{}; return cudaSuccess; }
inline cudaError_t cudaDeviceGetAttribute(int* v, cudaDeviceAttr a, int) {
  switch (a) {
    case cudaDevAttrMultiProcessorCount: *v = 2; break;
    case cudaDevAttrMemoryClockRate: *v = 1215000; break;
    case cudaDevAttrGlobalMemoryBusWidth: *v = 5120; break;
    case cudaDevAttrMaxSharedMemoryPerBlockOptin: *v = 163 * 1024; break;
    case cudaDevAttrConcurrentManagedAccess: *v = 1; break;
  }
  return cudaSuccess;
}
template <class F> cudaError_t cudaFuncSetAttribute(F, cudaFuncAttribute, int) { return cudaSuccess; }
template <class F> cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* n, F, int block, size_t smem) {
  int by_threads = 2048 / block, by_smem = smem ? int((164 * 1024) / smem) : 32;
  *n = std::min({by_threads, 32, by_smem});
  return cudaSuccess;
}
template <class F> cudaError_t cudaOccupancyMaxPotentialBlockSize(int* min_grid, int* block, F, size_t = 0, int = 0) {
  *min_grid = 4; *block = 1024; return cudaSuccess;
}
inline cudaError_t cudaEventCreate(cudaEvent_t* e) { *e = new EmuEvent{}; return cudaSuccess; }
inline cudaError_t cudaEventDestroy(cudaEvent_t e) { delete e; return cudaSuccess; }
inline cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t = nullptr) { e->t = std::chrono::steady_clock::now(); return cudaSuccess; }
inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t a, cudaEvent_t b) {
  *ms = std::chrono::duration<float, std::milli>(b->t - a->t).count(); return cudaSuccess;
}
inline cudaError_t cudaStreamCreate(cudaStream_t* s) { *s = nullptr; return cudaSuccess; }
inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }

// ---- streams, events, graphs, managed memory ----
#define CUDART_VERSION 13000
enum { cudaStreamDefault = 0, cudaStreamNonBlocking = 1 };
enum { cudaEventDefault = 0, cudaEventDisableTiming = 2 };
enum cudaStreamCaptureMode { cudaStreamCaptureModeGlobal, cudaStreamCaptureModeThreadLocal, cudaStreamCaptureModeRelaxed };
enum { cudaMemAttachGlobal = 1 };
constexpr int cudaCpuDeviceId = -1;
enum cudaMemLocationType { cudaMemLocationTypeInvalid, cudaMemLocationTypeDevice, cudaMemLocationTypeHost };
struct cudaMemLocation { cudaMemLocationType type; int id; };
struct EmuGraph { std::vector<std::function<void()>> ops; };
using cudaGraph_t = EmuGraph*;
using cudaGraphExec_t = EmuGraph*;
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned) { *s = nullptr; return cudaSuccess; }
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned) { *e = new EmuEvent{}; return cudaSuccess; }
inline cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned = 0) { return cudaSuccess; }
inline cudaError_t cudaStreamBeginCapture(cudaStream_t, cudaStreamCaptureMode) {
  emu::capturing = true; emu::captured.clear(); return cudaSuccess;
}
inline cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t* g) {
  emu::capturing = false; *g = new EmuGraph{emu::captured}; emu::captured.clear(); return cudaSuccess;
}
inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t* e, cudaGraph_t g, unsigned long long = 0) { *e = new EmuGraph{g->ops}; return cudaSuccess; }
inline cudaError_t cudaGraphLaunch(cudaGraphExec_t e, cudaStream_t) { for (auto& op : e->ops) op(); return cudaSuccess; }
inline cudaError_t cudaGraphExecDestroy(cudaGraphExec_t e) { delete e; return cudaSuccess; }
inline cudaError_t cudaGraphDestroy(cudaGraph_t g) { delete g; return cudaSuccess; }
template <class T> cudaError_t cudaMallocManaged(T** p, size_t n, unsigned = cudaMemAttachGlobal) { return cudaMalloc(p, n); }
inline cudaError_t cudaMemPrefetchAsync(const void*, size_t, cudaMemLocation, unsigned, cudaStream_t = nullptr) { return cudaSuccess; }

// Timing loops are pointless in the emulator: run the callable once.
template <class F> float emu_time_ms(F&& fn, int = 0, int = 0) { fn(); return 1.f; }
