// Emulator cp.async primitives: copies complete immediately, so commit/wait are no-ops.
// This checks buffer/index logic of multi-stage pipelines, not their timing.
#pragma once
#include <cstring>
#include <cstddef>
inline void __pipeline_memcpy_async(void* dst, const void* src, size_t size, size_t zfill = 0) {
  std::memcpy(dst, src, size - zfill);
  if (zfill) std::memset(static_cast<char*>(dst) + size - zfill, 0, zfill);
}
inline void __pipeline_commit() {}
inline void __pipeline_wait_prior(size_t) {}
