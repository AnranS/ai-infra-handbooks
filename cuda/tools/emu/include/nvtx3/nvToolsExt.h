// Emulator stand-in for NVTX ranges (no-ops).
#pragma once
inline int nvtxRangePushA(const char*) { return 0; }
inline int nvtxRangePop() { return 0; }
