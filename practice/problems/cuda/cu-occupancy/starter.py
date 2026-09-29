import math


def occupancy(threads_per_block, regs_per_thread, smem_per_block, gpu):
    blocks = gpu["max_threads"] // threads_per_block      # 只考虑了线程数
    return blocks, blocks * threads_per_block / 32 / gpu["max_warps"], "threads"
