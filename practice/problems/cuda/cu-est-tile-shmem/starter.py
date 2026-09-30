def shmem_bytes(block_m, block_n, block_k, stages, dtype_bytes=2):
    return (block_m * block_k + block_k * block_n) * dtype_bytes      # 忘了乘流水级数


def blocks_per_sm(block_m, block_n, block_k, stages, smem_per_sm=228 * 1024, dtype_bytes=2):
    return smem_per_sm // shmem_bytes(block_m, block_n, block_k, stages, dtype_bytes)


def accum_regs(block_m, block_n, threads):
    return block_m * block_n // threads


def is_feasible(block_m, block_n, block_k, stages, threads, smem_per_sm=228 * 1024, max_regs=255):
    return accum_regs(block_m, block_n, threads) <= max_regs          # 没检查共享内存
