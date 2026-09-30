def shmem_bytes(block_m, block_n, block_k, stages, dtype_bytes=2):
    per_stage = (block_m * block_k + block_k * block_n) * dtype_bytes
    return per_stage * stages


def blocks_per_sm(block_m, block_n, block_k, stages, smem_per_sm=228 * 1024, dtype_bytes=2):
    need = shmem_bytes(block_m, block_n, block_k, stages, dtype_bytes)
    return smem_per_sm // need if need else 0


def accum_regs(block_m, block_n, threads):
    return block_m * block_n // threads if threads else 0


def is_feasible(block_m, block_n, block_k, stages, threads, smem_per_sm=228 * 1024, max_regs=255):
    if blocks_per_sm(block_m, block_n, block_k, stages, smem_per_sm) < 1:
        return False
    return accum_regs(block_m, block_n, threads) <= max_regs
