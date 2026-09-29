import math


def occupancy(threads_per_block, regs_per_thread, smem_per_block, gpu):
    if threads_per_block <= 0 or threads_per_block > 1024:
        return 0, 0.0, "threads"
    warps = math.ceil(threads_per_block / 32)
    limits = {"threads": gpu["max_threads"] // threads_per_block, "blocks": gpu["max_blocks"]}
    if regs_per_thread:
        unit = gpu["reg_alloc_unit"]
        per_warp = math.ceil(regs_per_thread * 32 / unit) * unit
        limits["registers"] = gpu["regs"] // (per_warp * warps)
    else:
        limits["registers"] = math.inf
    unit = gpu["smem_alloc_unit"]
    per_block = math.ceil((smem_per_block + gpu["smem_per_block_reserved"]) / unit) * unit
    limits["shared_memory"] = gpu["smem"] // per_block
    limiter = min(limits, key=lambda k: limits[k])       # dict 保持插入顺序：并列时取靠前的
    blocks = int(limits[limiter])
    if blocks == 0:
        return 0, 0.0, limiter
    return blocks, blocks * warps / gpu["max_warps"], limiter
