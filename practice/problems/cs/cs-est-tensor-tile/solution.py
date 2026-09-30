def operand_bytes_per_fma(m, n, a_from_smem=True, b_from_smem=True):
    elems = (m if a_from_smem else 0) + (n if b_from_smem else 0)
    return 2 * elems / (m * n)


def smem_pressure(m, n, fma_per_cycle, smem_bw=128, **kw):
    need = fma_per_cycle * operand_bytes_per_fma(m, n, **kw)
    return need, need / smem_bw


def accum_bytes(m, n, dtype_bytes=4):
    return m * n * dtype_bytes


def fits_in_registers(m, n, threads, regs_per_thread=255, dtype_bytes=4):
    regs = accum_bytes(m, n, dtype_bytes) / threads / 4
    return regs, regs <= regs_per_thread
