def operand_bytes_per_fma(m, n, a_from_smem=True, b_from_smem=True):
    return 2 * (m + n) / (m * n)                   # 忽略了操作数可能不从共享内存读


def smem_pressure(m, n, fma_per_cycle, smem_bw=128, **kw):
    need = fma_per_cycle * operand_bytes_per_fma(m, n, **kw)
    return need, need / smem_bw


def accum_bytes(m, n, dtype_bytes=4):
    return m * n * dtype_bytes


def fits_in_registers(m, n, threads, regs_per_thread=255, dtype_bytes=4):
    regs = accum_bytes(m, n, dtype_bytes) / threads    # 算的是字节数，不是 32 位寄存器个数
    return regs, regs <= regs_per_thread
