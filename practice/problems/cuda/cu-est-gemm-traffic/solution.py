def global_bytes(M, N, K, BM, BN, in_bytes=2, out_bytes=2):
    a_reads = -(-N // BN)
    b_reads = -(-M // BM)
    return (M * K * a_reads + K * N * b_reads) * in_bytes + M * N * out_bytes


def tiled_intensity(M, N, K, BM, BN, in_bytes=2, out_bytes=2):
    return 2 * M * N * K / global_bytes(M, N, K, BM, BN, in_bytes, out_bytes)


def smem_bytes(BM, BN, BK, stages, in_bytes=2):
    return (BM * BK + BK * BN) * in_bytes * stages


def max_blocks_by_smem(smem_per_block, smem_per_sm=228 * 1024, reserved=1024):
    return smem_per_sm // (smem_per_block + reserved)
