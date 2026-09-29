def waves(M, N, BM, BN, n_sms, blocks_per_sm=1):
    tiles = -(-M // BM) * -(-N // BN)
    slots = n_sms * blocks_per_sm
    n_waves = -(-tiles // slots)
    return tiles, n_waves, tiles / (n_waves * slots)


def best_tile(M, N, candidates, n_sms, blocks_per_sm=1):
    best, key = None, None
    for bm, bn in candidates:
        k = (waves(M, N, bm, bn, n_sms, blocks_per_sm)[2], bm * bn)
        if key is None or k > key:
            best, key = (bm, bn), k
    return best
