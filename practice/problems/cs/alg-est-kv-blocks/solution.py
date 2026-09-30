def blocks_for(tokens, block_size):
    if tokens <= 0 or block_size <= 0:
        return 0
    return (tokens + block_size - 1) // block_size


def waste_tokens(tokens, block_size):
    if tokens <= 0 or block_size <= 0:
        return 0
    return blocks_for(tokens, block_size) * block_size - tokens


def max_concurrent(total_blocks, tokens_each, block_size):
    need = blocks_for(tokens_each, block_size)
    return total_blocks // need if need else 0


def block_bytes(block_size, n_layers, n_kv_heads, head_dim, dtype_bytes=2):
    return block_size * n_layers * n_kv_heads * head_dim * dtype_bytes * 2   # K 和 V
