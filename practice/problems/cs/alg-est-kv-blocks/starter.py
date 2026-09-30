def blocks_for(tokens, block_size):
    return tokens // block_size                # 整除：不满一块的被忽略了


def waste_tokens(tokens, block_size):
    return block_size - tokens % block_size    # 正好整除时会算成一整块


def max_concurrent(total_blocks, tokens_each, block_size):
    return total_blocks // blocks_for(tokens_each, block_size)


def block_bytes(block_size, n_layers, n_kv_heads, head_dim, dtype_bytes=2):
    return block_size * n_layers * n_kv_heads * head_dim * dtype_bytes   # 漏了 K 和 V 两份
