from checker import check
from solution import block_bytes, blocks_for, max_concurrent, waste_tokens


def test_example():
    check(blocks_for(100, 16), 7, "不满一块也占一整块")
    check(waste_tokens(100, 16), 12, "最后一块浪费 12 个槽位")
    check(max_concurrent(1000, 2048, 16), 7, "每个请求要 128 块")
    check(block_bytes(16, 32, 8, 128, 2), 2097152, "一个块 2 MB")


def test_blocks_edges():
    check(blocks_for(0, 16), 0, "空请求")
    check(blocks_for(16, 16), 1, "正好一块")
    check(blocks_for(17, 16), 2, "多一个 token 就要两块")
    check(blocks_for(1, 16), 1, "一个 token 也要一整块")


def test_waste_edges():
    check(waste_tokens(16, 16), 0, "正好整除没有浪费")
    check(waste_tokens(32, 16), 0, "两整块")
    check(waste_tokens(1, 16), 15, "浪费 15 个")
    check(waste_tokens(0, 16), 0, "空请求")


def test_block_size_tradeoff():
    tokens = 1000
    check(waste_tokens(tokens, 1), 0, "token 级分页没有内部碎片")
    check(waste_tokens(tokens, 256) > waste_tokens(tokens, 16), True, "块越大浪费越多")


def test_concurrency():
    check(max_concurrent(100, 100, 16), 14, "每个请求 7 块")
    check(max_concurrent(10, 1000, 16), 0, "一个都放不下")
    check(max_concurrent(0, 100, 16), 0, "没有块")


def test_bytes_scaling():
    mha = block_bytes(16, 32, 32, 128)         # 32 个 KV 头（MHA）
    gqa = block_bytes(16, 32, 8, 128)          # 8 个 KV 头（GQA）
    check(mha // gqa, 4, "GQA 把 KV 缩小到四分之一")
    fp8 = block_bytes(16, 32, 8, 128, 1)
    check(gqa // fp8, 2, "KV 量化到 8 比特再减半")


def test_realistic():
    # 32 层、8 个 KV 头、头维度 128、BF16：每个 token 128 KB
    per_token = block_bytes(1, 32, 8, 128, 2)
    check(per_token, 131072, "每个 token 128 KB")
    total_blocks = int(40e9 // block_bytes(16, 32, 8, 128, 2))   # 40 GB 留给 KV
    check(max_concurrent(total_blocks, 4096, 16), 74, "40 GB 能服务 74 个 4K 上下文的请求")
