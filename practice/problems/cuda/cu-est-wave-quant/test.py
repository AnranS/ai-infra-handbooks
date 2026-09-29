from checker import check, check_close
from solution import best_tile, waves


def test_example():
    tiles, n, eff = waves(4096, 4096, 128, 128, 132)
    check((tiles, n), (1024, 8), "4096² / 128²")
    check_close(eff, 1024 / 1056, what="利用率")
    tiles, n, eff = waves(1536, 1536, 128, 128, 132)
    check((tiles, n), (144, 2), "1536² / 128²")
    check_close(eff, 144 / 264, what="第二波只有 12 个块")
    check(best_tile(1536, 1536, [(128, 128), (128, 256), (64, 128)], 132), (64, 128), "1536² 选 64 × 128")


def test_ragged_edges():
    check(waves(1000, 1000, 128, 128, 108)[0], 64, "不能整除时向上取整（A100 有 108 个 SM）")
    check(waves(1040, 4096, 128, 128, 132)[:2], (9 * 32, 3), "M 从 1024 变到 1040：多出一整行 tile")
    check_close(waves(1024, 4096, 128, 128, 132)[2], 256 / 264, what="M=1024 时两波几乎打满")


def test_blocks_per_sm():
    tiles, n, eff = waves(4096, 4096, 128, 128, 132, blocks_per_sm=2)
    check((tiles, n), (1024, 4), "每个 SM 同时跑 2 个块")
    check_close(eff, 1024 / 1056, what="利用率")


def test_tie_break():
    check(best_tile(4096, 4096, [(64, 64), (128, 128)], 128), (128, 128), "利用率相同时选面积大的")
    check(best_tile(4096, 4096, [(128, 64), (64, 128)], 128), (128, 64), "完全相同时选靠前的")
