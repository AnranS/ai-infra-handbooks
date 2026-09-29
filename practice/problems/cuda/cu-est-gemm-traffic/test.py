from checker import check, check_close
from solution import global_bytes, max_blocks_by_smem, smem_bytes, tiled_intensity

S = 4096


def test_example():
    check(global_bytes(S, S, S, 128, 128), (S * S * 32 * 2) * 2 + S * S * 2, "4096³、128 × 128")
    check_close(tiled_intensity(S, S, S, 128, 128), 2 * S ** 3 / ((S * S * 32 * 2) * 2 + S * S * 2), what="算术强度")
    check(smem_bytes(128, 128, 64, 3), 98304, "128 × 128 × 64、3 级流水线")
    check(max_blocks_by_smem(98304), 2, "每个 SM 放 2 个块")


def test_tile_size_effect():
    small = tiled_intensity(S, S, S, 64, 64)
    mid = tiled_intensity(S, S, S, 128, 128)
    big = tiled_intensity(S, S, S, 128, 256)
    assert small < mid < big, "tile 越大，算术强度越高"
    check_close(mid / small, 2.0, rtol=0.02, what="边长翻倍，算术强度约翻倍")
    naive = tiled_intensity(S, S, S, 1, 1)
    assert naive < 1.1, f"不分块时每个输出都把一行一列读一遍，算术强度约 1，算出来 {naive:.2f}"


def test_ragged_and_dtypes():
    check(global_bytes(1000, 1000, 512, 128, 128), (1000 * 512 * 8 + 512 * 1000 * 8) * 2 + 1000 * 1000 * 2, "向上取整")
    check(global_bytes(256, 256, 256, 128, 128, in_bytes=1, out_bytes=2), (256 * 256 * 2) * 2 + 256 * 256 * 2, "FP8 输入")


def test_smem_budget():
    check(smem_bytes(128, 256, 64, 4), (128 * 64 + 64 * 256) * 2 * 4, "128 × 256 × 64、4 级")
    check(max_blocks_by_smem(smem_bytes(128, 256, 64, 4)), 1, "大 tile 只能放 1 个块")
    check(max_blocks_by_smem(smem_bytes(64, 64, 32, 2)), 13, "小 tile 可以放很多块")
    check(max_blocks_by_smem(48 * 1024, smem_per_sm=164 * 1024), 3, "A100：每个 SM 164 KiB")
