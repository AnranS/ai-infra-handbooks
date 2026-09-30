from checker import check, check_close
from solution import attn_intensity, breakeven_batch, decode_floor_ms, ridge


def test_example():
    check_close(ridge(989, 3350), 295.2, rtol=1e-3, what="H100 BF16 的屋脊点")
    check_close(breakeven_batch(989, 3350), 295.2, rtol=1e-3, what="BF16 权重")
    check_close(breakeven_batch(989, 3350, 0.5), 73.8, rtol=1e-3, what="4 比特权重")
    check_close(attn_intensity(8), 8.0, rtol=1e-9, what="GQA 8 组")
    check_close(decode_floor_ms(16, 3350), 4.776, rtol=1e-3, what="16 GB 权重读一遍")


def test_generations():
    gens = [(125, 900), (312, 2039), (989, 3350), (2250, 8000)]      # V100、A100、H100、B200
    ridges = [ridge(t, b) for t, b in gens]
    check([round(r) for r in ridges], [139, 153, 295, 281], "各代 BF16 屋脊点")
    check(ridges[-1] > ridges[0] * 2, True, "屋脊点翻了一倍多")


def test_quantization_lowers_bar():
    bf16 = breakeven_batch(989, 3350, 2)
    fp8 = breakeven_batch(989, 3350, 1)
    int4 = breakeven_batch(989, 3350, 0.5)
    check([round(bf16), round(fp8), round(int4)], [295, 148, 74], "权重越小，门槛越低")


def test_low_precision_ridge():
    check_close(ridge(1979, 3350), 590.7, rtol=1e-3, what="H100 FP8 的屋脊点")
    check_close(breakeven_batch(1979, 3350, 1), 295.2, rtol=1e-3, what="FP8 算力配 FP8 权重，门槛回到 295")


def test_attention_never_compute_bound():
    r = ridge(989, 3350)
    for g in (1, 8, 128):
        check(attn_intensity(g) < r, True, f"每组 {g} 个查询头时注意力仍然带宽受限")
    check_close(attn_intensity(8, 1), 16.0, rtol=1e-9, what="KV 量化到 8 比特后强度翻倍")


def test_decode_floor():
    check_close(decode_floor_ms(140, 3350), 41.8, rtol=1e-2, what="70B 的 BF16 权重")
    check_close(decode_floor_ms(35, 3350), 10.4, rtol=1e-2, what="同一个模型量化到 4 比特")
