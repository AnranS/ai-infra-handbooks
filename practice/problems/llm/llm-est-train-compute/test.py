from checker import check_close
from solution import flops_per_token, gpu_hours, hfu, mfu, train_flops


def test_example():
    check_close(train_flops(405e9, 15.6e12), 3.7908e25, rtol=1e-9, what="405B × 15.6T token")
    check_close(train_flops(405e9, 15.6e12), 3.8e25, rtol=0.02, what="和公开的 3.8e25 FLOPs 对比")
    check_close(gpu_hours(train_flops(37e9, 14.8e12), 989, 0.35), 3.2856e24 / (989e12 * 0.35) / 3600, rtol=1e-9,
                what="37B 激活参数 × 14.8T token 的 GPU 时")


def test_mental_math():
    seconds = gpu_hours(train_flops(7e9, 2e12), 989, 0.4) * 3600 / 1024
    check_close(seconds / 86400, 2.4, rtol=0.02, what="7B × 2T token，1024 张 H100、40% MFU 要几天")


def test_attention_term():
    # Llama-3-8B：32 层，32 个头 × 128 维
    base = 6 * 8.03e9
    check_close(flops_per_token(8.03e9, 32, 4096, 8192), base + 12 * 32 * 4096 * 8192, rtol=1e-12, what="8K 序列")
    share8k = 12 * 32 * 4096 * 8192 / flops_per_token(8.03e9, 32, 4096, 8192)
    share32k = 12 * 32 * 4096 * 32768 / flops_per_token(8.03e9, 32, 4096, 32768)
    assert 0.18 < share8k < 0.24, f"8K 序列时注意力项约占每 token FLOPs 的 21%，算出来是 {share8k:.1%}"
    assert share32k > 0.5, f"32K 序列时注意力项超过一半，算出来是 {share32k:.1%}"


def test_mfu():
    # 8 张 H100 训练 Llama-3-8B，4K 序列，实测 3.2 万 tokens/s
    got = mfu(32000, 8, 989, 8.03e9, 32, 4096, 4096)
    check_close(got, 32000 * (6 * 8.03e9 + 12 * 32 * 4096 * 4096) / (8 * 989e12), rtol=1e-12, what="MFU")
    assert 0.2 < got < 0.3, f"这个吞吐对应的 MFU 应该在 20%～30% 之间，算出来是 {got:.1%}"


def test_hfu_ge_mfu():
    args = (32000, 8, 989, 8.03e9, 32, 4096, 4096)
    check_close(hfu(*args) / mfu(*args), 4 / 3, rtol=1e-12, what="全量重计算时 HFU / MFU = 8/6")
    check_close(hfu(1000, 1, 100, 1e9, 10, 1024, 2048),
                1000 * (8e9 + 16 * 10 * 1024 * 2048) / 100e12, rtol=1e-12, what="HFU 的注意力项是 16·L·d·s")
