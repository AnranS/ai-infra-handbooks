from checker import check, check_close
from solution import BW, decode_kernels, eager_time_us, fused_step_us, step_time_us

Q06 = dict(hidden=1024, inter=3072, layers=28, heads=16, kv_heads=8, head_dim=128, vocab=151936)
Q8 = dict(hidden=4096, inter=12288, layers=36, heads=32, kv_heads=8, head_dim=128, vocab=151936)


def test_example():
    k = decode_kernels(**Q06)
    check(len(k), 283, "Qwen3-0.6B 一步的 kernel 数")
    t_data, total, share = step_time_us(k, 2.0)
    check_close(t_data, 460.5635539947322, rtol=1e-9, what="读写数据的时间（µs）")
    check_close(total, 1026.5635539947323, rtol=1e-9, what="g = 2 µs 时一步的时间")
    check_close(share, 0.5513540762259561, rtol=1e-9, what="空隙的占比")


def test_kernel_list():
    k = decode_kernels(**Q06)
    check([n for n, _ in k[:10]], ["add_rmsnorm", "qkv_proj", "qk_norm_rope", "kv_cache_write", "attention", "o_proj",
                                    "add_rmsnorm", "gate_up_proj", "silu_mul", "down_proj"], "每层 kernel 的顺序")
    check([n for n, _ in k[-3:]], ["final_norm", "lm_head", "sample"], "最后三个 kernel")
    check(dict(k[:10])["qkv_proj"], 1024 * (2048 + 2048) * 2, "qkv_proj 的字节数")
    check(dict(k[:10])["attention"], 1024 * 2 * 1024 * 2, "attention 读的 KV（batch 1、上下文 1024）")
    check(k[-1], ("sample", 151936 * 4), "采样读 fp32 的 logits")
    big = decode_kernels(**Q06, batch=64, context=4096)
    check(dict(big[:10])["attention"], 64 * 4096 * 2 * 1024 * 2, "batch 64、上下文 4096 的注意力")
    check(dict(big[:10])["add_rmsnorm"], 4 * 64 * 1024 * 2, "逐元素 kernel 的字节数随 batch 增长")
    check(len(decode_kernels(**Q8)), 363, "Qwen3-8B 的 kernel 数")


def test_ratios():
    t_data, total, share = step_time_us(decode_kernels(**Q8), 2.0)
    check_close(t_data, 5370.926030553117, rtol=1e-9, what="Qwen3-8B 读写数据的时间")
    check_close(share, 0.11907639954328539, rtol=1e-9, what="Qwen3-8B 的空隙占比")
    _, _, share64 = step_time_us(decode_kernels(**Q06, batch=64, context=4096), 2.0)
    check_close(share64, 0.048839926552454865, rtol=1e-9, what="batch 64、上下文 4096 时的空隙占比")
    _, total_bw, _ = step_time_us(decode_kernels(**Q06), 0.0, bw=1e12)
    check_close(total_bw, sum(b for _, b in decode_kernels(**Q06)) / 1e12 * 1e6, what="换一个带宽、没有空隙")


def test_eager_and_fusion():
    k = decode_kernels(**Q06)
    check_close(eager_time_us(k), 283 * 5.0, what="0.6B 不用 CUDA Graph：CPU 发射跟不上")
    k8 = decode_kernels(**Q8)
    check_close(eager_time_us(k8), 5370.926030553117 + 363 * 0.5, rtol=1e-9, what="8B：GPU 这一侧更慢")
    check_close(eager_time_us(k, cpu_launch_us=1.0, gap_us=1.0), 460.5635539947322 + 283, rtol=1e-9,
                what="CPU 很快时取 GPU 一侧")
    check_close(fused_step_us(k, 28, 10, 6, 2.0), 802.5635539947323, rtol=1e-9, what="每层 10 个融合成 6 个")
    check_close(fused_step_us(k, 28, 10, 10, 1.0), step_time_us(k, 1.0)[1], rtol=1e-12, what="不融合时不变")
