from checker import check_close, raises
from solution import activation_bytes, model_state_bytes, per_gpu_gib


def test_example():
    check_close(model_state_bytes(7e9), 112e9, what="7B 模型的模型状态")
    check_close(model_state_bytes(7e9, 3, 8) / 2 ** 30, 112e9 / 8 / 2 ** 30, what="ZeRO-3、8 卡")


def test_zero_stages():
    psi, n = 70e9, 64
    check_close(model_state_bytes(psi, 1, n), 4 * psi + 12 * psi / n, what="ZeRO-1")
    check_close(model_state_bytes(psi, 2, n), 2 * psi + 14 * psi / n, what="ZeRO-2")
    check_close(model_state_bytes(psi, 3, n), 17.5e9, what="ZeRO-3：口算题")
    for stage in (1, 2, 3):
        check_close(model_state_bytes(psi, stage, 1), 16 * psi, what=f"dp=1 时 ZeRO-{stage} 不省显存")
    with raises(ValueError, "ZeRO-4"):
        model_state_bytes(psi, 4, n)


def test_activation():
    s, b, h, a, L = 4096, 1, 4096, 32, 32
    check_close(activation_bytes(s, b, h, a, L, flash=False), s * b * h * (34 + 5 * a * s / h) * L, what="不用 FlashAttention")
    check_close(activation_bytes(s, b, h, a, L) / 2 ** 30, 34 * s * h * L / 2 ** 30, what="FlashAttention")
    check_close(activation_bytes(s, b, h, a, L, recompute=True) / 2 ** 30, 1.0, what="全量重计算：约 1 GiB")
    no_flash = activation_bytes(s, b, h, a, L, flash=False) / 2 ** 30
    assert 90 < no_flash < 100, f"4K 序列不用 FlashAttention 时激活约 97 GiB，算出来 {no_flash:.1f}"


def test_per_gpu():
    act = activation_bytes(4096, 1, 4096, 32, 32)
    check_close(per_gpu_gib(7e9, 8, 3, act), (14e9 + act) / 2 ** 30, what="7B、ZeRO-3、8 卡、FlashAttention")
    assert per_gpu_gib(7e9, 1, 0, 0) > 80, "7B 模型不切分时，模型状态就超过 80 GB"
