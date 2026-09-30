from checker import check, check_close
from solution import accum_bytes, fits_in_registers, operand_bytes_per_fma, smem_pressure


def test_example():
    bytes_per_cycle, ratio = smem_pressure(64, 64, 2048)
    check_close(bytes_per_cycle, 128.0, rtol=1e-9, what="Hopper 64×64：每周期 128 字节")
    check_close(ratio, 1.0, rtol=1e-9, what="正好占满共享内存带宽")
    bytes_per_cycle, ratio = smem_pressure(64, 256, 2048)
    check_close(bytes_per_cycle, 80.0, rtol=1e-9, what="wgmma m64n256")
    check_close(ratio, 0.625, rtol=1e-9, what="占 62.5%")
    check(accum_bytes(128, 256), 131072, "128×256 的 FP32 累加器是 128 KB")
    regs, fits = fits_in_registers(128, 256, 128)
    check((regs, fits), (256.0, False), "超过每线程 255 个寄存器")


def test_bigger_tile_is_cheaper():
    small = operand_bytes_per_fma(64, 64)
    big = operand_bytes_per_fma(128, 256)
    check(big < small, True, "块越大，每次乘加要的操作数字节越少")
    check_close(small, 0.0625, rtol=1e-9, what="64×64")
    check_close(big, 2 * (128 + 256) / (128 * 256), rtol=1e-9, what="128×256")


def test_operand_from_registers():
    # FlashAttention-3 算 PV 时，A（softmax 的结果）就在寄存器里，只有 B 从共享内存读
    check_close(operand_bytes_per_fma(64, 128, a_from_smem=False), 2 * 128 / (64 * 128),
                rtol=1e-9, what="A 从寄存器读")
    _, ratio = smem_pressure(64, 128, 2048, a_from_smem=False)
    check_close(ratio, 0.5, rtol=1e-9, what="A 不走共享内存时压力减半")


def test_blackwell_two_sm():
    # Blackwell：每 SM 每周期 4096 次乘加。单 SM 算 128×256，或者两个 SM 合作算 256×256（每个 SM 放一半的 B）
    _, one = smem_pressure(128, 256, 4096)
    _, pair = smem_pressure(128, 256, 4096, b_from_smem=False)
    check_close(one, 0.75, rtol=1e-9, what="单 SM 128×256 要 75% 的带宽")
    check(pair < one, True, "两个 SM 分担 B 之后压力更小")


def test_tensor_memory():
    check(accum_bytes(128, 256) * 2 <= 256 * 1024, True, "256 KB 的 Tensor Memory 放得下两块累加器")
    check(fits_in_registers(64, 64, 128)[1], True, "Ampere 的 64×64 累加器放得进寄存器")
    regs, fits = fits_in_registers(64, 256, 128)
    check((regs, fits), (128.0, True), "Hopper 的 64×256：每线程 128 个寄存器")
