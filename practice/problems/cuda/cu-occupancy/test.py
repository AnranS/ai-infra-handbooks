from checker import check, check_close
from solution import occupancy

H100 = {"max_threads": 2048, "max_blocks": 32, "regs": 65536, "smem": 228 * 1024, "max_warps": 64,
        "reg_alloc_unit": 256, "smem_alloc_unit": 128, "smem_per_block_reserved": 1024}
A100 = dict(H100, smem=164 * 1024)


def test_example():
    b, occ, lim = occupancy(128, 168, 64 * 1024, H100)
    check((b, lim), (3, "registers"), "128 线程、168 寄存器、64 KB 共享内存")
    check_close(occ, 12 / 64, what="占用率")


def test_threads_and_blocks_limits():
    check(occupancy(1024, 32, 0, H100)[:1] + occupancy(1024, 32, 0, H100)[2:], (2, "threads"), "1024 线程的 block")
    check(occupancy(32, 16, 0, H100)[::2], (32, "blocks"), "32 线程的小 block 受 block 数限制")
    check_close(occupancy(32, 16, 0, H100)[1], 0.5, what="32 个 block × 1 warp / 64")


def test_register_granularity():
    """每线程 33 个寄存器：每 warp 1056 → 向上取整到 1280"""
    b, occ, lim = occupancy(256, 33, 0, A100)
    check((b, lim), (6, "registers"), "256 线程、33 寄存器")
    check_close(occ, 48 / 64, what="占用率")


def test_shared_memory_limit():
    b, occ, lim = occupancy(256, 32, 48 * 1024, A100)
    check((b, lim), (3, "shared_memory"), "每 block 48 KB 共享内存（加 1 KB 保留）")
    check(occupancy(128, 0, 100 * 1024, A100)[0], 1, "100 KB")


def test_does_not_fit():
    check(occupancy(128, 255, 300 * 1024, H100)[::2], (0, "shared_memory"), "共享内存超过 SM 的容量")
    check(occupancy(2048, 32, 0, H100)[0], 0, "超过 1024 线程")
    check(occupancy(1024, 255, 0, H100)[::2], (0, "registers"), "1024 线程 × 255 寄存器放不下")


def test_no_registers_given():
    check(occupancy(512, 0, 0, H100)[::2], (4, "threads"), "regs_per_thread=0 不受寄存器限制")
