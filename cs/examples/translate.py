# x86-64 四级页表：48 位虚拟地址 = 4 个 9 位的页表下标 + 12 位的页内偏移（页大小 4 KiB）
addr = 0x7F3A_1C2D_5E6F

offset = addr & 0xFFF
idx = [(addr >> shift) & 0x1FF for shift in (39, 30, 21, 12)]
for name, i in zip(["PML4（第 4 级）", "PDPT（第 3 级）", "PD（第 2 级）", "PT（第 1 级）"], idx):
    print(f"{name}下标：{i}")
print(f"页内偏移：{offset:#x}")
print(f"一张页表 512 项 × 8 字节 = {512 * 8} 字节，正好一页")
print(f"一个 2 MiB 大页直接由第 2 级页表项指向，偏移占低 {21} 位，少查一级")
