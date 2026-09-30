---
title: 地址翻译与 TLB 命中率
chapter: os/virtual-memory.md
difficulty: 简单
tags: [虚拟内存, TLB, 大页, LRU]
---
实现两个函数：

1. `split_va(addr)`：x86-64 四级页表、4 KiB 页，把 48 位虚拟地址拆成 `(pml4, pdpt, pd, pt, offset)`：4 个 9 位的页表下标（从高位到低位）和 12 位的页内偏移；
2. `tlb_stats(addrs, page_size, entries)`：按顺序访问 `addrs` 里的虚拟地址，TLB 缓存"虚拟页号"（`addr // page_size`），最多 `entries` 项，满了按 **LRU** 淘汰最久没用过的一项。返回 `(hits, misses)`。

```python
split_va(0x7F3A1C2D5E6F)                  # (254, 232, 225, 213, 3695)
tlb_stats([0, 4096, 0, 8192, 4096], 4096, 2)   # (1, 4)：第二次访问第 1 页时它已经被淘汰了
```

<!-- 题解 -->
`split_va` 就是移位加掩码：偏移是低 12 位（`addr & 0xFFF`），往上每 9 位是一级页表的下标（`(addr >> 12) & 0x1FF` 是最后一级，`>> 21`、`>> 30`、`>> 39` 依次往上）。一级页表 512 项、每项 8 字节，正好一页。

TLB 用 `collections.OrderedDict` 实现 LRU 很方便：命中时 `move_to_end`，缺失时插入，超出容量时 `popitem(last=False)` 弹出最久没用的。用先进先出淘汰会在"反复访问的热页"上吃亏：热页即使一直在用，也会按进入的顺序被踢掉。

测试里的顺序扫描说明了大页的作用：64 项的 TLB、4 KiB 页只能覆盖 256 KiB，扫描 8 MiB 每进入一页都缺失一次（2048 次）；换成 2 MiB 页只缺失 4 次。
