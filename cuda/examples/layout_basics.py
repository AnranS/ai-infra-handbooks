from layout_core import Layout, coalesce


def show(layout, rows, cols):
    """把二维布局画成表格：第 i 行第 j 列是坐标 (i, j) 映射到的下标"""
    print(layout)
    for i in range(rows):
        print("  " + " ".join(f"{layout((i, j)):3d}" for j in range(cols)))


col = Layout((4, 8))                                 # 默认步长：列优先，(4,8):(1,4)
show(col, 4, 8)
show(Layout((4, 8), (8, 1)), 4, 8)                   # 行优先
h = Layout((4, (2, 4)), (2, (1, 8)))                 # 列这一维拆成 (2,4)：两列一组，组内相邻，组间隔 8
show(h, 4, 8)
print("h((1,5)) =", h((1, 5)), " h((1,(1,2))) =", h((1, (1, 2))), " h(5) =", h(5))
print("一维坐标 0..7 →", [h(i) for i in range(8)])
print("size =", h.size(), " cosize =", h.cosize(), " rank =", len(h), " h[1] =", h[1])
print("coalesce((2,(1,6)):(1,(6,2))) =", coalesce(Layout((2, (1, 6)), (1, (6, 2)))))
print("coalesce((4,8):(1,4)) =", coalesce(col), " coalesce((4,8):(8,1)) =", coalesce(Layout((4, 8), (8, 1))))
