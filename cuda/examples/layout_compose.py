from layout_algebra import composition, complement, logical_divide
from layout_core import Layout, make_layout

a = Layout((6, 2), (8, 2))
b = Layout((4, 3), (3, 1))
r = composition(a, b)
print("A =", a, " B =", b, " A∘B =", r)
print("逐个核对 R(i) == A(B(i))：", all(r(i) == a(b(i)) for i in range(b.size())))
for bad in (Layout(4, 2), Layout(4, 4)):                 # 不满足整除条件的复合
    try:
        composition(a, bad)
    except AssertionError as e:
        print(f"A∘{bad}：{e}")

m = Layout((4, 8), (8, 1))                           # 4×8 的行优先矩阵
t = composition(m, Layout((8, 4), (4, 1)))
print("转置视图：", t, " t((5,2)) =", t((5, 2)), " m((2,5)) =", m((2, 5)))
e = composition(m, Layout((2, 8), (2, 4)))
print("只取偶数行：", e, " 第 1 行 =", [e((1, j)) for j in range(8)])

c = complement(Layout(4, 2), 24)
print("complement(4:2, 24) =", c, " 拼起来：", make_layout(Layout(4, 2), c),
      " 覆盖 0..23 各一次：", sorted(make_layout(Layout(4, 2), c)(i) for i in range(24)) == list(range(24)))
print("complement((2,2):(1,6), 24) =", complement(Layout((2, 2), (1, 6)), 24))

d = logical_divide(Layout(24), Layout(4, 2))
print("logical_divide(24:1, 4:2) =", d)
for k in range(d[1].size()):
    print(f"  第 {k} 块：", [d((i, k)) for i in range(4)])
