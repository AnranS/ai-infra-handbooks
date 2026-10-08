# 面试写题时 Python 特有的几个坑，每个都能在现场把人绊倒
import sys

# 1. 默认参数是在定义时求值的，可变默认参数会被所有调用共享
def collect(x, acc=[]):
    acc.append(x)
    return acc


print("可变默认参数：", collect(1), collect(2), "  <- 第二次调用带着上一次的结果")


def collect_ok(x, acc=None):
    acc = [] if acc is None else acc
    acc.append(x)
    return acc


print("正确写法：    ", collect_ok(1), collect_ok(2))

# 2. 二维数组不能用 [[0] * n] * m：外层是同一个列表的 m 个引用
grid = [[0] * 3] * 2
grid[0][0] = 1
print("\n[[0]*3]*2 改一个元素：", grid, " <- 两行都变了")
grid = [[0] * 3 for _ in range(2)]
grid[0][0] = 1
print("列表推导式：          ", grid)

# 3. 切片是拷贝：在循环里切片会让复杂度多一个 n
def has_dup_slow(a):
    return any(x in a[i + 1:] for i, x in enumerate(a))    # 每次切片都拷贝一份


def has_dup_fast(a):
    return len(set(a)) != len(a)


print("\n切片会拷贝：两种写法结果一致 =", has_dup_slow([1, 2, 3, 2]) == has_dup_fast([1, 2, 3, 2]))

# 4. 递归深度：默认上限 1000，链表和树的题目很容易超
def depth(n):
    return 0 if n == 0 else 1 + depth(n - 1)


print("默认递归上限：", sys.getrecursionlimit())
try:
    depth(2000)
except RecursionError:
    print("递归 2000 层：RecursionError —— 链表和退化成链的树要用迭代写法")

# 5. 整数除法与负数取整
print("\n-7 // 2 =", -7 // 2, "（向下取整），int(-7 / 2) =", int(-7 / 2), "（向零取整）")
print("-7 % 3 =", -7 % 3, "（Python 的余数跟除数符号），C 语言里是 -1")

# 6. 排序的稳定性与自定义键
words = [("b", 2), ("a", 2), ("c", 1)]
print("\n按第二个元素排序（稳定，相等的保持原序）：", sorted(words, key=lambda t: t[1]))
print("按第二个降序、第一个升序：", sorted(words, key=lambda t: (-t[1], t[0])))
