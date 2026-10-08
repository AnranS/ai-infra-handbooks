# 回溯：做选择 -> 递归 -> 撤销选择。剪枝决定了它能不能跑得动
def permutations(nums):
    out, path, used = [], [], [False] * len(nums)

    def dfs():
        if len(path) == len(nums):
            out.append(path[:])                # 一定要拷贝：path 后面还会被修改
            return
        for i, x in enumerate(nums):
            if used[i]:
                continue
            used[i], _ = True, path.append(x)
            dfs()
            used[i] = False                    # 撤销选择
            path.pop()

    dfs()
    return out


def subsets(nums):
    out, path = [], []

    def dfs(start):
        out.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            dfs(i + 1)                         # 从 i+1 开始：不重复选同一个
            path.pop()

    dfs(0)
    return out


def combination_sum(candidates, target):
    """每个数可以用多次，结果不能重复"""
    out, path = [], []
    nums = sorted(candidates)

    def dfs(start, remain):
        if remain == 0:
            out.append(path[:])
            return
        for i in range(start, len(nums)):
            if nums[i] > remain:               # 剪枝：排序后，后面的只会更大
                break
            path.append(nums[i])
            dfs(i, remain - nums[i])           # 还是 i：允许重复使用
            path.pop()

    dfs(0, target)
    return out


def n_queens(n):
    """统计 n 皇后的解数：用三个集合记录被占用的列和两条对角线"""
    cols, diag1, diag2 = set(), set(), set()
    count = 0

    def dfs(row):
        nonlocal count
        if row == n:
            count += 1
            return
        for col in range(n):
            if col in cols or row - col in diag1 or row + col in diag2:
                continue
            cols.add(col), diag1.add(row - col), diag2.add(row + col)
            dfs(row + 1)
            cols.remove(col), diag1.remove(row - col), diag2.remove(row + col)

    dfs(0)
    return count


print("全排列 [1,2,3]：", permutations([1, 2, 3]))
print("子集 [1,2,3]：", subsets([1, 2, 3]))
print("组合总和（[2,3,6,7] 凑 7）：", combination_sum([2, 3, 6, 7], 7))
print("n 皇后的解数：", {n: n_queens(n) for n in range(4, 10)})
print()
print("模板都是：做选择 -> 递归 -> 撤销。区别在于起点（start）怎么传、什么时候剪枝。")
