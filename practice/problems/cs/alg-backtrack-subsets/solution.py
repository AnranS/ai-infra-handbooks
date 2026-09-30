def subsets(nums):
    out, path = [], []

    def dfs(start):
        out.append(path[:])                    # 必须拷贝
        for i in range(start, len(nums)):
            path.append(nums[i])
            dfs(i + 1)
            path.pop()                         # 撤销选择

    dfs(0)
    return out


def permute(nums):
    out, path = [], []
    used = [False] * len(nums)

    def dfs():
        if len(path) == len(nums):
            out.append(path[:])
            return
        for i, x in enumerate(nums):
            if used[i]:
                continue
            used[i] = True
            path.append(x)
            dfs()
            path.pop()
            used[i] = False

    dfs()
    return out


def combination_sum(candidates, target):
    out, path = [], []
    nums = sorted(candidates)

    def dfs(start, remain):
        if remain == 0:
            out.append(path[:])
            return
        for i in range(start, len(nums)):
            if nums[i] > remain:               # 排序后可以直接 break
                break
            path.append(nums[i])
            dfs(i, remain - nums[i])           # 还是 i：可以重复使用
            path.pop()

    dfs(0, target)
    return out
