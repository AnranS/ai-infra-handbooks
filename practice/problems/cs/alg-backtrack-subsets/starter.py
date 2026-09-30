def subsets(nums):
    out, path = [], []

    def dfs(start):
        out.append(path)                       # 没有拷贝：所有答案指向同一个列表
        for i in range(start, len(nums)):
            path.append(nums[i])
            dfs(i + 1)
            path.pop()

    dfs(0)
    return out


def permute(nums):
    out, path = [], []

    def dfs():
        if len(path) == len(nums):
            out.append(path[:])
            return
        for x in nums:
            if x in path:                      # 用值判断：有重复元素时会漏解
                continue
            path.append(x)
            dfs()
            path.pop()

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
            if nums[i] > remain:
                break
            path.append(nums[i])
            dfs(i + 1, remain - nums[i])       # 从 i+1 开始：不能重复使用了
            path.pop()

    dfs(0, target)
    return out
