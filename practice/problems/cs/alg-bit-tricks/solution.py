def single_number(nums):
    result = 0
    for x in nums:
        result ^= x                            # 成对的互相抵消
    return result


def count_bits(n):
    f = [0] * (n + 1)
    for i in range(1, n + 1):
        f[i] = f[i >> 1] + (i & 1)             # 去掉最低位之后的结果已经算过
    return f


def is_power_of_two(n):
    return n > 0 and n & (n - 1) == 0
