def single_number(nums):
    for x in nums:
        if nums.count(x) == 1:                 # O(n²)，而且用了额外的遍历
            return x
    return 0


def count_bits(n):
    return [bin(i).count("1") for i in range(n)]   # 少了 n 本身


def is_power_of_two(n):
    return n & (n - 1) == 0                    # n = 0 时会误判为 True
