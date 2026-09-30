from checker import check
from solution import count_bits, is_power_of_two, single_number


def test_example():
    check(single_number([4, 1, 2, 1, 2]), 4, "找出单独的那个")
    check(count_bits(5), [0, 1, 1, 2, 1, 2], "0 到 5")
    check(is_power_of_two(16), True, "2 的四次方")


def test_single_edges():
    check(single_number([7]), 7, "只有一个数")
    check(single_number([0, 1, 1]), 0, "答案是 0")
    check(single_number([-3, 5, 5]), -3, "负数")


def test_count_edges():
    check(count_bits(0), [0], "只有 0")
    check(count_bits(1), [0, 1], "0 和 1")
    check(count_bits(8)[8], 1, "8 的二进制只有一个 1")
    check(len(count_bits(100)), 101, "包含 n 本身")


def test_power_edges():
    check(is_power_of_two(1), True, "2 的 0 次方")
    check(is_power_of_two(0), False, "0 不是")
    check(is_power_of_two(-2), False, "负数不是")
    check(is_power_of_two(3), False, "3 不是")
    check(is_power_of_two(1 << 30), True, "很大的 2 的幂")


def test_consistency():
    bits = count_bits(1000)
    for i in (0, 1, 7, 255, 256, 1000):
        check(bits[i], bin(i).count("1"), f"{i} 的位数")


def test_large():
    nums = []
    for i in range(50000):
        nums += [i, i]
    nums.append(123456)
    check(single_number(nums), 123456, "十万个数里找单独的")
    check(count_bits(100000)[100000], bin(100000).count("1"), "十万")
