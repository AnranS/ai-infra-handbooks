from checker import check
from solution import is_valid, longest_valid


def test_example():
    check(is_valid("()[]{}"), True, "都配对")
    check(is_valid("([)]"), False, "交叉")
    check(longest_valid(")()())"), 4, "中间四个")


def test_valid_edges():
    check(is_valid(""), True, "空串有效")
    check(is_valid("("), False, "只有左括号")
    check(is_valid(")"), False, "只有右括号")
    check(is_valid("(("), False, "结束时栈非空")
    check(is_valid("))"), False, "开头就是右括号")


def test_nested():
    check(is_valid("{[()]}"), True, "嵌套")
    check(is_valid("{[(])}"), False, "嵌套错位")


def test_longest_edges():
    check(longest_valid(""), 0, "空串")
    check(longest_valid("("), 0, "单个左括号")
    check(longest_valid("()"), 2, "一对")
    check(longest_valid("(()"), 2, "多一个左括号")
    check(longest_valid("()(()"), 2, "两段，最长的是 2")


def test_longest_nested():
    check(longest_valid("(())"), 4, "嵌套的整体有效")
    check(longest_valid("()(())"), 6, "连起来")
    check(longest_valid(")(((((()())()()))()(()))("), 22, "复杂例子")


def test_large():
    check(is_valid("(" * 50000 + ")" * 50000), True, "十万个括号")
    check(longest_valid("()" * 50000), 100000, "全部有效")
