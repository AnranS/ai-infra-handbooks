from checker import check
from solution import add_request, make_grid, remove_finished, same_model, total_cost


def test_example_mutable_default():
    check(add_request("a"), ["a"], 'add_request("a")')
    check(add_request("b"), ["b"], '第二次调用 add_request("b")')
    q = ["x"]
    check(add_request("y", q), ["x", "y"], "传入已有队列")


def test_remove_finished():
    reqs = [{"id": i, "finished": f} for i, f in enumerate([True, True, False, True, False, False, True])]
    alias = reqs
    ret = remove_finished(reqs)
    check(ret, None, "返回值")
    check([r["id"] for r in reqs], [2, 4, 5], "剩下的请求")
    assert alias is reqs and len(alias) == 3, "应该原地修改（同一个列表对象）"


def test_grid():
    g = make_grid(3, 2)
    g[0][1] = 7
    check(g, [[0, 7], [0, 0], [0, 0]], "修改一个格子之后的网格")


def test_same_model():
    a = "".join(["qwen3-", "0.6b"])
    b = "qwen3-0.6b"
    check(same_model(a, b), True, "内容相同但不是同一个对象的字符串")
    check(same_model("a", "b"), False, "不同的字符串")


def test_total_cost():
    check(total_cost([0.1, 0.2]), 0.3, "total_cost([0.1, 0.2])")
    check(total_cost([19.99] * 3), 59.97, "total_cost([19.99] * 3)")
    check(total_cost([]), 0, "total_cost([])")
