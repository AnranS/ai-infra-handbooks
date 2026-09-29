from checker import check
from solution import coords, intra_node, megatron_order, mesh_groups

SIZES = {"tp": 2, "cp": 2, "dp": 2, "pp": 2}
ORDER = ["pp", "dp", "cp", "tp"]


def test_example():
    g = mesh_groups(SIZES, ORDER)
    check(g["cp"][:2], [[0, 2], [1, 3]], "CP 组")
    check(coords(13, SIZES, ORDER), {"pp": 1, "dp": 1, "cp": 0, "tp": 1}, "rank 13 的坐标")


def test_groups():
    g = mesh_groups(SIZES, ORDER)
    check(g["tp"], [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]], "TP 组")
    check(g["cp"][:4], [[0, 2], [1, 3], [4, 6], [5, 7]], "CP 组")
    check(g["dp"][:4], [[0, 4], [1, 5], [2, 6], [3, 7]], "DP 组")
    check(g["pp"][:4], [[0, 8], [1, 9], [2, 10], [3, 11]], "PP 组")
    check([len(g[d]) for d in ORDER], [8, 8, 8, 8], "每一维都是 8 组")
    g = mesh_groups({"tp": 4, "dp": 3}, ["dp", "tp"])
    check(g["dp"], [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]], "度数不同的两维")
    g = mesh_groups({"tp": 2, "pp": 2, "dp": 2}, ["tp", "pp", "dp"])
    check(g["tp"], [[0, 4], [1, 5], [2, 6], [3, 7]], "TP 放在最外层时，TP 组的 rank 不再相邻")


def test_coords():
    for r in range(16):
        c = coords(r, SIZES, ORDER)
        check(c["pp"] * 8 + c["dp"] * 4 + c["cp"] * 2 + c["tp"], r, f"rank {r} 的坐标能还原")
    check(coords(11, {"tp": 4, "dp": 3}, ["dp", "tp"]), {"dp": 2, "tp": 3}, "混合进制")


def test_order_and_nodes():
    check(megatron_order("tp-cp-ep-dp-pp", SIZES), ["pp", "dp", "cp", "tp"], "Megatron 的默认顺序；ep 不占维度")
    check(megatron_order("tp-cp-pp-dp", SIZES), ["dp", "pp", "cp", "tp"], "Llama 3 的顺序")
    g = mesh_groups(SIZES, ORDER)
    check([intra_node(g[d], 8) for d in ORDER], [False, True, True, True], "pp 跨节点，其余都在节点内")
    g3 = mesh_groups(SIZES, megatron_order("tp-cp-pp-dp", SIZES))
    check((intra_node(g3["dp"], 8), intra_node(g3["pp"], 8)), (False, True), "Llama 3 的顺序：DP 跨节点、PP 在节点内")
    big = mesh_groups({"tp": 8, "dp": 4}, ["dp", "tp"])
    check((intra_node(big["tp"], 8), intra_node(big["dp"], 8)), (True, False), "TP=8 占满一个节点")
