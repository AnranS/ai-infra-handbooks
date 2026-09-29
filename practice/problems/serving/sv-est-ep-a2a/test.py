from checker import check_close
from solution import a2a_bytes, expected_nodes, inter_node_bytes_per_token


def test_example():
    d, c = a2a_bytes(4096, 7168, 8, 64)
    check_close(d, 4096 * 8 * 7168 * 63 / 64, rtol=1e-12, what="dispatch")
    check_close(c, 2 * d, rtol=1e-12, what="combine 是 BF16，字节数翻倍")
    check_close(expected_nodes(8, 8), 8 * (1 - (7 / 8) ** 8), rtol=1e-12, what="8 个专家落在 8 个节点")
    ratio = inter_node_bytes_per_token(7168, 8, 8, max_nodes=4) / inter_node_bytes_per_token(7168, 8, 8)
    check_close(ratio, 4 / expected_nodes(8, 8), rtol=1e-12, what="节点限制后的比例")
    assert 0.74 < ratio < 0.78, f"限制到 4 个节点后跨机流量约为原来的 76%，算出来 {ratio:.1%}"


def test_scale_independence():
    d64 = a2a_bytes(4096, 7168, 8, 64)[0]
    d256 = a2a_bytes(4096, 7168, 8, 256)[0]
    assert d256 / d64 < 1.02, "EP 从 64 扩到 256，每卡通信量几乎不变"
    check_close(a2a_bytes(100, 1024, 2, 1)[0], 0.0, atol=1e-9, what="ep=1 时没有通信")


def test_expected_nodes_limits():
    check_close(expected_nodes(1, 8), 1.0, what="top-1 只涉及 1 个节点")
    check_close(expected_nodes(8, 1), 1.0, what="只有 1 个节点")
    check_close(expected_nodes(1000, 4), 4.0, rtol=1e-9, what="专家很多时趋近全部节点")
    check_close(inter_node_bytes_per_token(4096, 4, 1), 0.0, atol=1e-9, what="单节点没有跨机流量")
    check_close(inter_node_bytes_per_token(4096, 8, 8, max_nodes=100), inter_node_bytes_per_token(4096, 8, 8),
                what="限制比期望还宽松时没有影响")
