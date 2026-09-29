def a2a_bytes(tokens_per_gpu, hidden, top_k, ep, dispatch_bytes=1, combine_bytes=2):
    remote = tokens_per_gpu * top_k * hidden * (ep - 1) / ep
    return remote * dispatch_bytes, remote * combine_bytes


def expected_nodes(top_k, n_nodes):
    return n_nodes * (1 - (1 - 1 / n_nodes) ** top_k)


def inter_node_bytes_per_token(hidden, top_k, n_nodes, max_nodes=None, dispatch_bytes=1):
    nodes = expected_nodes(top_k, n_nodes)
    if max_nodes is not None:
        nodes = min(nodes, max_nodes)
    return nodes * (n_nodes - 1) / n_nodes * hidden * dispatch_bytes
