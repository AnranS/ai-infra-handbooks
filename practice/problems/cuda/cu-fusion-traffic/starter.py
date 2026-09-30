def eager_bytes(graph):
    return sum(nbytes for name, kind, inputs, nbytes in graph if kind != "input")     # 只算了写，没算读


def fused_bytes(graph, group):
    pass


def fusion_groups(graph):
    pass


def plan_bytes(graph):
    pass
