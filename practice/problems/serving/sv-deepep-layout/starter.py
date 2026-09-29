import numpy as np


def dispatch_layout(topk_idx, num_experts, num_ranks):
    pass


def recv_plan(all_topk, num_experts):
    pass


def ll_slots(topk_idx, src_rank, num_experts, num_ranks, max_tokens):
    pass


def ll_buffer_bytes(num_experts, num_ranks, max_tokens, msg_bytes):
    return max_tokens * msg_bytes                     # 只算了一个槽位组
