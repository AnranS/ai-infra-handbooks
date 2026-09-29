from fractions import Fraction


def lost_experts(cards, dead):
    return sorted({e for g in dead for e in cards[g]})       # 坏卡上的专家不一定丢：别的卡上可能还有副本


def min_failures_to_lose(cards):
    pass


def prob_no_loss(cards, k):
    pass


def replica_counts(num_experts, load, extra):
    pass
