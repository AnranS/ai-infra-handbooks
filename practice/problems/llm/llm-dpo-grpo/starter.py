import numpy as np


def sequence_logprob(logits, tokens, mask):
    pass


def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=0.1):
    z = beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))
    return float(-np.log(1 / (1 + np.exp(-z))).mean()), 0.0


def grpo_advantages(rewards, group_size, eps=1e-6):
    pass
