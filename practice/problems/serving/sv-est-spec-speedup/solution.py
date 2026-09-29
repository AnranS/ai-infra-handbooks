def expected_tokens(alpha, gamma):
    if alpha >= 1:
        return gamma + 1
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


def speedup(alpha, gamma, c):
    return expected_tokens(alpha, gamma) / (gamma * c + 1)


def best_gamma(alpha, c, max_gamma=16):
    return max(range(max_gamma + 1), key=lambda g: (speedup(alpha, g, c), -g))


def estimate_alpha(accepted, gamma):
    success = sum(accepted)
    failure = sum(1 for a in accepted if a < gamma)
    return success / (success + failure) if success + failure else 0.0
