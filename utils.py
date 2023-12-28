import numpy as np
import torch


class EMAMeter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.value = np.nan

    def update(self, x):
        self.value = (
            (self.alpha * self.value + (1 - self.alpha) * x)
            if not np.isnan(self.value)
            else x
        )


def check_soft_divergence(q_val, gamma):
    max_abs_q = np.abs(q_val).max()
    if max_abs_q > (1 / (1 - gamma)):
        raise ValueError(f"q_val divergence detected: {max_abs_q}")
    return q_val


def random_of_max(x):
    return np.random.choice(np.nonzero(x >= (np.max(x) - 1e-6))[0])


def maybe_to_cuda(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x
