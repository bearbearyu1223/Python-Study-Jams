import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def top_k_logits(logits: torch.Tensor, k: int):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    Take a conditioning sequence of indices in x (of shape (b, t)) and predict the next
    token in the sequence, feeding the predictions back into the model each time.
    """
    pass
