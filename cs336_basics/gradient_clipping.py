import torch
from collections.abc import Iterable
import numpy
from torch.nn.utils.clip_grad import clip_grad_norm_



def gradient_clipping(params_lst: Iterable, l2_max: float, eps: torch.Tensor=1e-6):
    ''''''
    all_grads = 0
    for params in params_lst:
        if params.grad is not None:
            all_grads += torch.sum(params.grad**2)

    l2_p = torch.sqrt(torch.sum(all_grads))
    
    for params in params_lst:
        if params.grad is not None:
            if l2_p > l2_max:
                factor = l2_max / (l2_p + eps)
                params.grad.data.mul_(factor)


if __name__ == "__main__":
    ''''''