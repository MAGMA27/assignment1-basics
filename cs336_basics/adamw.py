from collections.abc import Callable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "eps": eps
            }
        
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            beta1, beta2 = group['betas']

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['t'] = 0

                state['t'] += 1  # Increment iteration number.
                m, v, t = state['m'], state['v'], state['t']
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data -= lr * weight_decay * p.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad ** 2
                p.data -= lr_t * m / (torch.sqrt(v) + eps)  # Update weight tensor in-place.
                state['m'], state['v'] = m, v

        return loss