import torch

def softmax(x: torch.Tensor, i: int = -1) -> torch.Tensor:
    x = x - torch.max(x, dim=i, keepdim=True).values
    x_exp = torch.exp(x)
    out = x_exp / torch.sum(x_exp, dim=i, keepdim=True)
    return out