import torch
import torch.nn as nn
from einops import einsum
import math

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        '''
        in_features: int  final dimension of the input
        out_features: int  final dimension of the output
        device: torch.device | None = None  Device to store the parameters on
        dtype: torch.dtype | None = None  Data type of the parameters
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(
            torch.empty((self.out_features, self.in_features), device=device, dtype=dtype)
        )
        
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weights, mean=0.0, std=std, a=-3*std, b=3*std)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''''''
        return einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")
    
if __name__ == "__main__":
    ln = Linear(2, 3, device='cpu', dtype=torch.float32)
    print(ln(torch.ones(2)))