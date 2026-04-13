import torch
import torch.nn as nn
from cs336_basics.linear import Linear

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        '''
        d_model: int  Hidden dimension of the model
        d_ff: int  Hidden dimension of the model
        device: torch.device | None = None  Device to store the parameters on
        dtype: torch.dtype | None = None  Data type of the parameters
        '''
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(self.d_model, self.d_ff)
        self.w2 = Linear(self.d_ff, self.d_model)
        self.w3 = Linear(self.d_model, self.d_ff)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''''''
        x1 = self.w1(x)
        x1 = x1 * torch.sigmoid(x1)
        x3 = self.w3(x)
        result = self.w2(x1 * x3)
        return result
    
if __name__ == "__main__":
    ''''''
