import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        '''
        d_model: int  Hidden dimension of the model
        eps: float = 1e-5  Epsilon value for numerical stability
        device: torch.device | None = None  Device to store the parameters on
        dtype: torch.dtype | None = None  Data type of the parameters
        '''
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''''''
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # Your code here performing RMSNorm
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.rsqrt(variance + self.eps)
        result = x * rms * self.gain
        # Return the result in the original dtype
        return result.to(in_dtype)
    
if __name__ == "__main__":
    ''''''
    t = torch.ones(5)
