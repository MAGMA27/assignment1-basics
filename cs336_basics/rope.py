import torch
import torch.nn as nn
from einops import einsum


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        '''
        theta: float  Θ value for the RoPE
        d_k: int  dimension of query and key vectors
        max_seq_len: int  Maximum sequence length that will be input
        device: torch.device | None = None  Device to store the buffer on
        '''
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        denominator = 1.0 / (theta ** (2 * torch.arange(0, d_k // 2).float() / d_k))
        token_id = torch.arange(max_seq_len)
        rope_theta = einsum(token_id.float(), denominator, 'max_seq_len, dk_2 -> max_seq_len dk_2')
        cos_buffer = torch.cos(rope_theta)
        self.register_buffer("cos_buffer", cos_buffer, persistent=False)
        sin_buffer = torch.sin(rope_theta)
        self.register_buffer("sin_buffer", sin_buffer, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor=None) -> torch.Tensor:
        '''
        x: Float[Tensor, "... sequence_length d_k"]
        token_positions: Int[Tensor, "... sequence_length"]
        '''
        if token_positions == None:
            token_positions = torch.arange(0, x.shape[-2])

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        output = torch.empty_like(x)
        
        # 旋转公式
        # x_out = [x1*cos - x2*sin, x1*sin + x2*cos]
        y1 = x1 * self.cos_buffer[token_positions] - x2 * self.sin_buffer[token_positions]
        output[..., 0::2] = y1
        y2 = x1 * self.sin_buffer[token_positions] + x2 * self.cos_buffer[token_positions]
        output[..., 1::2] = y2

        return output


if __name__ == "__main__":
    ''''''
