import torch
import torch.nn as nn
from cs336_basics.multihead_self_attention_rope import MultiheadSelfAttentionRoPE
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.positionwise_feedforward import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device=None, dtype=None):
        '''
        d_model: int Dimensionality of the Transformer block inputs.
        num_heads: int Number of heads to use in multi-head self-attention.
        d_ff: int Dimensionality of the position-wise feed-forward inner layer.
        '''
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.RMSN1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.MHA = MultiheadSelfAttentionRoPE(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.RMSN2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.FF = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor=None) -> torch.Tensor:
        ''''''
        y = x + self.MHA(self.RMSN1(x), token_positions=token_positions)
        output = y + self.FF(self.RMSN2(y))

        return output



