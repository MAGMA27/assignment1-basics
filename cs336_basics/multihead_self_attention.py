import torch
import torch.nn as nn
from einops import rearrange
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.linear import Linear


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        '''
        d_model: int Dimensionality of the Transformer block inputs.
        num_heads: int Number of heads to use in multi-head self-attention.
        '''
        super().__init__()
        self.d_model = d_model
        self.h = num_heads
        self.Q = Linear(d_model, d_model, device=None, dtype=None)
        self.K = Linear(d_model, d_model, device=None, dtype=None)
        self.V = Linear(d_model, d_model, device=None, dtype=None)
        self.O = Linear(d_model, d_model, device=None, dtype=None)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        x: (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.
        '''
        sq_l = x.shape[-2]
        ones_matrix = torch.ones(sq_l, sq_l)
        causal_mask = torch.tril(ones_matrix, diagonal=0).to(torch.bool)

        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)
        Q = rearrange(Q, "... sq_l (h d) -> h ... sq_l d", h=self.h)
        K = rearrange(K, "... sq_l (h d) -> h ... sq_l d", h=self.h)
        V = rearrange(V, "... sq_l (h d) -> h ... sq_l d", h=self.h)

        attention = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
        attention = rearrange(attention, "h ... sq_l d -> ... sq_l (h d)", h=self.h)
        output = self.O(attention)

        return output
    
if __name__ == "__main__":
    ''''''
    test = MultiheadSelfAttention(5, 1)
    test(torch.ones(5, 5))