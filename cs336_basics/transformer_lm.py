import torch
import torch.nn as nn
from cs336_basics.linear import Linear
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.embedding import Embedding
from cs336_basics.rope import RotaryPositionalEmbedding


class TransformerLM(nn.Module):
    def __init__(self, 
                vocab_size: int, 
                context_length: int, 
                num_layers: int, 
                d_model: int,
                num_heads: int,
                d_ff: int,
                rope_theta: float,
                device=None, 
                dtype=None
            ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.emb = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        self.rope = RotaryPositionalEmbedding(rope_theta, d_model//num_heads, context_length, device=device)

        self.TFB_lst = []
        for i in range(num_layers):
            self.TFB_lst.append(TransformerBlock(d_model, num_heads, d_ff, self.rope, device=device, dtype=dtype))

        self.final_rmsn = RMSNorm(d_model, device=device, dtype=dtype)

        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        ''''''
        x = self.emb(x)
        for tfb in self.TFB_lst:
            x = tfb(x)
        x = self.final_rmsn(x)
        return self.lm_head(x)
        

        
