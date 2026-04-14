import torch
from einops import einsum
from cs336_basics.softmax import softmax

def scaled_dot_product_attention(
        Q:torch.Tensor, 
        K:torch.Tensor, 
        V:torch.Tensor, 
        mask=None) -> torch.Tensor:
    '''
    Q k: (batch_size, ..., seq_len, d_k)
    v: (batch_size, ..., seq_len, d_v)
    '''
    dk = Q.shape[-1]
    QKT = einsum(Q, K, "... s_q d_k, ... s_k d_k -> ... s_q s_k")
    scores = QKT / torch.sqrt(torch.tensor(dk))
    if mask is not None:
        scores = scores.masked_fill(~mask, -1e9)
    attention = einsum(softmax(scores), V, "... s_q s_kv, ... s_kv d_v ->  ... s_q d_v")
    return attention

if __name__ == "__main__":
    ''''''
    q = torch.tensor([
    [1,1,1,1],
    [2,2,2,2],
    [3,3,3,3],
], dtype=torch.float32)
    k = torch.tensor([
    [1,1,1,1],
    [2,2,2,2],
    [3,3,3,3],
], dtype=torch.float32)
    v = torch.tensor([
    [1,1,1,1],
    [2,2,2,2],
    [3,3,3,3],
], dtype=torch.float32)
    attention = scaled_dot_product_attention(q, k, v)
    print(attention)