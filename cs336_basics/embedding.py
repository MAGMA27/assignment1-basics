import torch
import torch.nn as nn
from einops import einsum

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        '''
        num_embeddings: int  Size of the vocabulary
        embedding_dim: int  Dimension of the embedding vectors, i.e., d_model
        device: torch.device | None = None  Device to store the parameters on
        dtype: torch.dtype | None = None  Data type of the parameters
        '''
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding_mat = torch.Tensor(num_embeddings, embedding_dim)

        if device:
            self.device = device
            self.embedding_mat.to(self.device)

        if dtype:
            self.dtype = dtype
            self.embedding_mat.to(self.dtype)
        
        std = 1
        nn.init.trunc_normal_(self.embedding_mat, mean=0.0, std=std, a=-3, b=3)
        
        self.embedding_mat = nn.Parameter(self.embedding_mat)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        ''''''
        embedded_vectors = self.embedding_mat[token_ids]
        return embedded_vectors
    
if __name__ == "__main__":
    emb = Embedding(4, 5, device='cpu', dtype=torch.float32)
    data_list = [[0, 1], [2, 3]]
    print(emb(torch.tensor(data_list)))