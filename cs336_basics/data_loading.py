import numpy as np
import torch


def data_loading(x: np.array, batch_size: int, context_length: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    ''''''
    seq = torch.from_numpy(x[:-1]).to(device=device)
    target = torch.from_numpy(x[1:]).to(device=device)

    start_point = np.random.randint(0, len(x)-context_length, size=batch_size)
    
    indices = start_point[:, None] + np.arange(context_length)

    sampled_seq = seq[indices]
    sampled_tar = target[indices]

    return sampled_seq, sampled_tar


if __name__ == "__main__":
    ''''''
    x = np.arange(100)
    data_loading(x, 32, 7)
    # print(x[1:])