import numpy as np
from cs336_basics.data_loading import data_loading

tk_filepath = r"D:\Dev\assignment1-basics\data\tokens_TinyStoriesV2_valid.npy"
tokens_valid = np.fromfile(tk_filepath, dtype=np.uint16)

sequences, targets = data_loading(tokens_valid, 16, 10, device='cuda')
