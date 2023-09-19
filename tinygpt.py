import torch
import torch.nn as nn
from torch.nn import functional as F

from bpe import BPETokenizer

# hyperparameters
batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum content length for our predictions?
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embed = 32 # number of embedding dimensions, i.e., the size of the learned embedding for each word
n_heads = 4
n_layers = 4
dropout = 0.2
# ------------

torch.manual_seed(1337)

# Load data from file
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()
print(f"TEXT:\n{text}")


# Encode text with BPE
bpe_tokenizer = BPETokenizer()
data = bpe_tokenizer(text)

print(f"DATA:\n{data.shape}")

# Create train and test split
n_split = int(0.9 * len(data))
train_data = data[:n_split]
test_data = data[n_split+1:]

# B=batch_size, T==block_size, C==vocab_size

# Data batching
def get_batch(split):
    # generate small batch of data of inputs x and targets y
    # this batch is based on the batch_size (num rows)
    # the block size is the num_cols
    # x and y should be of size (B, T)
    data = train_data if split == "train" else test_data
    idx = torch.randint(high=len(data) - block_size, size=(batch_size,)) # random offsets in the dataset



