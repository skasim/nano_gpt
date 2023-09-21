from dataclasses import dataclass
import torch
@dataclass
class TinyGPTConfig:
    # hyperparameters
    batch_size = 5  # how many independent sequences will we process in parallel?
    block_size = 8  # what is the maximum content length for our predictions?
    max_iters = 3000
    eval_interval = 300
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_iters = 200
    n_embed = 32 # number of embedding dimensions, i.e., the size of the learned embedding for each word
    n_heads = 4
    n_layers = 2
    dropout = 0.1
    vocab_size = 50257
    input_file = "data/input.txt"
    train_test_split = 0.9
#=================================

@dataclass
class OptimizerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.1