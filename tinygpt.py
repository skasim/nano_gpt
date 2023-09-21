"""
GPT Language Model using BPE and sinusoidal and cosinusoidal embeddings.
Adapted from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
Base is from https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
With additional updates from https://github.com/openai/gpt-2
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from bpe import BPETokenizer
from configs import TinyGPTConfig, OptimizerConfig

config = TinyGPTConfig()
torch.manual_seed(1337)

# Load data from file
with open(config.input_file, "r", encoding="utf-8") as f:
    text = f.read()
# print(f"TEXT:\n{text}")

# Encode text with BPE
bpe_tokenizer = BPETokenizer()
data = bpe_tokenizer(text)


# Create train and test split
train_split = int(config.train_test_split * data.shape[1])
train_data = data[:,:train_split]
test_data = data[:,train_split:]

# B=batch_size, T==block_size, C==vocab_size

# Data batching
def get_batch(split):
    # generate small batch of data of inputs x and targets y
    # this batch is based on the batch_size (num rows)
    # the block size is the num_cols
    # x and y should be of size (B, T)
    data = train_data if split == "train" else test_data
    idx = torch.randint(high=data.shape[1] - config.block_size, size=(config.batch_size,)) # random offsets in the dataset of size batch_size, so just a list of len(batch_size) of random offsets
    x = torch.stack([data[:,i:i+config.block_size] for i in idx])
    y = torch.stack([data[:,i+1:i+config.block_size+1] for i in idx])
    x, y = x.to(config.device), y.to(config.device)
    return x.squeeze(), y.squeeze()


class MultiheadAttentionLayer(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        assert config.n_embed % config.n_heads == 0
        self.resid_drop = nn.Dropout(config.dropout)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, device=config.device, dtype=dtype)
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        self.attn = nn.MultiheadAttention(config.n_embed, config.n_heads, batch_first=True)

    def forward(self, x):
        _, seq_size, _ = x.size()
        y = self.attn(query=x.clone(), key=x, value=x, attn_mask=self.mask[0, 0, :seq_size, :seq_size])[0] # had to use an x.clone() because of an error in pytorch 2.0.1 with making this bool mask and resulting in NaNs
        y = self.resid_drop(self.c_proj(y))
        return y


class Block(nn.Module):
    """Transformer block: communication followed by computation"""
    def __init__(self, n_embed, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.attn = MultiheadAttentionLayer()
        self.mlp = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        out = self.attn(self.ln1(x))
        x = x + out # GPT-2 moved layer activation to the input of each sub-block
        x = x + self.mlp(self.ln2(x))
        return x



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) # slice all the even colums
        pe[:, 1::2] = torch.cos(position * div_term) # slice all the odd columns
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class EmbeddingStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embed)
        self.pos_emb = PositionalEncoding(d_model=config.n_embed)
        self.drop = nn.Dropout(config.dropout)

    def reset_parameters(self):
        self.tok_emb.reset_parameters()

    def forward(self, idx):
        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb(token_embeddings)
        return self.drop(token_embeddings + position_embeddings)


class TinyGPT(nn.Module):
    """GPT Language Model"""
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        # input embedding stem
        self.emb_stem = EmbeddingStem()
        # transformer
        self.blocks = nn.Sequential(*[Block(config.n_embed, config.n_heads) for _ in range(config.n_layers)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embed) # according to GPT-2 paper add a layer norm after the final self-attention block
        self.head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # init all weights and apply a special scaled init to the residual projections per the GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                p.data.normal_(mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(self, idx, targets=None):
        x = self.emb_stem(idx)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # the non-normalized predictions
            targets = targets.view(B * T)  # actual values
            loss = F.cross_entropy(logits, targets)  # measures quality of logits wrt targets
            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss


    # continues generation for all the batch dims along the time dims
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)  # goes to forward function
            # focus only on the last time step
            logits = logits[:, -1, :]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # torch.multinomial returns a tensor where each row contains num_samples indices sampled
            # from the multinomial probability distribution
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

        # # idx is (B, T) array of indices in the current context
        # for _ in range(max_new_tokens):
        #     # if the sequence context is growing too long crop it at the block_size
        #     idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
        #     # forward the model to get the logits for the index in the sequence
        #     logits, _ = self(idx_cond)
        #     # pluck the logits at the final step and scale by desired temperature
        #     logits = logits[:,-1,:] / temperature
        #     # optionally crop the logits to only the top k options
        #     if top_k is not None:
        #         v, _ = torch.topk(logits, top_k)
        #         logits[logits < v[:, [-1]]] = -float("inf")
        #     # apply softmax to convert logits to (normalized) probabilities
        #     probs = F.softmax(logits, dim=-1)
        #     # either sample from the distribution or take the most likely element
        #     if do_sample:
        #         idx_next = torch.mul(probs, num_samples=1)
        #     else:
        #         _, idx_next = torch.topk(probs, k=1, dim=-1)
        #     # append sampled index to the running sequence and continue
        #     idx = torch.cat((idx, idx_next), dim=1)
        # return idx



def create_optimizer(model: torch.nn.Module, opt_config: OptimizerConfig):
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear,)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
            # random note: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times. but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('in_proj_weight'):
                # MHA projection layer
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif pn.endswith('pos_emb'):
                # positional embedding shouldn't be decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert len(
        param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params),)

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": opt_config.weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=opt_config.learning_rate, betas=(0.9, 0.95))
    return optimizer