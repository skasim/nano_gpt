import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from bpe import BPETokenizer

# hyperparameters
batch_size = 5  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum content length for our predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embed = 32 # number of embedding dimensions, i.e., the size of the learned embedding for each word
n_heads = 4
n_layers = 4
dropout = 0.2
vocab_size = 50257
# ------------

torch.manual_seed(1337)

# Load data from file
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()
# print(f"TEXT:\n{text}")

# Encode text with BPE
bpe_tokenizer = BPETokenizer()
data = bpe_tokenizer(text)


# Create train and test split
train_split = int(0.9 * data.shape[1])
train_data = data[:,:train_split]
test_data = data[:,train_split:]

# B=batch_size, T==block_size, C==vocab_size

# Data batching
def process_batch_row(row):
    return [data[0][i] for i in row]
def get_batch(split):
    # generate small batch of data of inputs x and targets y
    # this batch is based on the batch_size (num rows)
    # the block size is the num_cols
    # x and y should be of size (B, T)
    data = train_data if split == "train" else test_data
    idx = torch.randint(high=data.shape[1] - block_size, size=(batch_size,)) # random offsets in the dataset of size batch_size, so just a list of len(batch_size) of random offsets
    x = torch.stack([data[:,i:i+block_size] for i in idx])
    y = torch.stack([data[:,i+1:i+block_size+1] for i in idx])
    x, y = x.to(device), y.to(device)
    return x.squeeze(), y.squeeze()


class Head(nn.Module):
    """one head of self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # B, T, C
        q = self.query(x) # B, T, C
        # compute attention scores ("affinities)
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T) AND C**-0.5 is 1/sqrt(C)
        # the way transpose works is that we are saying transpose the innermost (-1) with the one before (-2)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei) # dropout randomly prevents some of the nodes from communicating
        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of attention in parallel"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(Head(head_size) for _ in range(num_heads))
        # nn.ModuleList is like a python list, so this is creating num head modules
        self.proj = nn.Linear(n_embed, n_embed)  # to make linear the inputs
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concatenating outputs over channel dim
        out = self.proj(out) # project is the linear outcome of the above concatenation
        out = self.dropout(out)
        return out


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            NewGELU(),
            nn.Linear(4 * n_embed, n_embed), # project layer going back into the residual pathway
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""
    def __init__(self, n_embed, n_heads):
        super().__init__()
        head_size = n_embed//n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
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


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = PositionalEncoding(d_model=n_embed)
        # self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential( # so that we are interspersing communication and computation
            * [Block(n_embed, n_heads=n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)  # at the end fof the transformer and right before final linear layer you need a layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size)
        # nn.Embedding is a simple lookup table that stores embeddings of a fixed dictionary and size
        # you store word embeddings with this module and retrieve them with indices
        # input to the module is a list of indices and the output is corresponding word embeddings

    def forward(self, idx, targets=None):
        # idx is the input encoded input text. it is xb
        # targets are yb
        B, T = idx.shape
        # idx and targets are both (B, T) tensors of integers

        tok_emb = self.token_embedding_table(idx)  # (B, T, C). the embedding table maps an index value to a weight matrix of a certain dim
        # in the embedding table key is the ch index and value is the ch vector

        # pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        # x = tok_emb + pos_emb # (B, T, C) x holds token identifies AND the positions at which tokens occur
        x = self.position_embedding_table(tok_emb) # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        # go from token embeddings to logits
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        # logits are vector of raw non-normalized predictions that a classification model generates
        # logits are basically the scores for the next character in the sequence

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # the non-normalized predictions
            targets = targets.view(B * T)  # actual values
            loss = F.cross_entropy(logits, targets)  # measures quality of logits wrt targets

        return logits, loss

    # continues generation for all the batch dims along the time dims
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
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


# average losses over multiple batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "test"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = BigramLanguageModel()
m = model.to(device)
# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3) # 3e-4 is typically a good setting for a learning rate, but with smaller networks you can get away with higher

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, test loss {losses['test']:.4f}")
    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True) # zero out gradient from previous step
    loss.backward() # getting gradients for all parameters (weights)
    optimizer.step() # use gradients to update parameters


# generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
generated_idx = m.generate(idx=context, max_new_tokens=500)[0].tolist()
print(f"genid:\n{generated_idx}")
torch_gen = torch.tensor(generated_idx)
print(bpe_tokenizer.decode(torch_gen))