import torch
import torch.nn as nn
from torch.nn import functional as F

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

with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# all unique chars that occur in the text
chars = sorted(set(text))
vocab_size = len(chars)
# create mapping from chars to ints
# this is the tokenization at the character level
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of ints
decode = lambda l: "".join(itos[i] for i in l) # decoder: take a list of ints, output a string

# train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)) # first 90% is train and the rest test
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))  # random offsets in the dataset
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# average losses over multiple batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
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
    """multiple heads of self-attention in parallel"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # nn.ModuleList is like a python list, so this is creating num head modules
        self.proj = nn.Linear(n_embed, n_embed) # to make linear the inputs
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concatenating outputs over channel dim
        out = self.proj(out) # project is the linear outcome of the above concatenation
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed), # project layer going back into the residual pathway
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size) # communication
        self.ffwd = FeedForward(n_embed) # computation on each token
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # x adds residual connections
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential( # so that we are interspersing communication and computation
            *[Block(n_embed, n_head=n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed) # at the end fof the transformer and right before final linear layer you need a layer norm
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

        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C) x holds token identifies AND the positions at which tokens occur
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

model = BigramLanguageModel()
m = model.to(device)
# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3) # 3e-4 is typically a good setting for a learning rate, but with smaller networks you can get away with higher

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
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
print(decode(generated_idx))
