import torch

from tinygpt import TinyGPT, get_batch
from bpe import BPETokenizer
from configs import TinyGPTConfig, OptimizerConfig

config = TinyGPTConfig()
opt_config = OptimizerConfig()
model = TinyGPT(config)
m = model.to(config.device)
# create a PyTorch optimizer
# optimizer = create_optimizer(m, opt_config)
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# average losses over multiple batches
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "test"]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

for iter in range(config.max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % config.eval_interval == 0:
        losses = estimate_loss(m)
        print(f"step {iter}: train loss {losses['train']:.4f}, test loss {losses['test']:.4f}")
    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True) # zero out gradient from previous step
    loss.backward() # getting gradients for all parameters (weights)
    optimizer.step() # use gradients to update parameters


# generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=config.device)
generated_idx = m.generate(idx=context, max_new_tokens=500)[0].tolist()
print(f"genid:\n{generated_idx}")
torch_gen = torch.tensor(generated_idx)
bpe_tokenizer = BPETokenizer()
print(bpe_tokenizer.decode(torch_gen))

