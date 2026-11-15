"""
Mini deep-learning course: from one training step to a tiny GPT.
Stages 0-7 build up; Stage 8 lands on a Transformer LM.

Run:  python course.py
Deps: torch only.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0)


# ============================================================
# STAGE 0 — Where we start: one example, one step.
# ============================================================
print("\n=== STAGE 0: the original one-step snippet ===")

vocab0 = {"alice":0, "was":1, "beginning":2, "to":3, "get":4, "very":5, "tired":6}
context0, target0 = torch.tensor([0,1]), torch.tensor([2])

class MeanPool0(nn.Module):
    def forward(self, x): return x.mean(dim=0)

model0 = nn.Sequential(
    nn.Embedding(len(vocab0), 10),
    MeanPool0(),
    nn.Linear(10, len(vocab0)),
)
opt0 = torch.optim.SGD(model0.parameters(), lr=0.1)
opt0.zero_grad()
loss0 = F.cross_entropy(model0(context0).unsqueeze(0), target0)
loss0.backward()
opt0.step()
print(f"Loss after one step: {loss0.item():.4f}  (nothing was really learned)")


# ============================================================
# STAGE 1 — A real dataset: sliding windows over a corpus.
# ============================================================
print("\n=== STAGE 1: build (X, Y) windows from a corpus ===")

sentence = "alice was beginning to get very tired".split()
vocab = {w: i for i, w in enumerate(sorted(set(sentence)))}
inv_vocab = {i: w for w, i in vocab.items()}
ids = torch.tensor([vocab[w] for w in sentence])
V = len(vocab)
print("vocab:", vocab)

N_CTX = 2

def make_windows(ids, n_ctx):
    X, Y = [], []
    for i in range(len(ids) - n_ctx):
        X.append(ids[i:i+n_ctx])
        Y.append(ids[i+n_ctx])
    return torch.stack(X), torch.stack(Y)

X, Y = make_windows(ids, N_CTX)
print("X:", X.tolist(), "\nY:", Y.tolist())


# ============================================================
# STAGE 2 — Batch-aware model: mean-pool over time, not batch.
# ============================================================
print("\n=== STAGE 2: batch-aware MeanPool (B, T, D) -> (B, D) ===")

class MeanPool(nn.Module):
    def forward(self, x):          # (B, T, D)
        return x.mean(dim=1)       # (B, D)

model2 = nn.Sequential(
    nn.Embedding(V, 16),
    MeanPool(),
    nn.Linear(16, V),
)
logits = model2(X)
print("logits shape:", tuple(logits.shape))   # (B, V)


# ============================================================
# STAGE 3 — Real training loop on the mean-pool model.
# ============================================================
print("\n=== STAGE 3: train the mean-pool model ===")

opt3 = torch.optim.SGD(model2.parameters(), lr=0.1)
for epoch in range(200):
    opt3.zero_grad()
    loss = F.cross_entropy(model2(X), Y)
    loss.backward()
    opt3.step()
    if epoch % 40 == 0:
        print(f"epoch {epoch:3d}  loss {loss.item():.4f}")
print(f"final loss {loss.item():.4f}")
print(">> Note: mean pooling is order-blind. 'alice was' and 'was alice' look identical.")


# ============================================================
# STAGE 4 — Order-aware: tiny RNN over the sequence.
# ============================================================
print("\n=== STAGE 4: swap mean-pool for a tiny RNN ===")

class RNNModel(nn.Module):
    def __init__(self, V, D, H):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.rnn = nn.RNN(D, H, batch_first=True)
        self.head = nn.Linear(H, V)
    def forward(self, x):                 # (B, T)
        e = self.emb(x)                   # (B, T, D)
        out, _ = self.rnn(e)              # (B, T, H)
        return self.head(out[:, -1])      # (B, V)

model4 = RNNModel(V, D=16, H=32)
opt4 = torch.optim.Adam(model4.parameters(), lr=1e-2)
for epoch in range(200):
    opt4.zero_grad()
    loss = F.cross_entropy(model4(X), Y)
    loss.backward()
    opt4.step()
    if epoch % 40 == 0:
        print(f"epoch {epoch:3d}  loss {loss.item():.4f}")
print(f"final loss {loss.item():.4f}")


# ============================================================
# STAGE 5 — Standard toolkit: DataLoader, dropout, train/eval.
# ============================================================
print("\n=== STAGE 5: DataLoader + dropout + train/eval modes ===")

class DropoutRNN(nn.Module):
    def __init__(self, V, D, H, p=0.1):
        super().__init__()
        self.emb  = nn.Embedding(V, D)
        self.rnn  = nn.RNN(D, H, batch_first=True)
        self.drop = nn.Dropout(p)
        self.head = nn.Linear(H, V)
    def forward(self, x):
        e = self.emb(x)
        out, _ = self.rnn(e)
        return self.head(self.drop(out[:, -1]))

model5 = DropoutRNN(V, D=16, H=32)
opt5 = torch.optim.Adam(model5.parameters(), lr=1e-2)
loader = DataLoader(TensorDataset(X, Y), batch_size=2, shuffle=True)

for epoch in range(200):
    model5.train()
    for xb, yb in loader:
        opt5.zero_grad()
        loss = F.cross_entropy(model5(xb), yb)
        loss.backward()
        opt5.step()
    if epoch % 40 == 0:
        model5.eval()
        with torch.no_grad():
            full = F.cross_entropy(model5(X), Y)
        print(f"epoch {epoch:3d}  loss {full.item():.4f}")


# ============================================================
# STAGE 6 — Generation: sample from the trained model.
# ============================================================
print("\n=== STAGE 6: sample from the model ===")

@torch.no_grad()
def generate(model, start, n_new, temperature=1.0):
    model.eval()
    ctx = start.clone()
    out = start.tolist()
    for _ in range(n_new):
        logits = model(ctx.unsqueeze(0))[0] / temperature
        probs  = F.softmax(logits, dim=-1)
        nxt    = torch.multinomial(probs, 1)
        out.append(nxt.item())
        ctx = torch.cat([ctx[1:], nxt])       # slide the context window
    return " ".join(inv_vocab[i] for i in out)

print("greedy-ish sample:", generate(model5, ids[:N_CTX], n_new=5, temperature=0.5))
print("warmer sample:    ", generate(model5, ids[:N_CTX], n_new=5, temperature=1.0))


# ============================================================
# STAGE 7 — Train/val split: the generalization gap.
# ============================================================
print("\n=== STAGE 7: train/val split ===")

Xtr, Ytr = X[:4], Y[:4]
Xva, Yva = X[4:], Y[4:]

model7 = RNNModel(V, D=16, H=32)
opt7 = torch.optim.Adam(model7.parameters(), lr=1e-2)
for epoch in range(300):
    model7.train(); opt7.zero_grad()
    loss = F.cross_entropy(model7(Xtr), Ytr)
    loss.backward(); opt7.step()
    if epoch % 60 == 0:
        model7.eval()
        with torch.no_grad():
            val = F.cross_entropy(model7(Xva), Yva)
        print(f"epoch {epoch:3d}  train {loss.item():.4f}  val {val.item():.4f}")
print(">> With only 6 examples, the train/val gap is mostly noise — data size matters.")


# ============================================================
# STAGE 8 — Minimal GPT: causal Transformer LM.
# ============================================================
print("\n=== STAGE 8: minimal GPT (causal Transformer) ===")

class CausalSelfAttention(nn.Module):
    def __init__(self, D, n_head):
        super().__init__()
        assert D % n_head == 0
        self.n_head = n_head
        self.qkv  = nn.Linear(D, 3 * D)
        self.proj = nn.Linear(D, D)
    def forward(self, x):                       # (B, T, D)
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)   # each (B, T, D)
        # reshape into heads: (B, n_head, T, D/n_head)
        hd = D // self.n_head
        def split_heads(t): return t.view(B, T, self.n_head, hd).transpose(1, 2)
        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)      # (B, H, T, T)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = att @ v                                        # (B, H, T, hd)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)

class Block(nn.Module):
    def __init__(self, D, n_head):
        super().__init__()
        self.ln1  = nn.LayerNorm(D)
        self.attn = CausalSelfAttention(D, n_head)
        self.ln2  = nn.LayerNorm(D)
        self.mlp  = nn.Sequential(nn.Linear(D, 4*D), nn.GELU(), nn.Linear(4*D, D))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))          # residual + pre-norm
        x = x + self.mlp(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, V, D, n_head, n_layer, n_ctx):
        super().__init__()
        self.n_ctx = n_ctx
        self.tok_emb = nn.Embedding(V, D)
        self.pos_emb = nn.Embedding(n_ctx, D)
        self.blocks  = nn.Sequential(*[Block(D, n_head) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(D)
        self.head    = nn.Linear(D, V)
    def forward(self, x):                       # (B, T) with T <= n_ctx
        B, T = x.shape
        pos  = torch.arange(T, device=x.device)
        h    = self.tok_emb(x) + self.pos_emb(pos)
        h    = self.blocks(h)
        return self.head(self.ln_f(h))          # (B, T, V)

# Teacher-forcing setup: X is (B, T) input, Y is (B, T) shifted targets.
def make_lm_batch(ids, n_ctx):
    Xs, Ys = [], []
    for i in range(len(ids) - n_ctx):
        Xs.append(ids[i:i+n_ctx])
        Ys.append(ids[i+1:i+1+n_ctx])
    return torch.stack(Xs), torch.stack(Ys)

X8, Y8 = make_lm_batch(ids, N_CTX)
print("X8:", X8.tolist(), "\nY8:", Y8.tolist())   # each row is shifted-by-one

gpt = MiniGPT(V=V, D=32, n_head=2, n_layer=2, n_ctx=N_CTX)
opt8 = torch.optim.Adam(gpt.parameters(), lr=3e-3)

for epoch in range(300):
    gpt.train(); opt8.zero_grad()
    logits = gpt(X8)                                   # (B, T, V)
    loss = F.cross_entropy(logits.reshape(-1, V), Y8.reshape(-1))
    loss.backward(); opt8.step()
    if epoch % 60 == 0:
        print(f"epoch {epoch:3d}  loss {loss.item():.4f}")
print(f"final loss {loss.item():.4f}")

@torch.no_grad()
def gpt_generate(model, start, n_new, temperature=1.0):
    model.eval()
    ctx = start.clone()
    out = start.tolist()
    for _ in range(n_new):
        T = ctx.shape[0]
        logits = model(ctx.unsqueeze(0))[0, T-1] / temperature
        probs  = F.softmax(logits, dim=-1)
        nxt    = torch.multinomial(probs, 1)
        out.append(nxt.item())
        ctx = torch.cat([ctx, nxt])[-model.n_ctx:]       # respect context length
    return " ".join(inv_vocab[i] for i in out)

print("GPT sample:", gpt_generate(gpt, ids[:N_CTX], n_new=5, temperature=0.7))

print("\nDone. You've gone from one SGD step to a working causal Transformer LM.")
