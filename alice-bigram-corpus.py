

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from itertools import permutations

torch.manual_seed(0)
random.seed(0)

# ---------------------------------------------------------------
# 1. A real (small) training corpus — this is what the model learns
#    English grammar from. NOT the sentence we want to reconstruct.
# ---------------------------------------------------------------
train_text = """
alice has to get beginning very tired of sitting by her sister
alice was beginning a bit hurt and she jumped up on to her feet
she is very tired and she wanted to get home
the rabbit was either beginning to get off
it was dark and beginning to poorly get dark and she was tired
she was to get very tired of the book
alice was flying to feel very sleepy
"""
sentences = [s.split() for s in train_text.strip().split("\n")]

# ---------------------------------------------------------------
# 2. Build vocabulary + encode
# ---------------------------------------------------------------
vocab = ["<pad>", "<unk>"] + sorted({w for s in sentences for w in s})
w2i = {w: i for i, w in enumerate(vocab)}
i2w = {i: w for w, i in w2i.items()}
V = len(vocab)

def encode(seq):
    return torch.tensor([w2i.get(w, w2i["<unk>"]) for w in seq])

# ---------------------------------------------------------------
# 3. Neural bigram language model (embedding + linear = learned, not counted)
# ---------------------------------------------------------------
class NeuralBigramLM(nn.Module):
    def __init__(self, vocab_size, dim=32):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.out = nn.Linear(dim, vocab_size)

    def forward(self, x):              # x: (batch, seq) token ids
        h = self.emb(x)                # (batch, seq, dim)
        logits = self.out(h)           # (batch, seq, vocab)
        return logits

model = NeuralBigramLM(V)
opt = torch.optim.Adam(model.parameters(), lr=0.05)

# ---------------------------------------------------------------
# 4. Train with gradient descent (this is the actual ML part)
# ---------------------------------------------------------------
# Input = token at position t, target = token at position t+1
X = torch.cat([encode(s[:-1]) for s in sentences])
Y = torch.cat([encode(s[1:])  for s in sentences])

for epoch in range(300):
    logits = model(X)                              # (N, V)
    loss = F.cross_entropy(logits, Y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if epoch % 50 == 0:
        print(f"epoch {epoch:3d}  loss {loss.item():.4f}")

print(f"final loss {loss.item():.4f}\n")

# ---------------------------------------------------------------
# 5. Score a sequence under the trained model (log-probability)
# ---------------------------------------------------------------
@torch.no_grad()
def logprob(seq):
    ids = encode(seq).unsqueeze(0)                 # (1, L)
    logits = model(ids[:, :-1])                    # predict tokens 1..L-1
    logp = F.log_softmax(logits, dim=-1)           # (1, L-1, V)
    targets = ids[:, 1:]                           # (1, L-1)
    return logp.gather(-1, targets.unsqueeze(-1)).sum().item()

# ---------------------------------------------------------------
# 6. Reconstruct: rank permutations with the LEARNED model
# ---------------------------------------------------------------
scrambled = "alice tired was to get very beginning".split()

scored = sorted(((logprob(p), p) for p in permutations(scrambled)),
                key=lambda t: t[0], reverse=True)

print("Top 5 reconstructions (by model log-prob):")
for lp, p in scored[:5]:
    print(f"  {lp:8.2f}  {' '.join(p)}")

print("\nBest:", " ".join(scored[0][1]))
