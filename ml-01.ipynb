import torch, torch.nn as nn, torch.nn.functional as F

# 1. Fake data: vocab and a tiny sentence "alice was beginning to get very tired"
vocab = {"alice":0, "was":1, "beginning":2, "to":3, "get":4, "very":5, "tired":6}
context, target = torch.tensor([0,1]), torch.tensor([2])  # "alice was" -> "beginning"

# 2. Tiny model: embed words, average them, predict next word
model = nn.Sequential(
    nn.Embedding(len(vocab), 10),
    lambda x: x.mean(dim=0),
    nn.Linear(10, len(vocab))
)

# 3. One training step
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
optimizer.zero_grad()
loss = F.cross_entropy(model(context).unsqueeze(0), target)
loss.backward()
optimizer.step()

print(f"Loss after one step: {loss.item():.4f}")
