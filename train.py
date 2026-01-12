import torch
from data import generate_graph
from model import GNN

G, labels = generate_graph()
features = torch.randn(len(G.nodes()), 4)
y = torch.tensor(list(labels.values()))

model = GNN(4, 16)
opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

for i in range(50):
    out = model(features)
    loss = loss_fn(out, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

print("Training complete")
