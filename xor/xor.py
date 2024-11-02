import torch
import torch.optim as optim
import torch.nn as nn

inputs = (
    (0, 0),
    (0, 1),
    (1, 1),
    (1, 0)
)

class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.l1 = nn.Linear(2, 5)
        self.l2 = nn.Linear(5, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.sigmoid(self.l1(x))
        return torch.sigmoid(self.l2(x))
    
model = Model()
optimizer = optim.SGD(model.parameters(), lr=0.1)
lossFunc = nn.MSELoss()
EPOCHS = 3000

for i in range(EPOCHS):
    for input in inputs:
        pred = model.forward(torch.Tensor(input))
        truth = torch.Tensor([input[0] ^ input[1]])
        loss = lossFunc(pred, truth)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if i % 100 == 0: print(f"epoch: {i}, loss: {loss.item()}")

for input in inputs:
    pred = model.forward(torch.Tensor(input))
    print(f"{input} -> pred({pred.item()}), truth({input[0] ^ input[1]})")