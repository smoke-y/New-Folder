import numpy as np
import torch
import torch.nn as nn

def getReward(prob, n=10):
    reward = 0
    for _ in range(n):
        if np.random.random() < prob: reward += 1
    return reward
def getBestArm(estimate):
    best = 0
    index = 0
    for i in range(10):
        if estimate[i] > best:
            best = estimate[i]
            index = i
    return index

armMatrix = np.random.rand(10, 10)

model = torch.nn.Sequential(
    nn.Linear(10, 100),
    nn.ReLU(True),
    nn.Linear(100, 10),
    nn.ReLU(True)
)
lossfn = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(),lr=1e-2)

for i in range(5000):
    oneHotVec = np.zeros(10)
    state = np.random.randint(10)
    oneHotVec[state] = 1.0
    y = model(torch.tensor(oneHotVec, dtype=torch.float32))
    probs = nn.functional.softmax(y)
    choice = np.random.choice(10, p=probs.detach().numpy())
    reward = getReward(armMatrix[state][choice])
    oneHotVec = y.data.numpy().copy()
    oneHotVec[choice] = reward
    loss = lossfn(torch.tensor(oneHotVec), y)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(loss.detach().numpy())
