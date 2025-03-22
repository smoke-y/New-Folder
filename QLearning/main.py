from gridworld import *
import torch
import torch.nn as nn
import numpy as np
from collections import deque

actionset = {
        0: 'u',
        1: 'd',
        2: 'l',
        3: 'r',
        }

model = nn.Sequential(
        nn.Linear(4*4*4, 150),
        nn.ReLU(True),
        nn.Linear(150, 100),
        nn.ReLU(True),
        nn.Linear(100, 4),
        )
device = "cuda" 
model = model.to(device)
lossfn = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
size = 1000
buff = deque(maxlen = size)
batch = 200

def getState(game): return torch.tensor(game.board.render_np().reshape(1, 4*4*4) + np.random.rand(1,64)/10.0, dtype=torch.float32).to(device)

for _ in range(5000):
    game = Gridworld(size=4, mode="dynamic")
    moves = 0
    while True:
        moves += 1
        state = getState(game)
        qval = model(state)
        if np.random.random() > 0.2:
            choice = torch.argmax(qval).detach().cpu().numpy()
        else: choice = np.random.randint(4)
        action = actionset[int(choice)]
        game.makeMove(action)
        reward = game.reward()
        buff.append([state, int(choice), reward, getState(game), True if reward > 0 else False])
        if len(buff) >= batch:
            minibatch = random.sample(buff, batch)
            states = []
            actions = []
            rewards = []
            nextStates = []
            done = []
            for i in range(batch):
                states.append(minibatch[i][0])
                actions.append(minibatch[i][1])
                rewards.append(minibatch[i][2])
                nextStates.append(minibatch[i][3])
                done.append(minibatch[i][4])
            states = torch.cat(states)
            actions = torch.tensor(actions, dtype=torch.float32).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            nextStates = torch.cat(nextStates)
            done = torch.tensor(done, dtype=torch.float32).to(device)
            Q1 = model(states)
            with torch.no_grad():
                Q2 = model(nextStates)
                maxQ = torch.max(Q2, dim=1)[0]
            Y = rewards + 0.9 * (1-done) * maxQ 
            X = Q1.gather(dim = 1, index = actions.long().unsqueeze(dim=1)).squeeze()
            loss = lossfn(X, Y)
            optim.zero_grad()
            loss.backward()
            optim.step()
        if reward > -1 or moves > 10: break

game = Gridworld(size=4, mode="dynamic")
for _ in range(10):
    state = getState(game) 
    qval = model(state)
    choice = torch.argmax(qval).detach().cpu().numpy()
    action = actionset[int(choice)]
    print(action)
    game.makeMove(action)
    print(game.display())
    if game.reward() == 10: break
