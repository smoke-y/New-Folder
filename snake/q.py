import torch
import numpy as np
from Snake import SnakeGame
from collections import deque
from random import random, sample

EPOCH = 800
BOARD_X = 4
BOARD_Y = 4
EPSILON = 0.7
GAMMA = 0.9
MEM_SIZE = 1000
BATCH_SIZE = 100 

ACTION_NAMES = [
        "w",
        "d",
        "s",
        "a"
        ]

Q = torch.nn.Sequential(
    torch.nn.Linear(BOARD_X*BOARD_Y*3, 150),
    torch.nn.ReLU(inplace=True),
    torch.nn.Linear(150, 100),
    torch.nn.ReLU(inplace=True),
    torch.nn.Linear(100, 4),
)
lossFn = torch.nn.MSELoss()
optim  = torch.optim.Adam(Q.parameters(), 1e-3)

def createQFuncInp(game) -> torch.Tensor:
    food = torch.tensor(game.board == game.foodVal, dtype=torch.float)
    body = torch.tensor(game.board == game.bodyVal, dtype=torch.float)
    head = torch.tensor(game.board == game.headVal, dtype=torch.float)
    state = torch.stack([food, body, head])
    return (state + torch.randn_like(state)*0.1).flatten()

expBuff = deque(maxlen=MEM_SIZE)

for epoch in range(EPOCH):
    game = SnakeGame(BOARD_X, BOARD_Y)
    while True:
        state = createQFuncInp(game)
        qvals = Q(state)
        if random() > 0.7: action = np.random.randint(0, 4)
        else:
            action = torch.argmax(qvals)
        _, reward, gameOver, score = game.makeMove(action)
        exp = [state, torch.tensor(action) if type(action) != torch.Tensor else action, torch.tensor(reward)]
        if reward != -2:
            with torch.no_grad():
                newState = createQFuncInp(game)
                newQvals = Q(newState)
            exp.append(newState)
        else: exp.append(torch.zeros(BOARD_X*BOARD_Y*3))
        exp.append(torch.tensor(gameOver))
        expBuff.append(exp)
        if len(expBuff) >= BATCH_SIZE:
            exps = sample(expBuff, BATCH_SIZE)
            states, actions, rewards, nextStates, dones = zip(*exps)
            dones = 1-torch.stack(dones).float().unsqueeze(1)
            states = torch.stack(states)
            actions = torch.stack(actions).unsqueeze(1)
            rewards = torch.stack(rewards).unsqueeze(1)
            nextStates = torch.stack(nextStates)
            Q1 = Q(states)
            with torch.no_grad(): Q2 = Q(nextStates)
            X = rewards + dones*GAMMA*Q2.gather(1, actions)
            Y = Q1.gather(dim=1, index=actions.long())
            loss = lossFn(X.detach(), Y)
            optim.zero_grad()
            loss.backward()
            optim.step()
        if gameOver: break

game = SnakeGame(BOARD_X, BOARD_Y)
while True:
    game.display()
    state = createQFuncInp(game)
    qvals = Q(state)
    action = torch.argmax(qvals)
    _, reward, gameOver, score = game.makeMove(action)
    if gameOver: break
