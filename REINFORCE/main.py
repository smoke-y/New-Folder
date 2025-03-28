import gym 
import torch
import torch.nn as nn
import numpy as np

env = gym.make("CartPole-v1")

model = nn.Sequential(
        nn.Linear(4, 150),
        nn.LeakyReLU(),
        nn.Linear(150, 2),
        nn.Softmax(dim=0),
        )
optim = torch.optim.Adam(model.parameters(), lr = 0.009)

def discountReward(rewards, gamma=0.99):
    lenr = len(rewards)
    disc_return = torch.pow(gamma,torch.arange(lenr).float()) * rewards
    disc_return /= disc_return.max()
    return disc_return
def lossfn(rewards, actions):
    return -1 * torch.sum(rewards * torch.log(actions))


for i in range(1000):
    states = []
    rewards = []
    actions = []
    state = env.reset()[0]
    for j in range(200):
        probs = model(torch.tensor(state, dtype=torch.float32))
        action = np.random.choice(np.array([0,1]), p=probs.data.numpy())
        nextState, reward, done, info, _ = env.step(action)
        states.append(state)
        rewards.append(j + 1)
        actions.append(action)
        state = nextState
        if done: break
    print(len(actions))
    probs = model(torch.tensor(np.array(states), dtype=torch.float32))
    discRewards = discountReward(np.array(rewards))
    probs = probs.gather(dim=1, index=torch.tensor(actions, dtype=torch.long).view(-1, 1))
    loss = lossfn(discRewards.squeeze(), probs)
    optim.zero_grad()
    loss.backward()
    optim.step()
