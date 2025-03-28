import torch
import torch.nn as nn
import numpy as np
import gym

class ActorCrit(nn.Module):
    def __init__(self):
        self.base = nn.Sequential(
                nn.Linear(4, 25),
                nn.ReLU(True),
                nn.Linear(25, 50),
                nn.ReLU(True),
                )
        self.actor = nn.Linear(50, 2)
        self.beforeCrit = nn.Linear(50, 25)
        self.critic = nn.Linear(25, 1)
    def forward(self, x):
        x = nn.functional.normalize(x)
        x = self.base(x)
        z = nn.functional.log_softmax(self.actor(x))
        y = nn.functional.relu(self.beforeCrit(x.detach()))
        y = self.critic(y)
        return z, y

model = ActorCrit()

env = gym.make("CartPole-v1")
optim = torch.optim.Adam(model.parameters())
for _ in range(1000):
    state = env.reset()[0]
    values = []
    while True:
        policy, actionProbs = model(torch.tensor(state, dtype=torch.float32))
        values.append(actionProbs)
        action = policy.view(-1)[torch.distributions.Categorical(logits=actionProbs.view(-1))]

