import numpy as np

def getReward(prob, n=10):
    reward = 0
    for _ in range(n):
        if np.random.random() < prob: reward += 1
    return reward
def getBestArm(estimate):
    best = 0
    index = 0
    for i in range(10):
        if estimate[i][0] > best:
            best = estimate[i][0]
            index = i
    return index
def updateEstimate(cur, index, estimate):
    run = estimate[index][1] 
    estimate[index][0] = (estimate[index][0] * run + cur) / (run+1) 
    estimate[index][1] += 1

probs = np.random.random(10)
estimate = [[0,0] for _ in range(10)]
eps = 0.2
rewards = []
for i in range(500):
    if np.random.random() > eps: choice = getBestArm(estimate)
    else: choice = np.random.randint(10)
    x = getReward(probs[choice])
    updateEstimate(x, choice, estimate)
    rewards.append(x)

print(np.average(rewards))
print(estimate)
print(probs)
