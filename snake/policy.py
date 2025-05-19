import torch
from Snake import SnakeGame

EPISODES = 800
BOARD_X = 4
BOARD_Y = 4
EPSILON = 0.7
GAMMA = 0.99

ACTION_NAMES = [
        "w",
        "d",
        "s",
        "a"
        ]

policy = torch.nn.Sequential(
    torch.nn.Linear(BOARD_X*BOARD_Y*3, 150),
    torch.nn.ReLU(inplace=True),
    torch.nn.Linear(150, 100),
    torch.nn.ReLU(inplace=True),
    torch.nn.Linear(100, 4),
)
optim  = torch.optim.Adam(policy.parameters(), 1e-3)

def createQFuncInp(game) -> torch.Tensor:
    food = torch.tensor(game.board == game.foodVal, dtype=torch.float)
    body = torch.tensor(game.board == game.bodyVal, dtype=torch.float)
    head = torch.tensor(game.board == game.headVal, dtype=torch.float)
    state = torch.stack([food, body, head])
    return (state + torch.randn_like(state)*0.1).flatten()
def discountRewards(rewards: list) -> torch.Tensor:
    rewardsT = torch.tensor(rewards[::-1])
    tens = torch.pow(GAMMA, torch.arange(len(rewards))) * rewardsT
    return torch.flip(tens, [0]) / abs(torch.max(tens))
def lossFn(predHigh: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    return -1 * torch.sum(rewards * torch.log(predHigh))

for episode in range(10):
    game = SnakeGame(BOARD_X, BOARD_Y)
    states, predHigh, rewards = [], [], []
    print("######################START##########################")
    while True:
        state = createQFuncInp(game)
        preds = policy(state)
        action = torch.argmax(preds)
        game.display()
        _, reward, gameOver, score = game.makeMove(action)
        if reward == 0: reward = -1
        elif reward == 1: reward = 10
        else: reward = -10
        print(f"action: {ACTION_NAMES[action]}, reward: {reward}")
        states.append(state)
        predHigh.append(preds[action])
        rewards.append(reward)
        if gameOver: break
    rewards = discountRewards(rewards)
    print(f"discounted_rewards: {rewards}")
    loss = lossFn(torch.stack(predHigh), rewards)
    optim.zero_grad()
    loss.backward()
    optim.step()
