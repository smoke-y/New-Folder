import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import os.path

class MobileNet(nn.Module):
    def __init__(self, channelIn: int, outClasses: int) -> None:
        super(MobileNet, self).__init__()
        def convDepth(input: int, output: int, stride: int) -> nn.Sequential:
            return nn.Sequential(
                # NOTE: groups=input. This creates a filter for each layer and concatinates the output
                nn.Conv2d(input, input, 3, stride, groups=input, bias=False),
                nn.BatchNorm2d(input),
                nn.ReLU(inplace=True),

                # NOTE: we run a 1x1xinput kernel to get the final output. Since the output is already concatinated, we use a Conv2d of kernel_size=1
                nn.Conv2d(input, output, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(output),
                nn.ReLU(inplace=True)
            )
        
        self.model = nn.Sequential(
            nn.Conv2d(channelIn, 32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            convDepth(32, 64, 1),
            convDepth(64, 128, 1),
            convDepth(128, 128, 1),
            convDepth(128, 256, 2),
            convDepth(256, 256, 1),
            convDepth(256, 512, 2),
            convDepth(512, 512, 1),
            convDepth(512, 512, 1),
            convDepth(512, 512, 1),
            convDepth(512, 512, 1),
            convDepth(512, 512, 1),
            convDepth(512, 1024, 2),
            convDepth(1024, 1024, 1),
            # pool 1 from 1024x1024
            nn.AdaptiveAvgPool2d(1),
        )
        self.net = nn.Linear(1024, outClasses)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        #[batch_size, 1024, 1, 1] -> [batch_size, 1024]
        return self.net(x.view(-1, 1024))

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root="../data", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root="../data", train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)
    model = MobileNet(channelIn=3, outClasses=len(classes))
    if os.path.exists("model.pt"): model.load_state_dict(torch.load("model.pt", weights_only=True))
    model.to(device)
    lossFunc = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    EPOCH = 10
    runningLoss = 0
    try:
        for epoch in range(EPOCH):
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                optim.zero_grad()
                preds = model(inputs.to(device))
                preds = preds.cpu()
                loss = lossFunc(preds, labels)
                loss.backward()
                optim.step()
                runningLoss += loss.item()
                if i % 100 == 99:
                    print(f"running_loss({i} @ {epoch}): {runningLoss/100}")
                    runningLoss = 0
    except KeyboardInterrupt: print("Exiting training...")
    finally: torch.save(model.state_dict(), "model.pt")
    for i, data in enumerate(testloader, 0):
        with torch.no_grad():
            correct = 0
            inputs, labels = data
            preds = model(inputs.to(device))
            preds = preds.cpu()
            for j in range(len(labels)):
                if labels[j] == torch.argmax(preds[j]): correct += 1
            print(f"running_accuracy({i}): {(correct/len(labels)) * 100}%")

if __name__ == "__main__": main()