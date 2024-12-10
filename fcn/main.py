import torch.nn as nn
import torchvision.models as models
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

CLASSES = 20
color_map = [
    (1.0, 0.0, 0.0),    # Red
    (0.0, 1.0, 0.0),    # Green
    (0.0, 0.0, 1.0),    # Blue
    (1.0, 1.0, 0.0),    # Yellow
    (1.0, 0.0, 1.0),    # Magenta
    (0.0, 1.0, 1.0),    # Cyan
    (0.5, 0.5, 0.0),    # Olive
    (0.5, 0.0, 0.5),    # Purple
    (0.0, 0.5, 0.5),    # Teal
    (0.5, 0.5, 0.5),    # Gray
    (1.0, 0.5, 0.0),    # Orange
    (0.5, 1.0, 0.0),    # Lime Green
    (0.0, 0.5, 1.0),    # Sky Blue
    (1.0, 0.0, 0.5),    # Pink
    (0.5, 0.0, 1.0),    # Deep Purple
    (0.0, 1.0, 0.5),    # Aquamarine
    (0.75, 0.75, 0.0),  # Mustard Yellow
    (0.75, 0.0, 0.75),  # Violet
    (0.0, 0.75, 0.75),  # Turquoise
    (0.25, 0.25, 0.25)  # Dark Gray
]

class DataLoader:
    #128x256
    def __init__(self, path, size):
        self.path = path
        self.data = [None] * size
        self.size = size
    def __getitem__(self, index):
        with torch.no_grad():
            if self.data[index]: return self.data[index]

            npimg = np.moveaxis(np.load(f"{self.path}/image/{index}.npy"), -1, 0)
            image = torch.from_numpy(npimg)

            npsem = np.moveaxis(np.load(f"{self.path}/label/{index}.npy"), -1, 0)
            sem = torch.from_numpy(npsem)
            semTens = torch.zeros((CLASSES, 256, 128))

            for i in range(CLASSES): semTens[i] = (sem == i)

            self.data[index] = {"img": image.float().unsqueeze(0), "sem": semTens.float().unsqueeze(0)}
            return self.data[index]

class FCN(nn.Module):
    def __init__(self, preModel):
        super(FCN, self).__init__()
        self.preModel = nn.Sequential(
            preModel.features,
            preModel.avgpool
        )
        self.model = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(
                in_channels=32, 
                out_channels=20, 
                kernel_size=(4, 4),
                stride=(3, 3),
                padding=(1, 1)
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(20),
            nn.AdaptiveAvgPool2d((256, 128)),
        )
    def forward(self, x):
        x = self.preModel.forward(x)
        return self.model.forward(x)

def showImageAndSem(img, sem):
    semImg = np.zeros((128, 256 ,3))
    semTen = sem.permute(0, 2, 1).detach().numpy()
    for i in range(CLASSES):
        r,g,b = color_map[i]
        semImg[..., 0] += semTen[i] * r
        semImg[..., 1] += semTen[i] * g
        semImg[..., 2] += semTen[i] * b
    plt.subplot(1,3,1)
    plt.imshow(img.permute(2, 3, 1, 0).squeeze(3))
    plt.subplot(1,3,2)
    plt.imshow(semImg)
    plt.show()

def test(model):
    assert os.path.exists("model.pt")
    testData = DataLoader("../data/fcn/val", 500)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        alexnet = models.alexnet(weights = models.AlexNet_Weights.DEFAULT)
        model = FCN(alexnet)
        model.load_state_dict(torch.load("model.pt", weights_only=True))
    model.to(device)
    with torch.no_grad():
        for i in range(testData.size):
            input, truth = testData[i]["img"], testData[i]["sem"]
            preds = model(input.to(device))
            preds = preds.cpu()
            showImageAndSem(input, preds.squeeze(0))

def main():
    trainData = DataLoader("../data/fcn/train", 2975)
    alexnet = models.alexnet(weights = models.AlexNet_Weights.DEFAULT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)
    model = FCN(alexnet)
    if os.path.exists("model.pt"): model.load_state_dict(torch.load("model.pt", weights_only=True))
    model.to(device)
    lossFunc = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    EPOCH = 100
    runningLoss = 0
    try:
        for epoch in range(EPOCH):
            for i in range(trainData.size):
                input, truth = trainData[i]["img"], trainData[i]["sem"]
                optim.zero_grad()
                preds = model(input.to(device))
                preds = preds.cpu()
                loss = lossFunc(preds, truth)
                loss.backward()
                optim.step()
                runningLoss += loss.item()
                if i % 100 == 99:
                    print(f"running_loss({i} @ {epoch}): {runningLoss/100}")
                    runningLoss = 0
    except KeyboardInterrupt: print("Exiting training...")
    finally: torch.save(model.state_dict(), "model.pt")
    test(model)

if __name__ == "__main__": main()