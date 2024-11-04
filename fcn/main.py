import torch.nn as nn
import torchvision.models as models
import torch
import numpy as np
import os
import matplotlib.pyplot as plt

class DataLoader:
    #128x256
    def __init__(self, path, size):
        self.path = path
        self.data = [None] * size
        self.size = size
    def __getitem__(self, index):
        if self.data[index]: return self.data[index]

        npimg = np.moveaxis(np.load(f"{self.path}/image/{index}.npy"), -1, 0)
        image = torch.from_numpy(npimg)

        npsem = np.moveaxis(np.load(f"{self.path}/label/{index}.npy"), -1, 0)
        sem = torch.from_numpy(npsem)
        print("lksjdf", sem.shape)

        self.data[index] = {"img": image.float().unsqueeze(0), "sem": sem.float().unsqueeze(0)}
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
            nn.Conv2d(32, 20, kernel_size=1),  #instead of rgb we have 20 channels
        )
    def forward(self, x):
        x = self.preModel.forward(x)
        return self.model.forward(x)

def showImageAndSem(img):
    sem = img["sem"]
    img = img["img"]
    plt.subplot(1,3,1)
    plt.imshow(img.permute(1, 2, 0))
    plt.subplot(1,3,2)
    plt.imshow(sem.T)
    plt.show()

def main():
    trainData = DataLoader("../data/fcn/train", 2975)
    testData = DataLoader("../data/fcn/val", 500)
    alexnet = models.alexnet(weights = models.AlexNet_Weights.DEFAULT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)
    model = FCN(alexnet)
    if os.path.exists("model.pt"): model.load_state_dict(torch.load("model.pt", weights_only=True))
    model.to(device)
    lossFunc = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    EPOCH = 10
    runningLoss = 0
    try:
        for epoch in range(EPOCH):
            for i in range(trainData.size):
                input, truth = trainData[i]["img"], trainData[i]["sem"].T
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


if __name__ == "__main__": main()