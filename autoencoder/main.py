import torch
import torch.utils
import torch.utils.data
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt

EPOCH = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

train = torch.utils.data.DataLoader(torchvision.datasets.MNIST("../data/", True, transform=transform, download=True), batch_size=5)
test  = torch.utils.data.DataLoader(torchvision.datasets.MNIST("../data/", False, transform=transform, download=True), batch_size=1)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=4), #10, 25, 25
            nn.ReLU(True),
            nn.Conv2d(in_channels=10, out_channels=5, kernel_size=4), #10, 22, 22
            nn.ReLU(True),
        )
        self.nn = nn.Sequential(
            nn.Linear(5 * 22 * 22, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 500),
            nn.ReLU(True),
            nn.Linear(500, 16),
            nn.ReLU(True),
        )
    def forward(self, x):
        x = self.cn(x)
        x = torch.flatten(x, start_dim=1)
        return self.nn(x)
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cn = nn.Sequential(
            nn.ConvTranspose2d(1, 10, 4),   #10, 7, 7
            nn.ReLU(True),
            nn.ConvTranspose2d(10, 10, 4),  #10, 10, 10
            nn.ReLU(True),
        )
        self.nn = nn.Sequential(
            nn.Linear(10 * 10 * 10, 800),
            nn.ReLU(True),
            nn.Linear(800, 28*28),
        )
    def forward(self, x):
        x = self.cn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.nn(x)
        return x.view(-1, 1, 28, 28)
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.dec = Decoder()
    def forward(self, x):
        x = self.enc.forward(x)
        x = x.view(-1, 1, 4, 4)
        return self.dec.forward(x)
    
def showImage(image, img2):
    fig = plt.figure()
    plt.subplot(2,3,1)
    plt.tight_layout()
    plt.subplot(1,3,1)
    plt.imshow(image[0], cmap='gray', interpolation='none')
    plt.subplot(1,3,2)
    plt.imshow(img2[0], cmap='gray', interpolation='none')
    plt.xticks([])
    plt.yticks([])
    plt.show()

autoEnc = AutoEncoder()
autoEnc.to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(autoEnc.parameters())

def testAutoEnc(model):
    with torch.no_grad():
        for data in test:
            img = data[0]
            img.requires_grad = False
            imgDevice = img.to(DEVICE)
            rec = model.forward(imgDevice)
            showImage(img.squeeze(0), rec.cpu().squeeze(0).detach().numpy())
def trainAutoEnc():
    for epoch in range(EPOCH):
        print("epoch", epoch)
        try:
            for data in train:
                img = data[0]
                img.requires_grad = False
                imgDevice = img.to(DEVICE)
                rec = autoEnc.forward(imgDevice)
                loss = criterion(rec.cpu(), img)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f"loss: {loss.cpu().detach().numpy()}", end='\r')
        except:
            print("Exiting training")
            torch.save(autoEnc.state_dict(), "model.pt")
            testAutoEnc(autoEnc)
            return
        
trainAutoEnc()