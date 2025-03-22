import torch.nn as nn
import torchvision
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F

EPOCH = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.cnbn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.cnbn2 = nn.BatchNorm2d(20)
        self.conv2_drop = nn.Dropout2d()
        self.ln1 = nn.LayerNorm(320)
        self.fc1 = nn.Linear(320, 50)
        self.ln2 = nn.LayerNorm(50)
        self.fc2 = nn.Linear(50, 1)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.cnbn1(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.cnbn2(self.conv2(x))), 2))
        x = x.view(-1, 320)
        x = self.ln1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.ln2(x)
        x = self.fc2(x)
        return torch.sigmoid(x)
class Generator(nn.Module):
    def __init__(self, latentDim):
        super().__init__()
        self.lin1 = nn.Linear(latentDim, 7*7*64)  # [n, 256, 7, 7]
        self.ct1 = nn.ConvTranspose2d(64, 32, 4, stride=2) # [n, 64, 16, 16]
        self.ct2 = nn.ConvTranspose2d(32, 16, 4, stride=2) # [n, 16, 34, 34]
        self.conv = nn.Conv2d(16, 1, kernel_size=7)  # [n, 1, 28, 28]
    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = x.view(-1, 64, 7, 7)
        x = self.ct1(x)
        x = F.relu(x)
        x = self.ct2(x)
        x = F.relu(x)
        return self.conv(x)
    
pyone = Variable(torch.ones((1,1), device=DEVICE, requires_grad=False))
pyzero = Variable(torch.zeros((1,1), device=DEVICE, requires_grad=False))
def calcLossD(gan, realImg, genImg):
    real = nn.functional.binary_cross_entropy(gan.D.forward(realImg), pyone)
    fake = nn.functional.binary_cross_entropy(gan.D.forward(genImg), pyzero)
    return real+fake
def calcLossG(gan, realImg, genImg): return nn.functional.binary_cross_entropy(gan.D.forward(genImg), pyone)

class GAN(nn.Module):
    def __init__(self, latentDim=100) -> None:
        super().__init__()
        self.G = Generator(latentDim)
        self.D = Discriminator()
    def forward(self, z): return self.G(z)
def showImage(image):
    fig = plt.figure()
    plt.subplot(2,3,1)
    plt.tight_layout()
    plt.imshow(image[0], cmap='gray', interpolation='none')
    plt.xticks([])
    plt.yticks([])
    plt.show()

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])

train = torchvision.datasets.MNIST("../data/", True, transform=transform, download=True)
test  = torchvision.datasets.MNIST("../data/", False, transform=transform, download=True)
gan = GAN(100)

def testGAN():
    with torch.no_grad():
        import os
        assert os.path.exists("model.pt")
        gan = GAN(100)
        gan.load_state_dict(torch.load("model.pt", weights_only=True))
        for i in range(15):
            sample = torch.rand((1, 100))
            gen = gan.forward(sample)
            showImage(gen.detach().squeeze(0).numpy())

def trainStep(gan, optim, calcLoss, img):
    sample = Variable(torch.randn((1,100), device=DEVICE, requires_grad=False))
    gen = gan.forward(sample)
    optim.zero_grad()
    loss = calcLoss(gan, img, gen)
    loss.backward()
    optim.step()
    return loss.cpu().detach().numpy()
def trainGAN():
    gan.to(DEVICE)
    optimG = torch.optim.Adam(gan.G.parameters(), lr=0.0002)
    optimD = torch.optim.Adam(gan.D.parameters(), lr=0.0002)
    try:
        for epoch in range(EPOCH):
            print(f"EPOCH({epoch})")
            runningDLoss = 0
            runningGLoss = 0
            for data in train:
                imgCPU, _ = data
                if imgCPU.dim() == 3: imgCPU = imgCPU.unsqueeze(0)
                imgWithNoise = imgCPU + torch.normal(mean=0.0, std=0.1, size=imgCPU.size())
                img = imgCPU.to(DEVICE)
                img = Variable(img)

                lossD = trainStep(gan, optimD, calcLossD, imgWithNoise.to(DEVICE))
                lossG = trainStep(gan, optimG, calcLossG, img)
                
                runningDLoss += lossD
                runningGLoss += lossG
                print(f"G({lossG}) D({lossD})", end='\r')
            print(f"running_loss: D({runningDLoss/len(train)})  G({runningGLoss/len(train)})")
    except: print("Exiting training...")
    torch.save(gan.state_dict(), "model.pt")
    testGAN()

if __name__ == "__main__": trainGAN()
