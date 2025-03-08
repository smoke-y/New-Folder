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

class VAE(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, hidden),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(True),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid()
        )
        self.mu = nn.Linear(hidden, hidden)
        self.logVar = nn.Linear(hidden, hidden)
    def reparameterize(self, mu: torch.Tensor, logVar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logVar)
        eps = torch.rand_like(mu)
        return mu + std*eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        mu = self.mu(x)
        logVar = self.logVar(x)
        z = self.reparameterize(mu, logVar)
        decoded = self.decoder(z)
        return x, decoded, mu, logVar

def lossFunc(truth,x,mu,logVar: torch.Tensor) -> torch.Tensor:
    BCE = nn.functional.binary_cross_entropy(truth, x.view(-1, 28*28), reduction="sum",)
    KLD = -0.5 * torch.sum(1 + logVar - mu.pow(2) - logVar.exp())
    return BCE + KLD

model = VAE(8)
for epoch in range(EPOCH):
    for i in train:
        x,y = i
        x.to(DEVICE)
        x = x.view(-1, 28*28)
        pred, decoded, mu, logVar = model.forward(x)
        loss = lossFunc(x, decoded, mu, logVar)
        loss.backward()