import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import random_split
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

torch.manual_seed(0)

image_path = './outputs/UE4 and Airsim/20240417-155906/results/Depth'

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(256 * 9 * 16, 1024),
        )

        self.fc_mu = nn.Linear(1024, 32)  # Producing the mean of the latent code
        self.fc_logvar = nn.Linear(1024, 32)  # Producing the log variance of the latent code
        self.fc_decoder = nn.Linear(32, 256 * 9 * 16)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 9, 16)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), 
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        x_recon = self.fc_decoder(z)
        x_recon = self.decoder(x_recon)
        return x_recon, mu, logvar

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.images = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f)) and f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.images[idx])
        image = Image.open(img_name).convert('L')
        if self.transform:
            image = self.transform(image)
        return image

transform = transforms.Compose([
    transforms.Resize((144, 256)),
    transforms.ToTensor(),
])

def loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div

total_data = MyDataset(image_folder=image_path, transform=transform)
m = len(total_data)
train_data, val_data = random_split(total_data, [int(m*0.8), m - int(m*0.8)])
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 100

def train(model, train_loader, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_images, mu, logvar = model(data)
        loss = loss_function(recon_images, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss / len(train_loader.dataset)

def validate(model, val_loader):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            data = data.to(device)
            recon_images, mu, logvar = model(data)
            loss = loss_function(recon_images, data, mu, logvar)
            val_loss += loss.item()
    return val_loss / len(val_loader.dataset)

if __name__ == '__main__':
    print('Training VAE...')
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer)
        val_loss = validate(model, val_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Train Loss: {train_loss:.4f}, Average Val Loss: {val_loss:.4f}')

    # Save the model
    torch.save(model.state_dict(), 'vae.pt')