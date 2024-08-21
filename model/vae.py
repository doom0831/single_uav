import torch
import torch.nn as nn
import torch.nn.functional as F
import os

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

    def forward(self, x, return_latent_code=False):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        x_recon = self.fc_decoder(z)
        x_recon = self.decoder(x_recon)
        if return_latent_code:
            return z
        else:
            return x_recon, mu, logvar
        
    def save(self, path, filename):
        torch.save(self.state_dict(), os.path.join(path, filename))

    def load(self, path):
        self.load_state_dict(torch.load(path))

