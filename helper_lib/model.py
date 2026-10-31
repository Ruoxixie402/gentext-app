"""Model factory for the Module 6 activity.

GAN returns (generator, discriminator); other names return a single nn.Module.
The non-GAN models are standalone MNIST-compatible reference implementations.
"""
import torch
from torch import nn
from app.gan.models import Generator, Discriminator


class FCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(32 * 7 * 7, 10))

    def forward(self, x):
        return self.net(x)


class EnhancedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x)


class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(784, 400), nn.ReLU())
        self.mu = nn.Linear(400, latent_dim)
        self.logvar = nn.Linear(400, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 400), nn.ReLU(), nn.Linear(400, 784), nn.Sigmoid())

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        reconstruction = self.decoder(z).reshape(-1, 1, 28, 28)
        return reconstruction, mu, logvar


def get_model(model_name):
    """Return FCNN, CNN, EnhancedCNN, VAE, or a (Generator, Discriminator) GAN pair."""
    name = model_name.strip().lower().replace('_', '').replace('-', '')
    models = {'fcnn': FCNN, 'cnn': CNN, 'enhancedcnn': EnhancedCNN, 'vae': VAE}
    if name == 'gan':
        return Generator(), Discriminator()
    if name not in models:
        raise ValueError(f"Unknown model {model_name!r}; choose FCNN, CNN, EnhancedCNN, VAE, or GAN")
    return models[name]()
