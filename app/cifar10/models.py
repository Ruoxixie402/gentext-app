import math

import torch
from torch import nn
import torch.nn.functional as F


def sinusoidal_embedding(timesteps, dim=64, max_period=10000):
    """Transformer convention: interleave sin/cos; frequencies max_period^(-2i/d)."""
    if dim < 2 or dim % 2:
        raise ValueError('Embedding dimension must be positive and even (at least 2)')
    frequencies = torch.exp(-math.log(max_period) * torch.arange(
        0, dim, 2, device=timesteps.device, dtype=torch.float32) / dim)
    angles = timesteps.float().reshape(-1, 1) * frequencies
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)


class EnergyModel(nn.Module):
    """Scalar E(x) per RGB image; smaller energy means higher modeled density."""
    def __init__(self, channels=32):
        super().__init__()
        self.channels = channels
        self.net = nn.Sequential(
            nn.Conv2d(3, channels, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(channels, channels * 2, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(channels * 2, channels * 2, 3, 2, 1), nn.SiLU(),
            nn.Flatten(), nn.Linear(channels * 2 * 4 * 4, 1),
        )

    def forward(self, x):
        return self.net(x).flatten()


class TimeBlock(nn.Module):
    def __init__(self, inputs, outputs, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(inputs, outputs, 3, padding=1)
        self.conv2 = nn.Conv2d(outputs, outputs, 3, padding=1)
        self.norm1 = nn.GroupNorm(4, outputs)
        self.norm2 = nn.GroupNorm(4, outputs)
        self.time = nn.Linear(time_dim, outputs)
        self.skip = nn.Conv2d(inputs, outputs, 1) if inputs != outputs else nn.Identity()

    def forward(self, x, time):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time(time)[:, :, None, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class NoiseUNet(nn.Module):
    """32 -> 16 -> 8 -> 16 -> 32, skip connections and timestep conditioning."""
    def __init__(self, channels=16, time_dim=64):
        super().__init__()
        self.channels, self.time_dim = channels, time_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.enc1 = TimeBlock(3, channels, time_dim)
        self.down1 = nn.Conv2d(channels, channels * 2, 4, 2, 1)
        self.enc2 = TimeBlock(channels * 2, channels * 2, time_dim)
        self.down2 = nn.Conv2d(channels * 2, channels * 4, 4, 2, 1)
        self.middle = TimeBlock(channels * 4, channels * 4, time_dim)
        self.up2 = nn.ConvTranspose2d(channels * 4, channels * 2, 4, 2, 1)
        self.dec2 = TimeBlock(channels * 4, channels * 2, time_dim)
        self.up1 = nn.ConvTranspose2d(channels * 2, channels, 4, 2, 1)
        self.dec1 = TimeBlock(channels * 2, channels, time_dim)
        self.output = nn.Conv2d(channels, 3, 1)

    def forward(self, x, t):
        time = self.time_mlp(sinusoidal_embedding(t, self.time_dim))
        e1 = self.enc1(x, time)
        e2 = self.enc2(self.down1(e1), time)
        h = self.middle(self.down2(e2), time)
        h = self.dec2(torch.cat((self.up2(h), e2), dim=1), time)
        h = self.dec1(torch.cat((self.up1(h), e1), dim=1), time)
        return self.output(h)
