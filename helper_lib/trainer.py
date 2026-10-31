"""Reusable GAN training loop for the Module 6 activity."""
import torch
from torch import nn


def train_gan(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    """Train a GAN and return epoch-average losses.

    model: (generator, discriminator) tuple.
    optimizer: (generator_optimizer, discriminator_optimizer) tuple.
    Data loader yields MNIST images or (images, labels). Images should be in [-1, 1].
    """
    if epochs < 1:
        raise ValueError('epochs must be positive')
    if not isinstance(model, (tuple, list)) or len(model) != 2:
        raise ValueError('model must be a (generator, discriminator) pair')
    if not isinstance(optimizer, (tuple, list)) or len(optimizer) != 2:
        raise ValueError('optimizer must be a (generator_optimizer, discriminator_optimizer) pair')
    generator, discriminator = model
    opt_g, opt_d = optimizer
    device = torch.device(device)
    generator.to(device)
    discriminator.to(device)
    history = []
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        sum_g = sum_d = total = 0
        for batch_data in data_loader:
            real = batch_data[0] if isinstance(batch_data, (tuple, list)) else batch_data
            real = real.to(device)
            batch_size = real.size(0)
            ones = torch.ones(batch_size, 1, device=device)
            zeros = torch.zeros(batch_size, 1, device=device)
            fake = generator(torch.randn(batch_size, generator.latent_dim, device=device))
            opt_d.zero_grad(set_to_none=True)
            loss_d = criterion(discriminator(real), ones) + criterion(discriminator(fake.detach()), zeros)
            loss_d.backward()
            opt_d.step()
            discriminator.requires_grad_(False)
            discriminator.eval()
            opt_g.zero_grad(set_to_none=True)
            loss_g = criterion(discriminator(fake), ones)
            loss_g.backward()
            opt_g.step()
            discriminator.requires_grad_(True)
            discriminator.train()
            sum_g += loss_g.item() * batch_size
            sum_d += loss_d.item() * batch_size
            total += batch_size
        if total == 0:
            raise ValueError('data_loader is empty')
        history.append({'epoch': epoch + 1, 'generator_loss': sum_g / total, 'discriminator_loss': sum_d / total})
    return history
