"""Module 6 helper-library smoke tests."""
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from helper_lib import get_model, train_gan, generate_samples


def test_model_factory():
    for name in ('FCNN', 'CNN', 'EnhancedCNN'):
        assert get_model(name)(torch.rand(2, 1, 28, 28)).shape == (2, 10)
    reconstruction, mu, logvar = get_model('VAE')(torch.rand(2, 1, 28, 28))
    assert reconstruction.shape == (2, 1, 28, 28)
    assert mu.shape == logvar.shape == (2, 20)
    generator, discriminator = get_model('GAN')
    assert generator(torch.randn(2, 100)).shape == (2, 1, 28, 28)
    assert discriminator(torch.rand(2, 1, 28, 28)).shape == (2, 1)


def test_gan_training_and_samples():
    generator, discriminator = get_model('GAN')
    loader = DataLoader(TensorDataset(torch.rand(2, 1, 28, 28) * 2 - 1, torch.zeros(2)), batch_size=2)
    optimizers = (torch.optim.Adam(generator.parameters(), lr=1e-4), torch.optim.Adam(discriminator.parameters(), lr=1e-4))
    history = train_gan((generator, discriminator), loader, nn.BCELoss(), optimizers, epochs=1)
    assert len(history) == 1
    assert history[0]['generator_loss'] >= 0
    image = generate_samples(generator, 'cpu', num_samples=3)
    assert isinstance(image, Image.Image)
    assert image.width > 28 and image.height >= 28
