"""Module 6 helper-library smoke tests."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
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


def test_gan_training_returns_trained_model():
    generator, discriminator = get_model('GAN')
    loader = DataLoader(TensorDataset(torch.rand(2, 1, 28, 28) * 2 - 1, torch.zeros(2)), batch_size=2)
    optimizers = (torch.optim.Adam(generator.parameters(), lr=1e-4), torch.optim.Adam(discriminator.parameters(), lr=1e-4))
    model = (generator, discriminator)
    before_g = next(generator.parameters()).detach().clone()
    before_d = next(discriminator.parameters()).detach().clone()
    trained_model = train_gan(model, loader, nn.BCELoss(), optimizers, epochs=1)
    assert trained_model is model
    assert not torch.equal(before_g, next(generator.parameters()))
    assert not torch.equal(before_d, next(discriminator.parameters()))


@pytest.mark.parametrize('count,use_pair', [(3, False), (10, True)])
def test_generated_samples_are_displayed(monkeypatch, count, use_pair):
    model = get_model('GAN')
    generator = model[0]
    generated_counts = []
    hook = generator.register_forward_hook(lambda _module, _inputs, output: generated_counts.append(len(output)))
    displayed = []
    def capture_display():
        figure = plt.gcf()
        assert len(figure.axes) == 1
        displayed.append(figure.axes[0].images[0].get_array().copy())
    monkeypatch.setattr(plt, 'show', capture_display)
    existing_figures = plt.get_fignums()
    try:
        if count == 10:
            image = generate_samples(model if use_pair else generator, 'cpu')
        else:
            image = generate_samples(model if use_pair else generator, 'cpu', num_samples=count)
    finally:
        hook.remove()
    assert generated_counts == [count]
    assert len(displayed) == 1
    assert isinstance(image, Image.Image)
    assert displayed[0].shape[:2] == (image.height, image.width)
    assert image.width > 28 and image.height >= 28
    assert generator.training  # Sampling restores the original mode.
    assert plt.get_fignums() == existing_figures  # No leaked figures.
