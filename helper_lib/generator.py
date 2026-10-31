"""Generate MNIST-style GAN samples and arrange them in a displayable grid."""
import torch
from PIL import Image
from torchvision.utils import make_grid


def generate_samples(model, device, num_samples=10):
    """Return a PIL image grid, displayable with display(image) or image.show().

    Accepts either a generator or a (generator, discriminator) pair.
    """
    if num_samples < 1:
        raise ValueError('num_samples must be positive')
    generator = model[0] if isinstance(model, (tuple, list)) else model
    device = torch.device(device)
    generator.to(device)
    was_training = generator.training
    generator.eval()
    try:
        with torch.inference_mode():
            images = generator(torch.randn(num_samples, generator.latent_dim, device=device)).cpu()
    finally:
        generator.train(was_training)
    grid = make_grid(images, nrow=min(5, num_samples), normalize=True, value_range=(-1, 1), padding=2)
    pixels = (grid.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(pixels)
