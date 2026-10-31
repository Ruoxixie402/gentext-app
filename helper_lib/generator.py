"""Generate and display a grid of MNIST-style GAN samples."""
import math
import torch


def generate_samples(model, device, num_samples=10):
    """Return a Matplotlib figure of generated digits (without blocking display).

    Accepts a generator or a (generator, discriminator) pair.
    """
    if num_samples < 1:
        raise ValueError('num_samples must be positive')
    import matplotlib.pyplot as plt
    generator = model[0] if isinstance(model, (tuple, list)) else model
    device = torch.device(device)
    generator.to(device)
    was_training = generator.training
    generator.eval()
    try:
        with torch.inference_mode():
            images = generator(torch.randn(num_samples, generator.latent_dim, device=device)).detach().cpu()
    finally:
        generator.train(was_training)
    columns = min(5, num_samples)
    rows = math.ceil(num_samples / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 2, rows * 2), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis('off')
        if i < num_samples:
            ax.imshow(images[i, 0].clamp(-1, 1).add(1).div(2).numpy(), cmap='gray', vmin=0, vmax=1)
    fig.tight_layout()
    return fig
