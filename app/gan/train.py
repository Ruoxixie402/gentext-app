"""Run from the repository root: python -m app.gan.train --epochs 20."""
import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from .models import Discriminator, Generator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts'))
    parser.add_argument('--mnist-mirror', help='Optional HTTPS base URL for the standard MNIST gzip files')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1:
        parser.error('epochs, batch-size, and threads must be positive')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    if args.mnist_mirror:
        datasets.MNIST.mirrors = [args.mnist_mirror.rstrip('/') + '/']
    dataset = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    generator, discriminator = Generator().to(device), Discriminator().to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, 100, device=device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        total_g = total_d = count = 0
        for real, _ in loader:
            real = real.to(device)
            batch = real.shape[0]
            ones, zeros = torch.ones(batch, 1, device=device), torch.zeros(batch, 1, device=device)
            opt_d.zero_grad(set_to_none=True)
            fake = generator(torch.randn(batch, 100, device=device))
            loss_d = criterion(discriminator(real), ones) + criterion(discriminator(fake.detach()), zeros)
            loss_d.backward()
            opt_d.step()
            # Freeze discriminator weights and running statistics for the G update.
            discriminator.requires_grad_(False)
            discriminator.eval()
            opt_g.zero_grad(set_to_none=True)
            loss_g = criterion(discriminator(fake), ones)
            loss_g.backward()
            opt_g.step()
            discriminator.requires_grad_(True)
            discriminator.train()
            total_g += loss_g.item() * batch
            total_d += loss_d.item() * batch
            count += batch
        metrics = {'epoch': epoch, 'generator_loss': total_g / count, 'discriminator_loss': total_d / count}
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        generator.eval()
        with torch.inference_mode():
            save_image(generator(fixed_noise).cpu(), args.output_dir / 'samples.png', nrow=8, normalize=True, value_range=(-1, 1))
        # Only tensors and basic metadata: safe to load using weights_only=True.
        torch.save({'generator': generator.cpu().state_dict(), 'epoch': epoch, 'seed': args.seed}, args.output_dir / 'generator.pt')
        generator.to(device)
        (args.output_dir / 'training.json').write_text(json.dumps({
            'dataset': 'MNIST training split', 'examples_per_epoch': len(dataset),
            'device': str(device), 'torch_version': str(torch.__version__),
            'seed': args.seed, 'batch_size': args.batch_size, 'learning_rate': 0.0002,
            'adam_betas': [0.5, 0.999], 'history': history,
        }, indent=2) + '\n')


if __name__ == '__main__':
    main()
