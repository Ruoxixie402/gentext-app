"""End-to-end Module 6: real MNIST -> train_gan -> displayed sample grid.

From the repository root: uv run python -m examples.module6_mnist --epochs 1
"""
import argparse
import json
from pathlib import Path

import torch
from torch import nn

from helper_lib import get_model, get_mnist_dataloader, train_gan, generate_samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/module6-example'))
    parser.add_argument('--mnist-mirror')
    parser.add_argument('--no-download', action='store_true')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1:
        parser.error('epochs, batch-size and threads must be positive')
    torch.manual_seed(42)
    torch.set_num_threads(args.threads)
    loader = get_mnist_dataloader(args.data_dir, args.batch_size,
        download=not args.no_download, mirror=args.mnist_mirror)
    print(f'Real torchvision MNIST: {len(loader.dataset)} training digits; '
          f'{len(loader)} batches per epoch; image shape (1,28,28)', flush=True)
    generator, discriminator = get_model('GAN')
    optimizers = (torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999)),
                  torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)))
    trained_model = train_gan((generator, discriminator), loader, nn.BCELoss(),
                               optimizers, device='cpu', epochs=args.epochs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'generator': trained_model[0].state_dict(), 'epoch': args.epochs,
                'seed': 42, 'dataset': 'MNIST'}, args.output_dir / 'generator.pt')
    grid = generate_samples(trained_model, 'cpu', num_samples=10)
    grid.save(args.output_dir / 'samples.png')
    report = {'dataset': 'torchvision.datasets.MNIST', 'split': 'train',
              'dataset_size': len(loader.dataset), 'examples_per_epoch': len(loader.dataset),
              'batches_per_epoch': len(loader), 'epochs_completed': args.epochs,
              'batch_size': args.batch_size, 'seed': 42, 'device': 'cpu',
              'image_shape': [1, 28, 28], 'normalization': 'ToTensor, Normalize(mean=0.5,std=0.5)',
              'loader_source': 'helper_lib/dataloader.py:get_mnist_dataloader',
              'training_source': 'helper_lib/trainer.py:train_gan',
              'torch_version': str(torch.__version__)}
    (args.output_dir / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
