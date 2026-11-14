"""Held-out diagnostics; these are not perceptual image-quality scores."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from helper_lib.dataloader import get_cifar10_dataset

from .models import EnergyModel, NoiseUNet
from .sampling import DiffusionSchedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/assignment4'))
    parser.add_argument('--examples', type=int, default=512)
    args = parser.parse_args()
    if not 1 <= args.examples <= 10000:
        parser.error('examples must be between 1 and 10000')
    torch.set_num_threads(2)
    torch.manual_seed(123)
    dataset = get_cifar10_dataset(args.data_dir, train=False, download=False)
    indices = torch.randperm(len(dataset))[:args.examples].tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=64)
    models = {}
    for kind, cls in [('ebm', EnergyModel), ('diffusion', NoiseUNet)]:
        checkpoint = torch.load(args.output_dir / f'{kind}.pt', weights_only=True, map_location='cpu')
        model = cls(checkpoint['channels'])
        model.load_state_dict(checkpoint['state_dict'])
        models[kind] = model.eval()
        if kind == 'diffusion':
            schedule = DiffusionSchedule(checkpoint['diffusion_steps'])
    torch.manual_seed(42)
    baseline = NoiseUNet(checkpoint['channels']).eval()
    rng = torch.Generator().manual_seed(1234)
    mse = baseline_mse = real_e = uniform_e = 0.
    with torch.inference_mode():
        for images, _ in loader:
            batch = len(images)
            t = torch.randint(schedule.steps, (batch,), generator=rng)
            noise = torch.randn(images.shape, generator=rng)
            noisy = schedule.q_sample(images, t, noise)
            mse += F.mse_loss(models['diffusion'](noisy, t), noise).item() * batch
            baseline_mse += F.mse_loss(baseline(noisy, t), noise).item() * batch
            real_e += models['ebm'](images).sum().item()
            uniform = torch.rand(images.shape, generator=rng) * 2 - 1
            uniform_e += models['ebm'](uniform).sum().item()
    report = {'dataset': 'CIFAR-10', 'split': 'test', 'examples': args.examples,
              'selection_seed': 123, 'evaluation_seed': 1234,
              'diffusion_noise_mse': mse / args.examples,
              'untrained_unet_noise_mse_same_inputs': baseline_mse / args.examples,
              'ebm_mean_real_energy': real_e / args.examples,
              'ebm_mean_uniform_noise_energy': uniform_e / args.examples,
              'interpretation': 'Noise-prediction and energy-separation diagnostics only. '
                  'Uniform images are not model samples. These are not FID, likelihood, '
                  'class accuracy, or evidence of recognizable generated objects.'}
    (args.output_dir / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
