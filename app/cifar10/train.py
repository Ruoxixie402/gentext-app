"""Train on all 50,000 CIFAR-10 training images; the test split is held out."""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from helper_lib.dataloader import get_cifar10_dataloader

from .models import EnergyModel, NoiseUNet
from .sampling import DiffusionSchedule, langevin_sample


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = get_cifar10_dataloader(args.data_dir, args.batch_size)
    dataset = loader.dataset
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.model == 'ebm':
        model = EnergyModel(args.channels).to(device)
        replay = torch.rand(1024, 3, 32, 32, device=device) * 2 - 1
    else:
        model = NoiseUNet(args.channels).to(device)
        schedule = DiffusionSchedule(args.diffusion_steps, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    report = {
        'model': args.model, 'dataset': 'CIFAR-10', 'split': 'train',
        'examples_per_epoch': len(dataset), 'seed': args.seed, 'device': str(device),
        'torch_version': str(torch.__version__), 'channels': args.channels,
        'batch_size': args.batch_size, 'learning_rate': args.learning_rate,
        'diffusion_steps': args.diffusion_steps if args.model == 'diffusion' else None,
        'langevin_steps': args.langevin_steps if args.model == 'ebm' else None,
        'langevin_step_size': args.step_size if args.model == 'ebm' else None,
        'energy_regularization': 0.01 if args.model == 'ebm' else None,
        'replay_buffer_size': 1024 if args.model == 'ebm' else None,
        'replay_restart_probability': 0.05 if args.model == 'ebm' else None,
        'history': [],
        'limitations': 'Small educational model; generated quality has not been quantified with FID or IS.',
    }
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = count = 0
        positive_total = negative_total = 0
        for real, _ in loader:
            real = real.to(device)
            batch = len(real)
            optimizer.zero_grad(set_to_none=True)
            if args.model == 'ebm':
                # Distinct replay slots avoid duplicate assignment ambiguity.
                indices = torch.randperm(len(replay), device=device)[:batch]
                initial = replay[indices].clone()
                restart = torch.rand(batch, device=device) < 0.05
                initial[restart] = torch.rand_like(initial[restart]) * 2 - 1
                negative = langevin_sample(model, initial, args.langevin_steps, args.step_size)
                replay[indices] = negative
                # Model parameters are trainable again; samples are detached.
                positive_energy, negative_energy = model(real), model(negative)
                loss = (positive_energy.mean() - negative_energy.mean()
                        + 0.01 * (positive_energy.square().mean() + negative_energy.square().mean()))
                positive_total += positive_energy.detach().sum().item()
                negative_total += negative_energy.detach().sum().item()
            else:
                t = torch.randint(schedule.steps, (batch,), device=device)
                noise = torch.randn_like(real)
                noisy = schedule.q_sample(real, t, noise)
                # Noisy inputs/targets have no parameter gradients; UNet weights do.
                loss = F.mse_loss(model(noisy, t), noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * batch
            count += batch
        metrics = {'epoch': epoch, 'loss': total / count, 'elapsed_seconds': time.monotonic() - started}
        if args.model == 'ebm':
            metrics.update(real_energy=positive_total / count, negative_energy=negative_total / count)
        report['history'].append(metrics)
        print(json.dumps(metrics), flush=True)
        model.eval()
        checkpoint = {'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                      'channels': args.channels, 'epoch': epoch, 'seed': args.seed,
                      'model': args.model, 'diffusion_steps': args.diffusion_steps}
        torch.save(checkpoint, args.output_dir / f'{args.model}.pt')
        (args.output_dir / f'{args.model}-training.json').write_text(json.dumps(report, indent=2) + '\n')
        rng = torch.Generator(device=device).manual_seed(args.seed + 1)
        if args.model == 'ebm':
            initial = torch.rand((16, 3, 32, 32), device=device, generator=rng) * 2 - 1
            samples = langevin_sample(model, initial, 100, args.step_size, generator=rng)
        else:
            samples = schedule.sample(model, 16, generator=rng)
        save_image(samples.cpu(), args.output_dir / f'{args.model}-samples.png', nrow=4,
                   normalize=True, value_range=(-1, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['ebm', 'diffusion'], required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--channels', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=0.0002)
    parser.add_argument('--diffusion-steps', type=int, default=100)
    parser.add_argument('--langevin-steps', type=int, default=10)
    parser.add_argument('--step-size', type=float, default=0.01)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/assignment4'))
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads, args.langevin_steps) < 1:
        parser.error('epochs, batch-size, threads and langevin-steps must be positive')
    if args.batch_size > 1024 or args.channels < 4 or args.channels % 4:
        parser.error('batch-size must be <=1024; channels must be a positive multiple of four')
    if args.diffusion_steps < 2 or args.step_size <= 0 or args.learning_rate <= 0:
        parser.error('diffusion-steps must be >=2; step-size and learning-rate must be positive')
    train(args)


if __name__ == '__main__':
    main()
