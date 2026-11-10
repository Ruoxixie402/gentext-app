from contextlib import contextmanager

import torch


@contextmanager
def frozen_parameters(model):
    """Freeze weights without disabling the input-gradient computation."""
    flags = [p.requires_grad for p in model.parameters()]
    training = model.training
    model.requires_grad_(False)
    model.eval()
    try:
        yield
    finally:
        for p, flag in zip(model.parameters(), flags):
            p.requires_grad_(flag)
        model.train(training)


def langevin_sample(model, initial, steps=30, step_size=0.02, generator=None, add_noise=True):
    """Projected ULA: x <- x - eta*grad_x E(x) + sqrt(2*eta)*N(0,I).

    Only x receives gradients. Updates run without a graph, and the result is
    detached before contrastive model training. Clamping approximates bounded
    image support; finite-step samples are not exact equilibrium samples.
    """
    if steps < 1 or step_size <= 0:
        raise ValueError('steps and step_size must be positive')
    x = initial.detach().clone()
    with frozen_parameters(model), torch.enable_grad():
        for _ in range(steps):
            x = x.detach().requires_grad_(True)
            gradient, = torch.autograd.grad(model(x).sum(), x, create_graph=False)
            with torch.no_grad():
                x = x - step_size * gradient
                if add_noise:
                    noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
                    x = x + (2 * step_size) ** 0.5 * noise
                x.clamp_(-1, 1)
    return x.detach()


class DiffusionSchedule:
    """Cosine DDPM schedule with epsilon prediction and ancestral sampling."""
    def __init__(self, steps=100, device='cpu'):
        if steps < 2:
            raise ValueError('At least two diffusion steps are required')
        self.steps = steps
        grid = torch.linspace(0, steps, steps + 1, device=device)
        curve = torch.cos(((grid / steps + 0.008) / 1.008) * torch.pi / 2).square()
        curve = curve / curve[0]
        self.betas = (1 - curve[1:] / curve[:-1]).clamp(0.0001, 0.999)
        self.alphas = 1 - self.betas
        self.alpha_bars = self.alphas.cumprod(0)
        previous = torch.cat((torch.ones(1, device=device), self.alpha_bars[:-1]))
        self.posterior_variance = self.betas * (1 - previous) / (1 - self.alpha_bars)
        self.mean_x0 = self.betas * previous.sqrt() / (1 - self.alpha_bars)
        self.mean_xt = (1 - previous) * self.alphas.sqrt() / (1 - self.alpha_bars)

    def q_sample(self, clean, t, noise):
        abar = self.alpha_bars[t][:, None, None, None]
        return abar.sqrt() * clean + (1 - abar).sqrt() * noise

    @torch.no_grad()
    def sample(self, model, count=1, generator=None):
        device = self.betas.device
        x = torch.randn((count, 3, 32, 32), device=device, generator=generator)
        for step in reversed(range(self.steps)):
            t = torch.full((count,), step, device=device, dtype=torch.long)
            predicted_noise = model(x, t)
            clean = ((x - (1 - self.alpha_bars[step]).sqrt() * predicted_noise)
                     / self.alpha_bars[step].sqrt()).clamp(-1, 1)
            x = self.mean_x0[step] * clean + self.mean_xt[step] * x
            if step > 0:
                x = x + self.posterior_variance[step].sqrt() * torch.randn(
                    x.shape, device=device, generator=generator)
        return x.clamp(-1, 1)
