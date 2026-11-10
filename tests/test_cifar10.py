import base64
import io

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from app.cifar10 import api
from app.cifar10.models import EnergyModel, NoiseUNet, sinusoidal_embedding
from app.cifar10.sampling import DiffusionSchedule, langevin_sample


def test_input_sampling_lowers_energy_without_weight_gradients():
    torch.set_num_threads(1)
    class Quadratic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
        def forward(self, x):
            return self.weight * x.square().flatten(1).sum(1)
    model = Quadratic().train()
    initial = torch.full((2, 3, 32, 32), 0.5)
    sample = langevin_sample(model, initial, 10, 0.1, add_noise=False)
    assert model(sample).mean() < model(initial).mean()
    assert model.weight.grad is None
    assert model.weight.requires_grad and model.training
    assert not sample.requires_grad and initial.requires_grad is False
    assert torch.equal(initial, torch.full_like(initial, 0.5))


def test_ebm_contrastive_update_changes_weights():
    torch.manual_seed(42)
    model = EnergyModel(4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    real = torch.zeros(4, 3, 32, 32)
    negative = langevin_sample(model, torch.rand_like(real), 2)
    before = model.net[0].weight.detach().clone()
    real_e, negative_e = model(real), model(negative)
    loss = real_e.mean() - negative_e.mean() + 0.01 * (real_e.square().mean() + negative_e.square().mean())
    loss.backward()
    assert model.net[0].weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, model.net[0].weight)


def test_diffusion_forward_noise_and_training_gradient():
    torch.manual_seed(42)
    schedule = DiffusionSchedule(10)
    clean, noise = torch.zeros(2, 3, 32, 32), torch.ones(2, 3, 32, 32)
    t = torch.tensor([0, 9])
    noisy = schedule.q_sample(clean, t, noise)
    expected = (1 - schedule.alpha_bars[t]).sqrt()[:, None, None, None] * noise
    torch.testing.assert_close(noisy, expected)
    model = NoiseUNet(4)
    predicted = model(noisy, t)
    assert predicted.shape == noise.shape
    torch.nn.functional.mse_loss(predicted, noise).backward()
    assert model.enc1.conv1.weight.grad.abs().sum() > 0
    assert model.time_mlp[0].weight.grad.abs().sum() > 0
    model.eval().requires_grad_(False)
    samples = schedule.sample(model, 2, torch.Generator().manual_seed(42))
    assert samples.shape == clean.shape and torch.isfinite(samples).all()
    assert samples.min() >= -1 and samples.max() <= 1


def test_sinusoidal_values_and_defaults():
    expected = torch.tensor([[0.84147098, 0.54030231, 0.09983342, 0.99500417,
                              0.00999983, 0.99995, 0.001, 0.9999995]])
    torch.testing.assert_close(sinusoidal_embedding(torch.tensor([1]), 8), expected)
    torch.testing.assert_close(sinusoidal_embedding(torch.tensor([0]), 8), torch.tensor([[0.,1.,0.,1.,0.,1.,0.,1.]]))


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, '_models', {})
    for kind, cls in [('ebm', EnergyModel), ('diffusion', NoiseUNet)]:
        path = tmp_path / f'{kind}.pt'
        torch.save({'model': kind, 'channels': 4, 'diffusion_steps': 4,
                    'state_dict': cls(4).state_dict()}, path)
        monkeypatch.setenv(f'{kind.upper()}_CHECKPOINT', str(path))
    app = FastAPI()
    app.include_router(api.router)
    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize('endpoint', ['energy', 'diffusion'])
def test_png_inference_and_repeatable_seed(client, endpoint):
    body = {'count': 2, 'seed': 123}
    if endpoint == 'energy':
        body['steps'] = 3
    response = client.post(f'/generate/{endpoint}', json=body)
    assert response.status_code == 200
    result = response.json()
    assert result == client.post(f'/generate/{endpoint}', json=body).json()
    assert len(result['images_base64']) == 2
    for encoded in result['images_base64']:
        image = Image.open(io.BytesIO(base64.b64decode(encoded)))
        assert image.size == (32, 32) and image.mode == 'RGB' and image.format == 'PNG'


@pytest.mark.parametrize('endpoint,body', [('energy', {'count': 0}), ('energy', {'steps': 201}),
    ('energy', {'step_size': 0}), ('diffusion', {'count': 9}), ('diffusion', {'seed': -1})])
def test_invalid_requests(client, endpoint, body):
    assert client.post(f'/generate/{endpoint}', json=body).status_code == 422


@pytest.mark.parametrize('kind,endpoint', [('ebm', 'energy'), ('diffusion', 'diffusion')])
def test_missing_checkpoint_returns_503(client, monkeypatch, kind, endpoint):
    monkeypatch.setenv(f'{kind.upper()}_CHECKPOINT', '/nonexistent/model.pt')
    assert client.post(f'/generate/{endpoint}', json={}).status_code == 503
