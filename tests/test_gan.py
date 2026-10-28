import base64
import io

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from app.gan import api
from app.gan.models import Generator, Discriminator


def test_architecture_and_adversarial_gradients():
    torch.set_num_threads(2)
    generator, discriminator = Generator(), Discriminator()
    fake = generator(torch.randn(4, 100))
    assert fake.shape == (4, 1, 28, 28)
    assert fake.min() >= -1 and fake.max() <= 1
    probability = discriminator(fake)
    assert probability.shape == (4, 1)
    assert ((probability >= 0) & (probability <= 1)).all()
    torch.nn.functional.binary_cross_entropy(probability, torch.ones_like(probability)).backward()
    assert generator.layers[0].weight.grad.abs().sum() > 0
    assert discriminator.layers[0].weight.grad.abs().sum() > 0


@pytest.fixture
def client(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'generator.pt'
    torch.save({'generator': Generator().state_dict()}, checkpoint)
    monkeypatch.setenv('GAN_CHECKPOINT', str(checkpoint))
    monkeypatch.setattr(api, '_model', None)
    app = FastAPI()
    app.include_router(api.router)
    with TestClient(app) as client:
        yield client


def test_digit_png_and_seed(client):
    body = {'count': 2, 'seed': 123}
    response = client.post('/generate/digits', json=body)
    assert response.status_code == 200
    data = response.json()
    assert data == client.post('/generate/digits', json=body).json()
    assert len(data['images_base64']) == 2
    for value in data['images_base64']:
        image = Image.open(io.BytesIO(base64.b64decode(value)))
        assert image.size == (28, 28)
        assert image.mode == 'L'
        assert image.format == 'PNG'


@pytest.mark.parametrize('body', [{'count': 0}, {'count': 65}, {'seed': -1}])
def test_digit_validation(client, body):
    assert client.post('/generate/digits', json=body).status_code == 422


def test_missing_checkpoint(client, monkeypatch):
    monkeypatch.setenv('GAN_CHECKPOINT', '/nonexistent/generator.pt')
    assert client.post('/generate/digits', json={}).status_code == 503
