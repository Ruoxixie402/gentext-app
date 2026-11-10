import base64
import io
import os
from pathlib import Path
from threading import Lock

import torch
from fastapi import APIRouter, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

from .models import EnergyModel, NoiseUNet
from .sampling import DiffusionSchedule, langevin_sample

router = APIRouter(prefix='/generate', tags=['Assignment 4: CIFAR-10'])
ROOT = Path(__file__).resolve().parents[2] / 'artifacts' / 'assignment4'
_models = {}
_lock = Lock()
# EBM sampling temporarily changes requires_grad flags, so serialize its requests.
_ebm_lock = Lock()


class ImageRequest(BaseModel):
    count: int = Field(default=1, ge=1, le=8)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)


class EnergyRequest(ImageRequest):
    steps: int = Field(default=100, ge=1, le=200)
    step_size: float = Field(default=0.01, gt=0, le=0.1)


def load_model(kind):
    with _lock:
        if kind not in _models:
            path = Path(os.getenv(f'{kind.upper()}_CHECKPOINT', str(ROOT / f'{kind}.pt')))
            if not path.is_file():
                raise HTTPException(503, f'{kind} checkpoint missing. Run python -m app.cifar10.train --model {kind}.')
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            if checkpoint['model'] != kind:
                raise RuntimeError('Checkpoint model type does not match endpoint')
            cls = EnergyModel if kind == 'ebm' else NoiseUNet
            model = cls(checkpoint['channels'])
            model.load_state_dict(checkpoint['state_dict'])
            model.eval().requires_grad_(False)
            schedule = DiffusionSchedule(checkpoint['diffusion_steps']) if kind == 'diffusion' else None
            _models[kind] = model, schedule
        return _models[kind]


def rng_for(seed):
    rng = torch.Generator(device='cpu')
    rng.seed() if seed is None else rng.manual_seed(seed)
    return rng


def encode(images, kind):
    pixels = ((images.detach().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    encoded = []
    for pixel in pixels:
        buffer = io.BytesIO()
        Image.fromarray(pixel.permute(1, 2, 0).contiguous().numpy()).save(buffer, format='PNG')
        encoded.append(base64.b64encode(buffer.getvalue()).decode('ascii'))
    return {'model': kind, 'count': len(images), 'width': 32, 'height': 32,
            'format': 'png', 'images_base64': encoded}


@router.post('/energy')
def generate_energy(request: EnergyRequest):
    """Generate RGB images by updating image tensors using Langevin dynamics."""
    model, _ = load_model('ebm')
    rng = rng_for(request.seed)
    initial = torch.rand((request.count, 3, 32, 32), generator=rng) * 2 - 1
    with _ebm_lock:
        images = langevin_sample(model, initial, request.steps, request.step_size, rng)
    return encode(images, 'ebm')


@router.post('/diffusion')
def generate_diffusion(request: ImageRequest):
    """Generate RGB images using every reverse step in the trained DDPM schedule."""
    model, schedule = load_model('diffusion')
    images = schedule.sample(model, request.count, rng_for(request.seed))
    return encode(images, 'diffusion')
