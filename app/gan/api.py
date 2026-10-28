import base64
import io
import os
from pathlib import Path
from threading import Lock

import torch
from fastapi import APIRouter, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

from .models import Generator

router = APIRouter()
_model = None
_lock = Lock()
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / 'artifacts' / 'generator.pt'


class DigitRequest(BaseModel):
    count: int = Field(default=1, ge=1, le=64)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)


def load_generator():
    global _model
    with _lock:
        if _model is None:
            path = Path(os.getenv('GAN_CHECKPOINT', str(DEFAULT_CHECKPOINT)))
            if not path.is_file():
                raise HTTPException(status_code=503, detail='GAN checkpoint missing. Run python -m app.gan.train first.')
            model = Generator()
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            model.load_state_dict(checkpoint['generator'])
            _model = model.eval()
    return _model


@router.post('/generate/digits')
def generate_digits(request: DigitRequest):
    """Generate unconditional MNIST-style digits as base64-encoded grayscale PNGs."""
    model = load_generator()
    rng = torch.Generator(device='cpu')
    if request.seed is None:
        rng.seed()
    else:
        rng.manual_seed(request.seed)
    with torch.inference_mode():
        images = model(torch.randn(request.count, 100, generator=rng))
    pixels = ((images.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    encoded = []
    for image in pixels:
        buffer = io.BytesIO()
        Image.fromarray(image.squeeze(0).numpy()).save(buffer, format='PNG')
        encoded.append(base64.b64encode(buffer.getvalue()).decode('ascii'))
    return {'count': request.count, 'width': 28, 'height': 28, 'format': 'png', 'images_base64': encoded}
