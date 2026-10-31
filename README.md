# gentext-app

A FastAPI application for text generation and word embedding queries.

## Requirements

- Python 3.12 or 3.13
- [uv](https://docs.astral.sh/uv/)
- Docker (optional)

## Install Dependencies

Clone the repository and enter the project directory:

```bash
git clone https://github.com/Ruoxixie402/gentext-app.git
cd gentext-app
```

Install the project dependencies:

```bash
uv sync
```

This installs FastAPI, Uvicorn, spaCy, and the pretrained `en_core_web_md`
English model from the committed `uv.lock`. The model includes 300-dimensional
word vectors. The first installation requires network access; API requests do not.

## Run the API Locally

Start the FastAPI server:

```bash
uv run uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive Swagger documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Query the API

### Generate Text

Send a POST request to the text-generation endpoint:

```bash
curl -X POST "http://127.0.0.1:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"start_word":"the","length":10}'
```

### Query a Word Embedding

Send a GET request to the embedding endpoint:

```bash
curl "http://127.0.0.1:8000/embedding?word=apple"
```

The response includes `word`, `model`, `dimension` (300), and `embedding`,
an array of 300 numbers from the pretrained spaCy model. Supply one word.
Empty or multiword inputs return HTTP 400 (an empty query string returns 422);
words without a pretrained vector return HTTP 404.

Text generation uses a small demonstration corpus and may stop before the
requested length if there is no next word. Start words are case-sensitive.

You can also test all available endpoints interactively at:

```text
http://127.0.0.1:8000/docs
```

## Run with Docker

Build the Docker image:

```bash
docker build -t gentext-app .
```

Run the container:

```bash
docker run -p 8000:8000 gentext-app
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## Verify the API

```bash
uv run python -m pytest tests -q
```

These tests load the real pretrained model and verify the vector response,
invalid-input handling, health check, and the documented text-generation request.

## Repository Contents

The repository includes the API source code and the required project configuration files:

- API source code
- `pyproject.toml`
- `uv.lock`
- `Dockerfile`
- `README.md`

## Assignment 3: MNIST GAN

The exact assignment architecture is in `app/gan/models.py`. The generator maps
100-dimensional Gaussian noise through a linear layer to 128×7×7, then uses
128→64→1 transposed convolutions (kernel 4, stride 2, padding 1), with BatchNorm
and ReLU after the first convolution and Tanh after the second. There is no
extra activation after the linear layer. The discriminator uses 1→64→128
convolutions with the specified BatchNorm and LeakyReLU(0.2), then a linear
layer and Sigmoid to return a real/fake probability.

### Train

```bash
uv sync --frozen
uv run python -m app.gan.train --epochs 20
```

MNIST pixels are normalized from [0,1] to [-1,1] to match Tanh. Each batch
alternates discriminator training on real images and detached generated images,
then generator training against real labels (the non-saturating GAN objective).
Training uses binary cross entropy and Adam with learning rate 0.0002 and
betas (0.5, 0.999). The discriminator is frozen, including BatchNorm statistics,
during the generator update. MNIST labels are unused: generation is
unconditional, so callers cannot select a particular digit.

The training script writes `artifacts/generator.pt` (inference weights and
metadata), `artifacts/training.json` (epoch-average losses and run settings), and
`artifacts/samples.png` (64 fixed-noise samples). Each epoch replaces these
files. The checkpoint is for inference; it does not include optimizer states
for resuming training. Losses alone do not measure image quality or digit coverage.
The included five-epoch sample grid contains recognizable digits with visible
noise and broken strokes. This is an early training checkpoint; longer training
can be explored, while checking samples for quality and diversity. No digit
classification accuracy or distribution-quality score has been measured.
The random seed helps reproduce a run; results can differ across hardware and
PyTorch versions. This lockfile uses CPU PyTorch wheels for a portable CPU API.

If the default MNIST download hosts are unavailable, an optional mirror can be
provided. To reproduce the five-epoch checkpoint included in this repository, use the
standard MNIST gzip files from this mirror:

```bash
uv run python -m app.gan.train --epochs 5 --threads 2 \
  --mnist-mirror https://raw.githubusercontent.com/fgnt/mnist/master/
```

Torchvision validates these downloads against its built-in MNIST checksums.
Downloaded data is ignored by Git. The trained generator is included so the
API can generate images without training or downloading data at startup.

### Generate digit images

```bash
uv run uvicorn app.main:app --reload
curl -X POST http://127.0.0.1:8000/generate/digits \
  -H 'Content-Type: application/json' \
  -d '{"count":4,"seed":42}'
```

The response contains `count`, `width`, `height`, `format`, and `images_base64`:
one base64-encoded grayscale PNG per image. Decode an entry with Python:

```python
import base64
from pathlib import Path
import httpx

response = httpx.post('http://127.0.0.1:8000/generate/digits',
                      json={'count': 4, 'seed': 42})
response.raise_for_status()
for i, encoded in enumerate(response.json()['images_base64']):
    Path(f'digit-{i}.png').write_bytes(base64.b64decode(encoded))
```

`count` must be 1–64. An optional integer `seed` from 0 to 2³²−1 makes requests
repeatable with the same weights and runtime. Invalid requests return 422.
A missing checkpoint returns 503; it never silently serves an untrained model.
Set `GAN_CHECKPOINT` before starting the server to use another compatible
checkpoint. The model is loaded on the first digit request and cached; restart
the server after changing the weights. Only load checkpoints you trust.

The existing `/generate`, `/embedding`, and `/health` routes remain available.
Theory answers with all calculations are in `docs/assignment3-answers.md`.

```bash
uv run python -m pytest tests -q
```

The GAN tests check output shapes/ranges, adversarial gradient flow, PNG decoding,
seed repeatability, request validation, and missing-checkpoint handling. The
existing API tests exercise the real pretrained spaCy embeddings as well.

## Module 6 Activity: GAN Helper Library

The `helper_lib` package provides the three requested reusable entry points:

- `helper_lib.model.get_model(model_name)`: supports `FCNN`, `CNN`, `EnhancedCNN`, `VAE`, and `GAN`. The GAN option returns `(generator, discriminator)` and reuses the exact Assignment 3 architecture. The other four options are MNIST-compatible reference models; compare their layers with any earlier Module 4 implementation if exact reuse is required.
- `helper_lib.trainer.train_gan(model, data_loader, criterion, optimizer, device='cpu', epochs=10)`: trains a GAN pair with separate generator and discriminator optimizers and returns epoch loss history. Supply a loader with images normalized to [-1,1].
- `helper_lib.generator.generate_samples(model, device, num_samples=10)`: creates a PIL image grid. In a notebook, use `display(grid)`; in a desktop session, use `grid.show()`.

Example:

```python
import torch
from torch import nn
from helper_lib import get_model, train_gan, generate_samples

generator, discriminator = get_model('GAN')
optimizers = (
    torch.optim.Adam(generator.parameters(), lr=0.0002),
    torch.optim.Adam(discriminator.parameters(), lr=0.0002),
)
# Use a prepared MNIST DataLoader with pixels normalized to [-1, 1]:
# history = train_gan((generator, discriminator), loader, nn.BCELoss(), optimizers, epochs=10)
image_grid = generate_samples(generator, 'cpu', num_samples=10)
image_grid.save('module6-samples.png')
```

The grid example above uses a newly initialized generator unless you train or load weights first. The existing `POST /generate/digits` endpoint and saved Assignment 3 checkpoint are unchanged. Run `uv run python -m pytest tests/test_module6.py -q` for helper-library smoke tests.
