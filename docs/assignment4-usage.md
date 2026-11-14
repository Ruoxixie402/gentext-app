# Assignment 4 — CIFAR-10 energy and diffusion models

This submission implements both models, trains on all 50,000 CIFAR-10 training
images, integrates sampling into the existing Module 6 FastAPI app, and answers
all nine theory questions. Existing text, embedding, GAN, and helper-library
functionality remains available. Models are unconditional: no class selection.

## Installation and serving

Use Python 3.12 or 3.13 and run commands from the repository root:

```bash
uv sync --frozen
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The committed checkpoints in `artifacts/assignment4/` are loaded lazily on CPU.
Serving needs no CIFAR download and does not train on request. Missing weights
return HTTP 503 instead of using randomly initialized models. To override paths,
set `EBM_CHECKPOINT` and/or `DIFFUSION_CHECKPOINT` before starting the server;
restart after replacing weights. Only load trusted checkpoints.

## Query the models

```bash
curl -X POST http://127.0.0.1:8000/generate/energy \
  -H 'Content-Type: application/json' \
  -d '{"count":2,"seed":42,"steps":100,"step_size":0.01}'

curl -X POST http://127.0.0.1:8000/generate/diffusion \
  -H 'Content-Type: application/json' \
  -d '{"count":2,"seed":42}'
```

Both routes return `model`, `count`, `width` (32), `height` (32), `format` (png),
and `images_base64`. Each entry is a base64-encoded RGB PNG, not a file path.
`count` must be 1–8; seed is optional and must be between 0 and 2^32−1.
The energy endpoint accepts 1–200 Langevin steps and 0 < step_size <= 0.1;
these are numerical sampler parameters, not learned weights. The diffusion
endpoint runs all 100 reverse steps from the committed checkpoint; skipping
steps would require a separately derived sampling rule. Invalid requests return
422. A fixed seed is repeatable with the same weights, sampler settings, and
runtime. Hardware/version changes can affect results.

Decode and save a response:

```python
import base64
from pathlib import Path
import httpx

response = httpx.post('http://127.0.0.1:8000/generate/diffusion',
                      json={'count': 2, 'seed': 42}, timeout=120)
response.raise_for_status()
for i, encoded in enumerate(response.json()['images_base64']):
    Path(f'cifar-sample-{i}.png').write_bytes(base64.b64decode(encoded))
```

Swagger UI is at `/docs`. These are CPU-heavy educational endpoints, not a
high-throughput service. EBM sampling is serialized because it temporarily
manages parameter gradient flags. CPU thread counts can be limited when serving:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run uvicorn app.main:app
```

## Dataset preparation and reproducibility

The explicit data-loader file is `helper_lib/dataloader.py`. Both energy and
diffusion training call its `get_cifar10_dataloader` function. Held-out evaluation
calls `get_cifar10_dataset(train=False)`. Module 6 and Assignment 3 use its
separate MNIST loader. See the [data preparation and reproducibility report](dataset-verification.md).

```bash
uv run python -m helper_lib.verify_data --dataset cifar10
```

## Reproduce the training runs

CIFAR-10 downloads from its official Toronto source and torchvision checks the
archive and batch-file MD5 checksums. Internet access is required for the first
download. Inputs are RGB 32×32 tensors, normalized from [0,1] to [-1,1]. Class
labels are unused. The 10,000 test images are not used for optimization.

```bash
uv run python -m app.cifar10.train --model ebm --epochs 3 --channels 16 --threads 1
uv run python -m app.cifar10.train --model diffusion --epochs 2 --channels 16 --threads 1
```

Both runs use seed 42, batch size 128, and Adam learning rate 0.0002. Each epoch
uses the complete training split, including the final partial batch. Run reports
record epoch-average training losses, configuration, and elapsed times. The
checkpoints include weights and inference configuration, but do not include
optimizer/RNG/replay state for exact training resumption. Each epoch replaces
its model's checkpoint, report, and sample grid. The checkpoint files are
inference artifacts; continuing training starts a new model unless resume
support is explicitly added. Longer runs can be requested with `--epochs`.

### Energy model and gradient control

The energy CNN maps (B,3,32,32) through three stride-2 convolution/SiLU layers,
then a linear layer, to B scalar energies. Its modeled density is proportional
to exp(-E(x)), so lower energy corresponds to higher modeled density.

For a batch of negatives, the sampler:

1. Freezes network parameters while preserving their original flags and mode.
2. Detaches the image, enables its gradient, and computes grad_x sum(E(x)).
3. Updates x under `torch.no_grad()` with
   x <- x - eta*grad_x E(x) + sqrt(2*eta)*GaussianNoise.
4. Clamps x to [-1,1], detaches between iterations, and restores parameter flags.

Training uses ten Langevin steps at eta=0.01, a replay buffer of 1024 images,
and 5% fresh uniform restarts. Negatives remain detached when computing the
contrastive objective mean(E(real))-mean(E(negative)), plus 0.01 times the
sum of real/negative squared-energy means. `loss.backward()` updates model
weights; sampling never accumulates parameter gradients. Clipping the weight
gradient norm at 1 reduces large optimizer updates. API sampling starts from
fresh uniform noise, rather than using the training replay buffer.

Finite chains, clipping, and replay introduce approximation and mixing bias;
this is not exact maximum likelihood. Energy loss can be negative and is not
an image-quality metric. Stochastic steps need not decrease energy every step.

### Diffusion model

A compact time-conditioned UNet downsamples 32→16→8, upsamples 8→16→32, and
uses skip connections, GroupNorm, SiLU, and sinusoidal timestep embeddings.
It predicts RGB noise with the same shape as the input. A 100-step cosine
schedule sets beta_t, alpha_t=1-beta_t, and cumulative alpha_bar_t.

Training samples a timestep uniformly and Gaussian noise epsilon, constructs
x_t=sqrt(alpha_bar_t)*x_0+sqrt(1-alpha_bar_t)*epsilon, and minimizes mean squared
error between predicted and sampled noise. Only network parameters are updated.
Generation starts from Gaussian noise and follows the DDPM posterior mean and
variance using predicted clean images clipped to [-1,1]. No extra noise is added
at the final step. Sampling uses no parameter/input gradients.

## Quality and limitations

These are small CPU-trained educational models, not pretrained high-quality
CIFAR generators. Inspect `ebm-samples.png` and `diffusion-samples.png` alongside
the actual training reports. Short training and finite Langevin chains can leave
noise, color fields, and weak or absent recognizable objects. Successful API
inference does not establish perceptual quality or convergence. No FID,
Inception Score, or class-coverage claim is made. `validation.json` reports
noise-prediction MSE and real-versus-uniform energy on a seeded 512-image
subset of the held-out test split. An untrained UNet is evaluated on the same
noisy inputs as a reference. These diagnostics do not measure visual quality. The PDF specifies neither a
training duration nor a numerical quality target. More compute, longer training,
and model/sampler tuning would be needed to pursue high-fidelity generation.

## Theory and verification

- `docs/assignment4-answers.md`: all nine questions, calculations, corrected code,
  embedding convention, and gradient explanations.
- `app/cifar10/gradient_examples.py`: executable Questions 6–9.
- `docs/assignment4-gradient-output.txt`: captured output from those examples.
- `tests/test_cifar10.py`: image-gradient descent, frozen weights, contrastive
  weight updates, noise schedule, UNet/time-MLP gradients, numeric embeddings,
  RGB PNG decoding, repeatable seeds, validation, and missing checkpoints.

```bash
uv run python -m app.cifar10.gradient_examples
uv run python -m app.cifar10.evaluate --examples 512
uv run python -m pytest tests -q
```

## Docker

The existing Dockerfile includes the checkpoints and starts FastAPI on port 8000.
No training-data directory is copied because `.dockerignore` excludes it.

```bash
docker build -t gentext-assignment4 .
docker run --rm -p 8000:8000 -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 gentext-assignment4
```

Building requires internet access for the pinned dependencies and spaCy model.
Once built, model requests work without downloading data or additional weights.
