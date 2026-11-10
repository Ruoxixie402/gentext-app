# Assignment 4 verification record

## Training artifacts

- Energy CNN: 3 complete epochs on all 50,000 CIFAR-10 training images per epoch.
- Diffusion UNet: checkpoint and sample grid saved after 2 complete epochs on
  all 50,000 training images per epoch. A longer run was stopped after this save;
  the committed inference weights are the completed epoch-2 checkpoint.
- Both use seed 42, base channels 16, batch size 128, Adam learning rate 0.0002,
  and CPU training. Exact histories and other configuration are in the JSON reports.
- CIFAR-10 downloaded from the official Toronto source and passed torchvision checksums.

## Tests

`uv run --frozen python -m pytest tests -q`: **28 passed**.

This includes the pre-existing API, GAN, and Module 6 helper tests, plus the new
energy/diffusion gradient-control, schedule, training-gradient, and endpoint tests.
Questions 6–9 were also executed with `python -m app.cifar10.gradient_examples`;
actual output is recorded in `assignment4-gradient-output.txt`.

## Held-out diagnostics

A seeded subset of 512 CIFAR-10 test images (unused for optimization) produced:

- Diffusion noise-prediction MSE: 0.0859322.
- Untrained UNet on the same noisy inputs: MSE 1.0936747.
- EBM mean real-image energy: -49.7607.
- EBM mean uniform-noise energy: 56.1953.

Full settings and interpretation are in `artifacts/assignment4/validation.json`.
These diagnostics do not establish visual quality, exact likelihood, or convergence.
Sample grids show noise and color/texture structure, rather than reliable recognizable
CIFAR objects. This limitation is retained in the usage guide.

## Docker and live HTTP

The Docker image built successfully using the repository Docker configuration,
with a temporary build-only CA secret in an external Dockerfile to trust this
managed environment's HTTPS proxy. No session CA or credentials were committed.
The repository Dockerfile itself was unchanged. Tested on Linux x86-64 CPU.

A running container was queried through its published HTTP port:

- `POST /generate/energy`: HTTP 200; two valid 32x32 RGB PNGs.
- `POST /generate/diffusion`: HTTP 200; two valid 32x32 RGB PNGs.
- Both repeated the same response for identical seeds and request parameters.
- Invalid energy step count returned 422.
- `/health`, `/embedding`, `/generate`, and `/generate/digits` also succeeded.

The instructor's initial build requires Docker and internet access for dependencies.
Runtime inference uses committed weights and does not download the training data.
