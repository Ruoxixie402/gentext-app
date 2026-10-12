# gentext-app

A FastAPI application for text generation and word embedding queries.

## Assignment 2: CNN image classifier

The API also classifies PNG/JPEG uploads into the ten CIFAR-10 classes:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.
This is a small educational classifier for CIFAR-10-style images; arbitrary
photos outside that dataset may be misclassified. Softmax confidence is not a
calibrated probability of correctness.

`app/cnn.py` implements the specified architecture exactly: RGB 64×64 input,
3×3 convolutions with 16 then 32 filters (stride 1, padding 1), each followed
by ReLU and 2×2/stride-2 max pooling, flatten, linear 8,192→100, ReLU,
linear 100→10. It contains 825,398 trainable parameters. CIFAR-10's 32×32
images are resized to 64×64 and normalized with channel mean/std 0.5.
Training and inference share the same preprocessing.

### Train and evaluate

Install locked dependencies, including CPU PyTorch, then obtain the dataset:

```bash
uv sync --frozen
uv run --frozen python -m app.download_cifar10
uv run --frozen python -m app.train_classifier --epochs 5 --threads 4
```

The download helper uses a pinned GitHub mirror and verifies every original
batch against torchvision's published checksums before any dataset is loaded.
An alternative is `python -m app.train_classifier --download`, which downloads
from the official Toronto host when network access allows it.
Data is stored in ignored `data/`; no dataset files are committed.

The seeded split uses 45,000 training and 5,000 validation images.
Adam (learning rate 0.001), batch size 128, and cross-entropy loss train the
network. The checkpoint with the highest validation accuracy is saved to
`models/cifar10.pt`. After selection, it is evaluated on the untouched
10,000-image test set. Epoch history and measured test accuracy are written
to `models/cifar10.metrics.json`. CUDA is used automatically if available;
the default locked installation supports CPU execution.

The committed checkpoint was trained for five epochs on CPU with seed 42.
Epoch 5 had the best validation accuracy, **64.88%**; the held-out test
accuracy was **65.08%** (6,508 of 10,000 images). The complete epoch history
is committed in `models/cifar10.metrics.json` alongside the trained weights.

### Classify an image

Start the existing API with `uv run --frozen uvicorn app.main:app --reload`.
Send a multipart upload using the field `image`:

```bash
curl -X POST http://127.0.0.1:8000/classify \
  -F 'image=@path/to/image.png'
```

The JSON response contains `class`, `class_id` (0–9), `confidence`, and
`probabilities` mapping all ten class names to softmax values. The class is
the largest output probability. PNG/JPEG inputs may be RGB, grayscale, or
RGBA; all are converted to RGB. Uploads are limited to 5 MiB and 16 million
pixels. Invalid images return 400, oversized uploads return 413, and missing
uploads return 422. The service returns 503 if no trained checkpoint exists;
it never substitutes random weights. Set `CNN_CHECKPOINT` to use a different
compatible checkpoint, or `CNN_NUM_THREADS` to change inference CPU parallelism.
Restart the server after replacing its checkpoint.

### Verify and review the theory answers

```bash
uv run --frozen python -m pytest tests -q
```

Classifier unit tests use deterministic synthetic weights to verify API
behavior; model accuracy comes from the real held-out dataset evaluation.
The full worked solutions to all six arithmetic questions and an architecture
shape/parameter table are in [docs/assignment2-answers.md](docs/assignment2-answers.md).

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
