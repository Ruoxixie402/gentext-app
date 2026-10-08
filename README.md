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
