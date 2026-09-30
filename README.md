# gentext-app

A FastAPI application for text generation and word embedding queries.

## Requirements

- Python
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
  -d '{"prompt":"the","max_tokens":10}'
```

### Query a Word Embedding

Send a GET request to the embedding endpoint:

```bash
curl "http://127.0.0.1:8000/embedding?word=apple"
```

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

## Project Files

The repository includes the API source code and the required project configuration files:

- API source code
- `pyproject.toml`
- `uv.lock`
- `Dockerfile`
- `README.md`
