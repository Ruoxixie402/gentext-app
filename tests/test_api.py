import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


def test_pretrained_word_embedding(client):
    response = client.get("/embedding", params={"word": "apple"})
    assert response.status_code == 200
    body = response.json()
    assert body["dimension"] == len(body["embedding"]) == 300
    assert np.isfinite(body["embedding"]).all()
    assert np.linalg.norm(body["embedding"]) > 0
    np.testing.assert_allclose(body["embedding"], app.state.nlp.vocab["apple"].vector)


@pytest.mark.parametrize("word,status", [("", 422), ("   ", 400), ("two words", 400), ("zzzxxyyqqqnonword", 404)])
def test_invalid_embedding_input(client, word, status):
    assert client.get("/embedding", params={"word": word}).status_code == status


def test_generate_with_documented_parameters(client):
    response = client.post("/generate", json={"start_word": "The", "length": 3})
    assert response.status_code == 200
    assert response.json()["generated_text"] == "The Count of"


def test_health(client):
    assert client.get("/health").json() == {"status": "healthy"}
