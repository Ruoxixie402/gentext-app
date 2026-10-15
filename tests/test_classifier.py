"""Unit checks use synthetic weights; they do not measure CIFAR-10 accuracy."""

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image
import pytest
import torch
from torch import nn

from app.cnn import ARCHITECTURE, CLASSES, CIFAR10CNN, image_transform
from app.image_classifier import ImageClassifier, MAX_IMAGE_BYTES
from app.main import app


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    torch.set_num_threads(2)
    path = tmp_path_factory.mktemp("cnn") / "synthetic-test-only.pt"
    model = CIFAR10CNN()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.classifier[-1].bias[0] = 3
    torch.save({"architecture": ARCHITECTURE, "classes": list(CLASSES),
                "epoch": 1, "state_dict": model.state_dict()}, path)
    return path


@pytest.fixture(scope="module")
def classifier_client(checkpoint):
    # Preserve any caller-provided checkpoint path.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CNN_CHECKPOINT", str(checkpoint))
        with TestClient(app) as client:
            yield client


def image_bytes(mode="RGB", format="PNG"):
    buffer = BytesIO()
    Image.new(mode, (32, 32)).save(buffer, format=format)
    return buffer.getvalue()


def test_exact_architecture_and_shapes():
    model = CIFAR10CNN()
    assert sum(p.numel() for p in model.parameters()) == 825398
    assert [type(layer) for layer in model.features] == [
        nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.Conv2d, nn.ReLU, nn.MaxPool2d]
    assert [type(layer) for layer in model.classifier] == [nn.Flatten, nn.Linear, nn.ReLU, nn.Linear]
    shapes = []
    handles = [layer.register_forward_hook(lambda module, inputs, output: shapes.append(tuple(output.shape)))
               for layer in model.features]
    output = model(torch.zeros(2, 3, 64, 64))
    for handle in handles:
        handle.remove()
    assert shapes == [(2, 16, 64, 64), (2, 16, 64, 64), (2, 16, 32, 32),
                      (2, 32, 32, 32), (2, 32, 32, 32), (2, 32, 16, 16)]
    assert output.shape == (2, 10)
    for layer in (model.features[0], model.features[3]):
        assert layer.kernel_size == (3, 3)
        assert layer.stride == (1, 1)
        assert layer.padding == (1, 1)
    assert model.classifier[1].out_features == 100


def test_gradients_and_parameter_update():
    model = CIFAR10CNN()
    before = model.features[0].weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss = nn.CrossEntropyLoss()(model(torch.randn(2, 3, 64, 64)), torch.tensor([1, 2]))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert not torch.equal(before, model.features[0].weight)


def test_reject_wrong_model_input():
    with pytest.raises(ValueError, match="64"):
        CIFAR10CNN()(torch.zeros(1, 3, 32, 32))


def test_preprocessing():
    tensor = image_transform()(Image.new("RGB", (32, 32), color=(255, 255, 255)))
    assert tensor.shape == (3, 64, 64)
    assert torch.equal(tensor, torch.ones_like(tensor))


@pytest.mark.parametrize("mode,format", [("RGB", "PNG"), ("RGB", "JPEG"), ("L", "PNG"), ("RGBA", "PNG")])
def test_classify_upload(classifier_client, mode, format):
    response = classifier_client.post("/classify", files={"image": ("test", image_bytes(mode, format))})
    assert response.status_code == 200
    body = response.json()
    assert body["class"] == "airplane"
    assert body["class_id"] == 0
    assert 0 < body["confidence"] <= 1
    assert tuple(body["probabilities"]) == CLASSES
    assert sum(body["probabilities"].values()) == pytest.approx(1)


@pytest.mark.parametrize("data", [b"", b"not an image", image_bytes(format="GIF"), image_bytes()[:20]])
def test_invalid_image(classifier_client, data):
    response = classifier_client.post("/classify", files={"image": ("test.png", data, "image/png")})
    assert response.status_code == 400


def test_too_many_pixels(classifier_client):
    buffer = BytesIO()
    Image.new("RGB", (4096, 4096)).save(buffer, format="PNG")
    response = classifier_client.post("/classify", files={"image": ("large.png", buffer.getvalue())})
    assert response.status_code == 400


def test_oversized_upload(classifier_client):
    response = classifier_client.post("/classify", files={"image": ("huge.png", b"x" * (MAX_IMAGE_BYTES + 1))})
    assert response.status_code == 413


def test_missing_upload(classifier_client):
    assert classifier_client.post("/classify").status_code == 422


def test_missing_checkpoint_returns_503(classifier_client, monkeypatch):
    monkeypatch.setattr(app.state, "image_classifier", None)
    response = classifier_client.post("/classify", files={"image": ("test.png", image_bytes())})
    assert response.status_code == 503
    assert classifier_client.get("/health").status_code == 200


def test_untrained_checkpoint_rejected(checkpoint, tmp_path):
    saved = torch.load(checkpoint, weights_only=True)
    saved["epoch"] = 0
    path = tmp_path / "untrained.pt"
    torch.save(saved, path)
    with pytest.raises(ValueError, match="trained"):
        ImageClassifier(path)
