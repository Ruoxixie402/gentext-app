"""CIFAR-10 checkpoint persistence shared by training and the API."""

from pathlib import Path
import torch

from helper_lib.model import ARCHITECTURE, CLASSES, get_model


def save_model(model, path, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        **metadata, "architecture": ARCHITECTURE, "classes": list(CLASSES),
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
    }, path)


def load_checkpoint(path, device='cpu'):
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved.get("architecture") != ARCHITECTURE or tuple(saved.get("classes", ())) != CLASSES:
        raise ValueError("Incompatible CIFAR-10 checkpoint metadata.")
    if saved.get("epoch", 0) < 1:
        raise ValueError("The checkpoint must contain trained weights.")
    return saved


def load_model(path, device='cpu'):
    saved = load_checkpoint(path, device)
    model = get_model("CNN").to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    return model
