"""Train on CIFAR-10, select on validation data, then evaluate the test set."""

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10

from app.cnn import ARCHITECTURE, CLASSES, CIFAR10CNN, image_transform
from app.image_classifier import DEFAULT_CHECKPOINT


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(dim=1)
            correct += int((predictions == labels.to(device)).sum())
            total += len(labels)
    return correct / total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1:
        parser.error("epochs, batch-size and threads must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CIFAR10(args.data_dir, train=True, download=args.download, transform=image_transform())
    training, validation = random_split(dataset, [45000, 5000], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(validation, batch_size=args.batch_size)
    model = CIFAR10CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    history = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
        accuracy = evaluate(model, val_loader, device)
        record = {"epoch": epoch, "train_loss": loss_sum / len(training), "validation_accuracy": accuracy}
        history.append(record)
        print(json.dumps(record), flush=True)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                "architecture": ARCHITECTURE, "classes": list(CLASSES),
                "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
                "epoch": epoch, "seed": args.seed, "validation_accuracy": accuracy,
            }, args.output)
    saved = torch.load(args.output, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    test_data = CIFAR10(args.data_dir, train=False, download=args.download, transform=image_transform())
    test_accuracy = evaluate(model, DataLoader(test_data, batch_size=args.batch_size), device)
    report = {
        "architecture": ARCHITECTURE, "seed": args.seed, "device": str(device),
        "training_images": len(training), "validation_images": len(validation),
        "test_images": len(test_data), "best_epoch": saved["epoch"],
        "test_accuracy": test_accuracy, "history": history,
    }
    args.output.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"test_accuracy": test_accuracy, "checkpoint": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
