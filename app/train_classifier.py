"""Train on CIFAR-10, select on validation data, then evaluate the test set."""

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import random_split

from helper_lib.model import ARCHITECTURE, get_model
from helper_lib.data_loader import get_data_loader
from helper_lib.trainer import train_model
from helper_lib.evaluator import evaluate_model
from helper_lib.utils import save_model, load_checkpoint
from app.image_classifier import DEFAULT_CHECKPOINT


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
    dataset = get_data_loader(args.data_dir, args.batch_size, download=args.download).dataset
    training, validation = random_split(dataset, [45000, 5000], generator=torch.Generator().manual_seed(args.seed))
    train_loader = get_data_loader(args.data_dir, args.batch_size, dataset=training)
    val_loader = get_data_loader(args.data_dir, args.batch_size, train=False, dataset=validation)
    model = get_model("CNN").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    history = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def on_epoch_end(epoch, train_loss):
        nonlocal best_accuracy
        _, accuracy = evaluate_model(model, val_loader, loss_function, device)
        record = {"epoch": epoch, "train_loss": train_loss, "validation_accuracy": accuracy}
        history.append(record)
        print(json.dumps(record), flush=True)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_model(model, args.output, epoch=epoch, seed=args.seed, validation_accuracy=accuracy)

    train_model(model, train_loader, loss_function, optimizer, device,
                args.epochs, on_epoch_end=on_epoch_end)
    saved = load_checkpoint(args.output, device)
    model.load_state_dict(saved["state_dict"])
    test_loader = get_data_loader(args.data_dir, args.batch_size, train=False, download=args.download)
    test_loss, test_accuracy = evaluate_model(model, test_loader, loss_function, device)
    report = {
        "architecture": ARCHITECTURE, "seed": args.seed, "device": str(device),
        "training_images": len(training), "validation_images": len(validation),
        "test_images": len(test_loader.dataset), "best_epoch": saved["epoch"],
        "test_accuracy": test_accuracy, "test_loss": test_loss, "history": history,
    }
    args.output.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"test_accuracy": test_accuracy, "checkpoint": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
