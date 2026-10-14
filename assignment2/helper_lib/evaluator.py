"""Sample-weighted classification loss and accuracy."""

import torch


def evaluate_model(model, data_loader, criterion, device='cpu'):
    """Return (average loss, accuracy) using a scalar mean-loss criterion."""
    model.to(device)
    model.eval()
    total_loss = correct = count = 0
    with torch.inference_mode():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * len(labels)
            correct += int((logits.argmax(dim=1) == labels).sum())
            count += len(labels)
    if not count:
        raise ValueError("Evaluation data loader is empty")
    return total_loss / count, correct / count
