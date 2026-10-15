"""Reusable supervised classification training."""

import torch


def train_model(model, data_loader, criterion, optimizer, device='cpu', epochs=10,
                *, on_epoch_end=None):
    """Return the trained model; callback receives (epoch, mean training loss).

    criterion must return a scalar mean loss. The optional callback supports
    validation and checkpoint selection without coupling this helper to a dataset.
    """
    if epochs < 1:
        raise ValueError("epochs must be positive")
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = count = 0
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            count += len(labels)
        if not count:
            raise ValueError("Training data loader is empty")
        if on_epoch_end is not None:
            on_epoch_end(epoch, total_loss / count)
    return model
