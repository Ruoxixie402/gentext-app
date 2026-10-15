"""CIFAR-10 loading and shared RGB 64x64 preprocessing."""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def image_transform():
    # CIFAR-10 images are 32x32; the assignment architecture requires 64x64.
    return transforms.Compose([
        transforms.Resize((64, 64), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_data_loader(data_dir, batch_size=32, train=True, *, dataset=None, download=False):
    """data_dir is the CIFAR-10 root; dataset optionally supplies a validation subset."""
    if dataset is None:
        dataset = datasets.CIFAR10(data_dir, train=train, download=download, transform=image_transform())
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)
