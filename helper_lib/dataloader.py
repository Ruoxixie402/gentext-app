"""Real torchvision datasets and DataLoaders used by the assignment trainers.

MNIST: grayscale digits, (1, 28, 28), 60,000 train / 10,000 test.
CIFAR-10: RGB objects, (3, 32, 32), 50,000 train / 10,000 test.
Both loaders return (images, labels), with images normalized to [-1, 1].
No synthetic data or FakeData is used in this module.
"""
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class _MirroredMNIST(datasets.MNIST):
    """Optional download mirror while retaining torchvision's MNIST cache paths."""
    def __init__(self, root, mirror, **kwargs):
        self.mirrors = [mirror.rstrip('/') + '/']
        super().__init__(root, **kwargs)

    @property
    def raw_folder(self):
        return str(Path(self.root) / 'MNIST' / 'raw')

    @property
    def processed_folder(self):
        return str(Path(self.root) / 'MNIST' / 'processed')


def get_mnist_dataset(data_dir='data', *, train=True, download=True, mirror=None):
    """Load real MNIST; torchvision checks downloaded gzip-file MD5 checksums."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    if mirror:
        return _MirroredMNIST(data_dir, mirror, train=train, download=download, transform=transform)
    return datasets.MNIST(data_dir, train=train, download=download, transform=transform)


def get_cifar10_dataset(data_dir='data', *, train=True, download=True):
    """Load real CIFAR-10; torchvision verifies its official batch-file checksums."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return datasets.CIFAR10(data_dir, train=train, download=download, transform=transform)


def get_mnist_dataloader(data_dir='data', batch_size=128, *, train=True,
                         download=True, shuffle=None, mirror=None):
    """Return real MNIST batches; shuffle training data by default, keep all images."""
    dataset = get_mnist_dataset(data_dir, train=train, download=download, mirror=mirror)
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=train if shuffle is None else shuffle, drop_last=False)


def get_cifar10_dataloader(data_dir='data', batch_size=128, *, train=True,
                           download=True, shuffle=None):
    """Return real CIFAR-10 batches; shuffle training data, never use test for fitting."""
    dataset = get_cifar10_dataset(data_dir, train=train, download=download)
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=train if shuffle is None else shuffle, drop_last=False)
