"""Integration checks on genuine cached datasets; never download during pytest.

For a fresh checkout, run python -m helper_lib.verify_data first to fetch the data.
"""
from pathlib import Path

import pytest
import torch
from torchvision import datasets
from helper_lib.dataloader import get_mnist_dataloader, get_cifar10_dataloader

DATA = Path(__file__).resolve().parents[1] / 'data'


@pytest.mark.parametrize('factory,kind,train,size,shape', [
    (get_mnist_dataloader, datasets.MNIST, True, 60000, (1,28,28)),
    (get_mnist_dataloader, datasets.MNIST, False, 10000, (1,28,28)),
    (get_cifar10_dataloader, datasets.CIFAR10, True, 50000, (3,32,32)),
    (get_cifar10_dataloader, datasets.CIFAR10, False, 10000, (3,32,32)),
])
def test_real_cached_dataset(factory, kind, train, size, shape):
    try:
        loader = factory(DATA, 16, train=train, download=False, shuffle=False)
    except RuntimeError as error:
        if 'not found' in str(error).lower() or 'corrupted' in str(error).lower():
            pytest.skip('Dataset cache unavailable; run python -m helper_lib.verify_data first')
        raise
    assert isinstance(loader.dataset, kind)
    assert len(loader.dataset) == size
    assert loader.dataset.train is train
    assert loader.drop_last is False
    images, labels = next(iter(loader))
    assert images.shape == (16, *shape)
    assert images.dtype == torch.float32
    assert -1 <= images.min() <= images.max() <= 1
    assert labels.shape == (16,) and labels.dtype == torch.int64
    assert labels.min() >= 0 and labels.max() <= 9
    raw, target = loader.dataset.data[0], loader.dataset.targets[0]
    raw = torch.as_tensor(raw).float()
    expected = raw.unsqueeze(0) if kind is datasets.MNIST else raw.permute(2,0,1)
    torch.testing.assert_close(images[0], expected / 127.5 - 1)
    assert labels[0].item() == int(target)


def test_mnist_mirror_uses_same_verified_cache():
    if not (DATA / 'MNIST' / 'raw').is_dir():
        pytest.skip('MNIST cache unavailable')
    loader = get_mnist_dataloader(DATA, 2, download=False,
        mirror='https://raw.githubusercontent.com/fgnt/mnist/master/')
    assert isinstance(loader.dataset, datasets.MNIST)
    assert Path(loader.dataset.raw_folder) == DATA / 'MNIST' / 'raw'
    assert len(loader.dataset) == 60000
