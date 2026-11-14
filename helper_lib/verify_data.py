"""Verify actual dataset files and save real input-image grids for instructor review.

uv run python -m helper_lib.verify_data --dataset all --no-download
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import torch
from torchvision import datasets
from torchvision.datasets.utils import check_integrity
from torchvision.utils import save_image

from .dataloader import get_mnist_dataloader, get_cifar10_dataloader


def sha256_stream(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(block)
    return digest.hexdigest()


def checked_sources(dataset):
    sources = []
    if isinstance(dataset, datasets.MNIST):
        prefix = 'train' if dataset.train else 't10k'
        for filename, expected_md5 in dataset.resources:
            if not filename.startswith(prefix):
                continue
            archive = Path(dataset.raw_folder) / filename
            raw = archive.with_suffix('')
            with raw.open('rb') as stream:
                raw_sha = sha256_stream(stream)
            item = {'file': f'MNIST/raw/{raw.name}', 'sha256': raw_sha}
            if archive.is_file():
                if not check_integrity(str(archive), expected_md5):
                    raise ValueError(f'Official MNIST gzip MD5 mismatch: {filename}')
                with gzip.open(archive, 'rb') as stream:
                    if sha256_stream(stream) != raw_sha:
                        raise ValueError(f'MNIST extracted content differs from archive: {raw.name}')
                item.update(official_gzip_md5=expected_md5,
                            checksum_status='passed; extracted file matches verified gzip')
            else:
                item['checksum_status'] = 'gzip unavailable; raw-file SHA256 recorded only'
            sources.append(item)
    else:
        files = dataset.train_list if dataset.train else dataset.test_list
        for filename, expected_md5 in files:
            path = Path(dataset.root) / dataset.base_folder / filename
            if not check_integrity(str(path), expected_md5):
                raise ValueError(f'Official CIFAR-10 batch MD5 mismatch: {filename}')
            with path.open('rb') as stream:
                digest = sha256_stream(stream)
            sources.append({'file': f'{dataset.base_folder}/{filename}',
                'official_md5': expected_md5, 'checksum_status': 'passed', 'sha256': digest})
    return sources


def verify(name, args):
    factory = get_mnist_dataloader if name == 'mnist' else get_cifar10_dataloader
    kwargs = {'train': args.split == 'train', 'download': not args.no_download, 'shuffle': False}
    if name == 'mnist':
        kwargs['mirror'] = args.mnist_mirror
    loader = factory(args.data_dir, 64, **kwargs)
    dataset = loader.dataset
    images, labels = next(iter(loader))
    expected_class = datasets.MNIST if name == 'mnist' else datasets.CIFAR10
    expected_size = (60000 if name == 'mnist' else 50000) if dataset.train else 10000
    expected_shape = (1, 28, 28) if name == 'mnist' else (3, 32, 32)
    if not isinstance(dataset, expected_class) or len(dataset) != expected_size:
        raise ValueError(f'Unexpected dataset type or size: {name}')
    if tuple(images.shape[1:]) != expected_shape or images.min() < -1 or images.max() > 1:
        raise ValueError(f'Unexpected normalized image shape/range: {name}')
    counts = torch.bincount(torch.as_tensor(dataset.targets), minlength=10).tolist()
    report = {'dataset': 'MNIST' if name == 'mnist' else 'CIFAR-10',
        'torchvision_dataset_class': f'torchvision.datasets.{expected_class.__name__}',
        'split': args.split, 'examples': len(dataset), 'image_shape': list(expected_shape),
        'dtype': str(images.dtype), 'batch_size': len(images),
        'batch_pixel_min': images.min().item(), 'batch_pixel_max': images.max().item(),
        'normalization': 'ToTensor then per-channel Normalize(mean=0.5,std=0.5)',
        'class_counts': {str(label): value for label, value in enumerate(counts)},
        'preview_labels_in_row_order': labels.tolist(),
        'preview_file': f'{name}-{args.split}-real-images.png',
        'loader': f'helper_lib/dataloader.py:{factory.__name__}',
        'source_files': checked_sources(dataset),
        'scope': 'Verification of data loaded now; not independent proof of historical checkpoint training.'}
    save_image(images, args.output_dir / report['preview_file'], nrow=8,
               normalize=True, value_range=(-1, 1))
    (args.output_dir / f'{name}-{args.split}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'dataset': report['dataset'], 'split': args.split,
                      'examples': len(dataset), 'shape': list(expected_shape),
                      'sources_checked': len(report['source_files'])}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['all', 'mnist', 'cifar10'], default='all')
    parser.add_argument('--split', choices=['train', 'test'], default='train')
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/dataset-verification'))
    parser.add_argument('--no-download', action='store_true')
    parser.add_argument('--mnist-mirror')
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in (['mnist', 'cifar10'] if args.dataset == 'all' else [args.dataset]):
        verify(name, args)


if __name__ == '__main__':
    main()
