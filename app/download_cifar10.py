"""Download original CIFAR-10 batches from a pinned, checksum-verified mirror."""

import argparse
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen

from torchvision.datasets import CIFAR10
from torchvision.datasets.utils import check_integrity

# An accessible mirror of the original Toronto dataset, not re-encoded images.
# Validate every file against the authoritative checksums shipped by torchvision.
MIRROR = "https://raw.githubusercontent.com/0Colin/cifar-10-batches-py/48ece5ce490482885f7a9a492fa031c1de79519c"


def download(data_dir):
    target = Path(data_dir) / CIFAR10.base_folder
    target.mkdir(parents=True, exist_ok=True)
    files = CIFAR10.train_list + CIFAR10.test_list + [
        [CIFAR10.meta["filename"], CIFAR10.meta["md5"]],
    ]
    for filename, checksum in files:
        destination = target / filename
        if check_integrity(destination, checksum):
            print(f"Verified existing {filename}", flush=True)
            continue
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target, delete=False) as stream:
                temporary = Path(stream.name)
                # Keep standard HTTPS certificate validation enabled.
                with urlopen(f"{MIRROR}/{filename}", timeout=120) as response:
                    shutil.copyfileobj(response, stream)
            if not check_integrity(temporary, checksum):
                raise ValueError(f"CIFAR-10 checksum mismatch: {filename}")
            temporary.replace(destination)
            print(f"Downloaded and verified {filename}", flush=True)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    download(args.data_dir)


if __name__ == "__main__":
    main()
