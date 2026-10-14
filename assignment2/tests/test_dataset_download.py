from io import BytesIO
import hashlib

import pytest

from app import download_cifar10


@pytest.fixture
def mirror(monkeypatch):
    payload = b"test-only dataset bytes, never unpickled"
    checksum = hashlib.md5(payload).hexdigest()
    monkeypatch.setattr(download_cifar10.CIFAR10, "train_list", [["batch", checksum]])
    monkeypatch.setattr(download_cifar10.CIFAR10, "test_list", [])
    monkeypatch.setattr(download_cifar10.CIFAR10, "meta", {"filename": "meta", "md5": checksum})
    return payload


def test_verified_download_is_repeatable(tmp_path, monkeypatch, mirror):
    requests = []

    def open_url(url, timeout):
        requests.append(url)
        assert url.startswith("https://raw.githubusercontent.com/")
        return BytesIO(mirror)

    monkeypatch.setattr(download_cifar10, "urlopen", open_url)
    download_cifar10.download(tmp_path)
    download_cifar10.download(tmp_path)
    assert len(requests) == 2  # two files; the second run makes no network calls
    assert (tmp_path / "cifar-10-batches-py" / "batch").read_bytes() == mirror


def test_checksum_failure_discards_download(tmp_path, monkeypatch, mirror):
    monkeypatch.setattr(download_cifar10, "urlopen", lambda *args, **kwargs: BytesIO(b"corrupt"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_cifar10.download(tmp_path)
    assert list((tmp_path / "cifar-10-batches-py").iterdir()) == []
