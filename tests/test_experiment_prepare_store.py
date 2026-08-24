"""Tests for experiments/sieve_experiments/prepare_store.py's download
verification. No network, no store: ``urllib.request.urlopen`` is monkeypatched
to serve fixed bytes, so these run in the fast suite.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from io import BytesIO

import pytest


@contextmanager
def _fake_response(payload: bytes):
    stream = BytesIO(payload)

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n):
            return stream.read(n)

    yield _Resp()


def test_download_chaos_store_rejects_wrong_published_md5(tmp_path, monkeypatch):
    """Zenodo publishes an md5 checksum for chaos-store.zip; a download whose
    bytes don't match it must be rejected, not silently accepted."""
    from sieve_experiments import prepare_store

    payload = b"not the real zip content"
    monkeypatch.setattr(
        prepare_store.urllib.request, "urlopen", lambda url: _fake_response(payload)
    )
    monkeypatch.setattr(
        prepare_store, "EXPECTED_ZIP_MD5", "0" * 32
    )  # deliberately wrong

    with pytest.raises(ValueError, match="md5"):
        prepare_store.download_chaos_store(tmp_path, store_dirname="chaos-store")

    assert not (tmp_path / "chaos-store.zip").exists()


def test_download_chaos_store_accepts_matching_published_md5(tmp_path, monkeypatch):
    from sieve_experiments import prepare_store
    from sieve_experiments.prepare_store import zipfile

    payload = b"pretend zip bytes"
    real_md5 = hashlib.md5(payload).hexdigest()

    monkeypatch.setattr(
        prepare_store.urllib.request, "urlopen", lambda url: _fake_response(payload)
    )
    monkeypatch.setattr(prepare_store, "EXPECTED_ZIP_MD5", real_md5)

    class _FakeZipFile:
        def __init__(self, path, mode):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extractall(self, dest):
            (dest / "chaos-store").mkdir(exist_ok=True)

    monkeypatch.setattr(zipfile, "ZipFile", _FakeZipFile)

    prepare_store.download_chaos_store(tmp_path, store_dirname="chaos-store")

    sha_path = tmp_path / "chaos-store" / ".download.sha256"
    assert sha_path.exists()
