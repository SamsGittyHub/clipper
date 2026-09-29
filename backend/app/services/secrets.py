"""Encryption for OAuth tokens at rest (PRD 12).

A local key file holds a Fernet key; tokens are stored as ciphertext in the
database so a copied `clipz.db` is not a set of working credentials. The
key lives beside it with 0600 permissions -- this raises the bar, it does not
defend against someone who already has read access to the whole storage tree.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from ..config import settings

log = logging.getLogger("clipz.secrets")


def key_path() -> Path:
    return settings.storage_dir / "token.key"


@lru_cache(maxsize=1)
def _cipher() -> Fernet:
    path = key_path()
    if path.exists():
        key = path.read_bytes().strip()
    else:
        key = Fernet.generate_key()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(key)
        os.chmod(path, 0o600)
        log.info("Created token encryption key at %s", path)
    return Fernet(key)


def encrypt(value: str) -> str:
    return _cipher().encrypt(value.encode()).decode()


def decrypt(value: str) -> str:
    """Decrypt a stored token; raises ValueError if the key no longer matches."""
    try:
        return _cipher().decrypt(value.encode()).decode()
    except InvalidToken as exc:
        raise ValueError(
            "Stored token could not be decrypted -- the key file may have been "
            "replaced. Reconnect the account."
        ) from exc


def reset_cache() -> None:
    _cipher.cache_clear()
