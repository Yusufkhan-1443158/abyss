"""Pickle loading with module-path remapping.

The pretrained bundles were pickled from the original training package
(`backend.uae_pretrained*`); this maps those module paths onto the vendored
modules so the bundles load without the original package installed.
"""
from __future__ import annotations

import gzip
import pickle

_MODULE_MAP = {
    "backend.uae_pretrained": "sdb_engine.uae_rf",
    "uae_pretrained": "sdb_engine.uae_rf",
    "backend.uae_pretrained_cnn": "sdb_engine.uae_cnn",
    "uae_pretrained_cnn": "sdb_engine.uae_cnn",
}


class _RemapUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        module = _MODULE_MAP.get(module, module)
        return super().find_class(module, name)


def load_pickle(path):
    """Load a (possibly gzip-compressed) pickle with module remapping."""
    with open(path, "rb") as fh:
        magic = fh.read(2)
    opener = gzip.open if magic == b"\x1f\x8b" else open
    with opener(path, "rb") as fh:
        return _RemapUnpickler(fh).load()
