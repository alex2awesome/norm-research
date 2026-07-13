"""Reproducible environment fingerprint for reconstruction-v2 manifests."""

from __future__ import annotations

import hashlib
from importlib import metadata
import json
import pathlib
import platform
import sys


PACKAGES = (
    "numpy",
    "scikit-learn",
    "scipy",
    "sympy",
    "spacy",
    "networkx",
    "python-dateutil",
    "pylatexenc",
)


def environment_fingerprint() -> dict:
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    try:
        packages["en-core-web-sm"] = metadata.version("en-core-web-sm")
    except metadata.PackageNotFoundError:
        packages["en-core-web-sm"] = None
    payload = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "executable": str(pathlib.Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "packages": packages,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


if __name__ == "__main__":
    print(json.dumps(environment_fingerprint(), indent=2, sort_keys=True))

