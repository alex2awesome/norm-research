from __future__ import annotations

from scripts.tools.silver_match_v3.select_humor_fresh_release_relocated import (
    _replace_artifact,
)


def test_relocation_replaces_only_exact_path_and_identity() -> None:
    value = {
        "implementations": [
            {"path": "/repo/a.py", "bytes": 3, "sha256": "abc"},
            {"path": "/repo/b.py", "bytes": 4, "sha256": "def"},
        ]
    }
    relocation = {
        "original": {"path": "/repo/a.py", "bytes": 3, "sha256": "abc"},
        "relocated": {"path": "/archive/a.py", "bytes": 3, "sha256": "abc"},
    }
    assert _replace_artifact(value, relocation) == 1
    assert value["implementations"][0]["path"] == "/archive/a.py"
    assert value["implementations"][1]["path"] == "/repo/b.py"
