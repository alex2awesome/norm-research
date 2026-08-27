from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from methods.metric_seam.family_scale.semantic_alignment_glm import (
    main,
    raw_response_is_degenerate,
)


REQUEST_SCHEMA = "metric-seam.three-fleet-semantic-alignment-request.v1"


def _request(stem: str) -> dict:
    return {
        "schema": REQUEST_SCHEMA,
        "status": "compiled_for_exactly_one_alignment_call",
        "model": "sonnet",
        "system_prompt": "SYS",
        "user_prompt": f"USER_{stem}",
        "request_sha256": f"sha_{stem}",
    }


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc) -> None:
        return None


def _api_body(text: str) -> bytes:
    return json.dumps({
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
    }).encode("utf-8")


def test_degeneracy_detector() -> None:
    assert raw_response_is_degenerate("")
    assert raw_response_is_degenerate("   ")
    assert raw_response_is_degenerate("!" * 3000)
    assert not raw_response_is_degenerate('{"clusters": [], "unmatched_unit_ids": ["u_a"]}')


def test_main_writes_manifest_and_raws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    req_dir = tmp_path / "requests"
    req_dir.mkdir()
    out_dir = tmp_path / "out"
    key_path = tmp_path / "key.txt"
    key_path.write_text("fake-key\n")
    for stem in ["a", "b"]:
        (req_dir / f"{stem}_request.json").write_text(json.dumps(_request(stem)))
    (req_dir / "._sidecar_request.json").write_text("garbage")

    def fake_urlopen(req, timeout=None):
        # Route response text by which user_prompt was sent.
        payload = json.loads(req.data.decode("utf-8"))
        stem = payload["messages"][0]["content"].removeprefix("USER_")
        return _FakeResponse(_api_body(f'{{"stem": "{stem}"}}'))

    monkeypatch.setattr(
        "methods.metric_seam.family_scale.semantic_alignment_glm.urllib.request.urlopen",
        fake_urlopen,
    )

    monkeypatch.setattr(sys, "argv", [
        "semantic_alignment_glm.py",
        "--requests-dir", str(req_dir),
        "--output-dir", str(out_dir),
        "--key-file", str(key_path),
        "--concurrency", "2",
    ])
    rc = main()
    assert rc == 0
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["schema"] == "metric-seam.semantic-alignment-glm-run.v1"
    assert manifest["status"] == "raw_responses_complete"
    assert manifest["n"] == 2
    assert manifest["n_degenerate"] == 0
    assert manifest["instrument"] == "glm_subscription_api"
    raw_files = sorted(p.name for p in out_dir.glob("*_raw.txt"))
    assert raw_files == ["a_raw.txt", "b_raw.txt"]
    assert json.loads((out_dir / "a_raw.txt").read_text()) == {"stem": "a"}


def test_main_raises_when_degenerate_rate_too_high(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    req_dir = tmp_path / "requests"
    req_dir.mkdir()
    out_dir = tmp_path / "out"
    key_path = tmp_path / "key.txt"
    key_path.write_text("fake-key\n")
    for stem in ["a", "b", "c"]:
        (req_dir / f"{stem}_request.json").write_text(json.dumps(_request(stem)))

    def fake_urlopen(req, timeout=None):
        return _FakeResponse(_api_body("!" * 100))

    monkeypatch.setattr(
        "methods.metric_seam.family_scale.semantic_alignment_glm.urllib.request.urlopen",
        fake_urlopen,
    )

    monkeypatch.setattr(sys, "argv", [
        "semantic_alignment_glm.py",
        "--requests-dir", str(req_dir),
        "--output-dir", str(out_dir),
        "--key-file", str(key_path),
        "--concurrency", "1",
    ])
    with pytest.raises(RuntimeError, match="degenerate response rate too high"):
        main()
    # Raw files are still written for forensic inspection before the raise.
    assert len(list(out_dir.glob("*_raw.txt"))) == 3
    assert not (out_dir / "manifest.json").exists()


def test_main_rejects_non_frozen_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    req_dir = tmp_path / "requests"
    req_dir.mkdir()
    bad = _request("a")
    bad["status"] = "draft"
    (req_dir / "a_request.json").write_text(json.dumps(bad))
    monkeypatch.setattr(sys, "argv", [
        "semantic_alignment_glm.py",
        "--requests-dir", str(req_dir),
        "--output-dir", str(tmp_path / "out"),
    ])
    with pytest.raises(ValueError, match="not frozen pre-call"):
        main()
