"""Tests for the opaque sanitized heldout sealer and steward audits."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

try:
    from . import seal_sanitized_ctext_heldout_v1 as sealer
    from .audit_sanitized_ctext_heldout_v1 import (
        build_privacy_receipt,
        build_replay_receipt,
    )
except ImportError:  # pragma: no cover - direct-test compatibility
    import seal_sanitized_ctext_heldout_v1 as sealer  # type: ignore[no-redef]
    from audit_sanitized_ctext_heldout_v1 import (  # type: ignore[no-redef]
        build_privacy_receipt,
        build_replay_receipt,
    )


class _GuardedRow(dict):
    def __getitem__(self, key):
        if key not in {"ctext", "datapoint_id"}:
            raise AssertionError("a forbidden source value was indexed")
        return super().__getitem__(key)


def _rows() -> list[dict]:
    return [
        {
            "datapoint_id": f"private-id-{index}",
            "ctext": (
                "diff --git a/a.py b/a.py\n+api_key='ABCDEFGHIJKLMNOPQRST'\n"
                if index == 3
                else f"diff --git a/a.py b/a.py\n+value_{index} = {index}\n"
            ),
            "judgement": index / 10,
            "historical_note": f"forbidden-{index}",
        }
        for index in range(1, 6)
    ]


class HeldoutSealerTests(unittest.TestCase):
    def test_indexes_only_allowlisted_values_and_projects_every_row(self) -> None:
        rows = [_GuardedRow(row) for row in _rows()]
        original = sealer.project_ctext
        call_count = 0

        def counted(text: str) -> str:
            nonlocal call_count
            call_count += 1
            return original(text)

        with patch.object(sealer, "project_ctext", side_effect=counted):
            heldout, projection, counts = sealer._trusted_projection(
                rows, train_count=3, heldout_count=2, split_seed=7
            )
        self.assertEqual(call_count, 5)
        self.assertEqual(
            [row["item_key"] for row in heldout],
            ["heldout_0001", "heldout_0002"],
        )
        self.assertTrue(all(set(row) == {"ctext", "item_key"} for row in heldout))
        self.assertFalse(projection["source_identifiers_emitted"])
        self.assertFalse(projection["source_identifier_map_emitted"])
        self.assertEqual(counts["full"]["changed_row_count"], 1)

    def test_seal_is_readonly_opaque_and_replays_exactly(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            bundle = root / "heldout.json"
            manifest = root / "manifest.json"
            source.write_text(json.dumps(_rows()), encoding="utf-8")
            sealer.seal_heldout_view(
                source_path=source,
                bundle_path=bundle,
                manifest_path=manifest,
                task="code_review",
                criterion_id="a407",
                train_count=3,
                heldout_count=2,
                split_seed=7,
            )
            payload = json.loads(bundle.read_text(encoding="utf-8"))
            encoded = bundle.read_text(encoding="utf-8")
            self.assertTrue(all(set(row) == {"ctext", "item_key"} for row in payload["heldout_items"]))
            self.assertNotIn("private-id-", encoded)
            self.assertNotIn("forbidden-", encoded)
            self.assertNotIn("ABCDEFGHIJKLMNOPQRST", encoded)
            self.assertEqual(bundle.stat().st_mode & 0o222, 0)
            self.assertEqual(manifest.stat().st_mode & 0o222, 0)

            privacy = build_privacy_receipt(
                source_path=source, bundle_path=bundle, manifest_path=manifest
            )
            replay = build_replay_receipt(
                source_path=source, bundle_path=bundle, manifest_path=manifest
            )
            self.assertTrue(privacy["audit_passed"])
            self.assertTrue(replay["replay_passed"])

    def test_refuses_overwrite(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            bundle = root / "heldout.json"
            manifest = root / "manifest.json"
            source.write_text(json.dumps(_rows()), encoding="utf-8")
            bundle.write_text("existing", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                sealer.seal_heldout_view(
                    source_path=source,
                    bundle_path=bundle,
                    manifest_path=manifest,
                    task="code_review",
                    criterion_id="a407",
                    train_count=3,
                    heldout_count=2,
                    split_seed=7,
                )


if __name__ == "__main__":
    unittest.main()
