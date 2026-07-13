"""Tests for the one-way ctext-only blind input sealer."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

try:
    from .seal_ctext_items_v2 import seal
except ImportError:
    from seal_ctext_items_v2 import seal  # type: ignore[no-redef]


class SealCtextItemsTests(unittest.TestCase):
    def test_discards_outcome_values_and_all_non_interface_keys(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, output, manifest = root / "source.json", root / "sealed.json", root / "m.json"
            source.write_text(
                json.dumps(
                    [
                        {
                            "datapoint_id": "d2",
                            "ctext": "operative two",
                            "judgement": 0.9,
                            "text": "raw two",
                        },
                        {
                            "datapoint_id": "d1",
                            "ctext": "operative one",
                            "judgement": 0.1,
                            "text": "raw one",
                        },
                    ]
                )
            )
            payload = seal(source, output, manifest)
            rows = json.loads(output.read_text())
            self.assertEqual([row["datapoint_id"] for row in rows], ["d1", "d2"])
            self.assertTrue(all(set(row) == {"datapoint_id", "ctext"} for row in rows))
            encoded = output.read_text()
            self.assertNotIn("judgement", encoded)
            self.assertNotIn("raw one", encoded)
            self.assertFalse(payload["projection"]["outcome_values_recorded_in_manifest"])
            self.assertNotIn("0.9", manifest.read_text())

    def test_refuses_overwrite(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, output, manifest = root / "source.json", root / "sealed.json", root / "m.json"
            source.write_text(json.dumps([{"datapoint_id": "d1", "ctext": "x"}]))
            output.write_text("existing")
            with self.assertRaises(FileExistsError):
                seal(source, output, manifest)


if __name__ == "__main__":
    unittest.main()
