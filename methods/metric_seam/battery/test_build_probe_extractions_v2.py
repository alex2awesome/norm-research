"""Regression tests for the real L-channel probe-extraction producer."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

try:
    from .build_probe_extractions_v2 import finalize, prepare
except ImportError:
    from build_probe_extractions_v2 import finalize, prepare  # type: ignore[no-redef]


class ProbeExtractionProducerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.contract = self.root / "contract.json"
        self.candidate = self.root / "candidate.py"
        self.out = self.root / "out"
        self.contract.write_text(
            json.dumps(
                {
                    "construct_definition": "Distinguish the operative relation.",
                    "cf_probes": [
                        {
                            "text_pos": "The claim is proved by executing the witness.",
                            "text_neg": "The claim is merely mentioned.",
                            "why": "execution versus mention",
                            "corpus_phenomenon": "synthetic fixture",
                            "probe_type": "genuine_contrast",
                            "channel": "L",
                        },
                        {
                            "text_pos": "x = 2",
                            "text_neg": "x",
                            "why": "assignment versus token",
                            "corpus_phenomenon": "synthetic fixture",
                            "probe_type": "genuine_contrast",
                            "channel": "CODE",
                        },
                    ],
                    "discrimination_checks": {"min_std": 0.01, "max_frac_at_mode": 0.9},
                }
            )
        )
        self.candidate.write_text(
            "LLM_FIELDS = {'relation': 'Reply EXECUTED or MENTIONED.'}\n"
            "def score(text, extracted, ops): return 0.5\n"
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_prepare_is_pair_blind_and_finalize_binds_real_responses(self) -> None:
        summary = prepare(
            self.contract,
            self.candidate,
            backend="fixture",
            model="fixture-model",
            out_dir=self.out,
        )
        self.assertEqual(summary["n_prompts"], 2)
        prompts = [json.loads(line) for line in (self.out / "prompts.jsonl").read_text().splitlines()]
        self.assertNotIn("positive", prompts[0]["prompt"].lower())
        self.assertNotIn("negative", prompts[1]["prompt"].lower())
        self.assertNotIn(prompts[1]["prompt"], prompts[0]["prompt"])
        responses = [
            {
                "channel": row["channel"],
                "aspect_id": row["aspect_id"],
                "datapoint_id": row["datapoint_id"],
                "raw": (
                    "```json\n{\"relation\": \"EXECUTED\"}\n```"
                    if row["datapoint_id"] == "pos"
                    else '{"relation": "MENTIONED"}'
                ),
            }
            for row in prompts
        ]
        (self.out / "results.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in responses)
        )
        frozen = finalize(
            self.contract,
            self.candidate,
            backend="fixture",
            model="fixture-model",
            out_dir=self.out,
        )
        self.assertEqual(frozen["n_available"], 1)
        payload = json.loads((self.out / "probe_extractions.json").read_text())
        self.assertEqual(payload["probes"][0]["pos"]["relation"], "EXECUTED")
        self.assertEqual(payload["probes"][0]["neg"]["relation"], "MENTIONED")

    def test_manifest_change_fails_closed(self) -> None:
        prepare(
            self.contract,
            self.candidate,
            backend="fixture",
            model="fixture-model",
            out_dir=self.out,
        )
        manifest = json.loads((self.out / "extractor_manifest.json").read_text())
        manifest["model"] = "tampered"
        (self.out / "extractor_manifest.json").write_text(json.dumps(manifest))
        (self.out / "results.jsonl").write_text("")
        with self.assertRaises(ValueError):
            finalize(
                self.contract,
                self.candidate,
                backend="fixture",
                model="fixture-model",
                out_dir=self.out,
            )


if __name__ == "__main__":
    unittest.main()
