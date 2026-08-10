#!/usr/bin/env python3
"""Focused no-GPU positive/negative tests for the frozen Gemma queue launcher."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from .audit_exact_directory_inventory import assert_exact_inventory
from .common import sha256_file
from .freeze_directory_inventory import build_inventory
from .run_truth_blind_gemma_baseline_queue import (
    _bank_source_sha256,
    _rehash_all_bound_inputs,
    _validate_completed_order,
)


def _ref(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


class TruthBlindGemmaQueueRoundTripTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[str]]:
        bank_hash = "a" * 64
        bank_ids = ["m1", "m2", "m3"]
        candidates_by_uid = {
            uid: {
                "norm_uid": uid,
                "candidates": [{"metric_id": metric_id} for metric_id in bank_ids],
            }
            for uid in ("n1", "n2")
        }
        candidates_path = root / "candidates.jsonl"
        candidates_path.write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n"
                for row in candidates_by_uid.values()
            ),
            encoding="utf-8",
        )
        bank_path = root / "bank.json"
        bank_path.write_text(json.dumps({"metrics": bank_ids}), encoding="utf-8")
        prompt = root / "prompt.txt"
        addon = root / "addon.txt"
        prompt.write_text("prompt\n", encoding="utf-8")
        addon.write_text("addon\n", encoding="utf-8")
        combined_prompt_hash = "b" * 64
        model = str(root / "model")
        runtime = {
            "model": model,
            "python": str(Path(sys.executable).resolve()),
            "temperature": 0.0,
            "seed": 17,
            "batch_size": 2,
            "max_model_len": 8192,
            "max_tokens": 180,
            "gpu_memory_utilization": 0.88,
            "keep_raw": True,
            "resume": True,
            "context_chars": 800,
            "description_chars": 40,
            "example_chars": 40,
            "max_examples": 0,
        }
        output = root / "original.jsonl"
        rows = [
            {
                "norm_uid": "n1",
                "task": "math-stackexchange",
                "order_mode": "original",
                "candidate_bank_source_sha256": bank_hash,
                "candidate_ids": bank_ids,
                "decision": "MATCH",
                "metric_id": "m2",
                "confidence": "high",
                "reason": "direct fit",
                "parse_error": None,
                "model": model,
                "prompt_sha256": combined_prompt_hash,
            },
            {
                "norm_uid": "n2",
                "task": "math-stackexchange",
                "order_mode": "original",
                "candidate_bank_source_sha256": bank_hash,
                "candidate_ids": bank_ids,
                "decision": "INVALID_OUTPUT",
                "metric_id": None,
                "confidence": "low",
                "reason": "bad JSON",
                "parse_error": "bad JSON",
                "model": model,
                "prompt_sha256": combined_prompt_hash,
            },
        ]
        output.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        prompt_entries = [_ref(prompt), _ref(addon)]
        meta = {
            "input_candidates": str(candidates_path),
            "input_candidates_sha256": sha256_file(candidates_path),
            "output": str(output),
            "output_sha256": sha256_file(output),
            "prompt": str(prompt),
            "prompt_addons": [str(addon)],
            "prompt_component_sha256": {
                str(prompt.resolve()): sha256_file(prompt),
                str(addon.resolve()): sha256_file(addon),
            },
            "prompt_sha256": combined_prompt_hash,
            "model": model,
            "python_executable": str(Path(sys.executable).resolve()),
            "order_mode": "original",
            "max_candidates": len(bank_ids),
            "prompt_rendering": {
                key: runtime[key]
                for key in (
                    "context_chars",
                    "description_chars",
                    "example_chars",
                    "max_examples",
                )
            },
            "runtime": {
                key: runtime[key]
                for key in (
                    "temperature",
                    "seed",
                    "batch_size",
                    "max_model_len",
                    "max_tokens",
                    "gpu_memory_utilization",
                    "keep_raw",
                    "resume",
                )
            },
            "shard_id": 0,
            "num_shards": 1,
            "invalid_count": 1,
            "elapsed_seconds": 2.5,
        }
        output.with_suffix(output.suffix + ".meta.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )
        bank_ref = _ref(bank_path)
        bank_ref["source_sha256"] = bank_hash
        queue = {
            "task": "math-stackexchange",
            "scientific_contract": {"bank_source_sha256": bank_hash},
            "inputs": {
                "bank": bank_ref,
                "candidates": _ref(candidates_path),
                "prompt_components": prompt_entries,
            },
            "prompt": {"combined_sha256": combined_prompt_hash},
            "runtime": runtime,
            "outputs": {"original": {"path": str(output)}},
        }
        return queue, candidates_by_uid, bank_ids

    @staticmethod
    def _rewrite_output_and_meta(queue: dict[str, Any], rows: list[dict[str, Any]]) -> None:
        output = Path(queue["outputs"]["original"]["path"])
        output.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        meta_path = output.with_suffix(output.suffix + ".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["output_sha256"] = sha256_file(output)
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

    def test_completed_output_round_trip_and_final_invalid_count(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            queue, candidates_by_uid, bank_ids = self._fixture(Path(raw_root))
            report = _validate_completed_order(
                queue=queue,
                order="original",
                candidates_by_uid=candidates_by_uid,
                expected_bank_ids=bank_ids,
            )
            self.assertEqual(report["final_invalid_output_count"], 1)
            self.assertEqual(report["final_parse_error_count"], 1)

    def test_missing_contract_hash_fails_as_validation_not_keyerror(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not share one source hash"):
            _bank_source_sha256({"scientific_contract": {}, "inputs": {"bank": {}}})

    def test_mutated_bound_input_fails_rehash(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            queue, _, _ = self._fixture(Path(raw_root))
            self.assertEqual(
                _rehash_all_bound_inputs(queue)["status"],
                "ALL_BOUND_INPUT_PATH_HASH_SIZE_RECHECK_PASS",
            )
            Path(queue["inputs"]["candidates"]["path"]).write_text(
                "mutated\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "hash/size drifted"):
                _rehash_all_bound_inputs(queue)

    def test_mutated_model_snapshot_fails_exact_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            model = root / "model"
            model.mkdir()
            weight = model / "weight.bin"
            weight.write_bytes(b"frozen")
            inventory_path = root / "inventory.json"
            inventory_path.write_text(
                json.dumps(build_inventory(model, workers=1)), encoding="utf-8"
            )
            self.assertEqual(
                assert_exact_inventory(inventory_path)["status"],
                "EXACT_RECURSIVE_PYCACHE_FREE_INVENTORY_PASS",
            )
            weight.write_bytes(b"drifted")
            with self.assertRaisesRegex(ValueError, "hash/size drift"):
                assert_exact_inventory(inventory_path)

    def test_wrong_candidate_order_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            queue, candidates_by_uid, bank_ids = self._fixture(Path(raw_root))
            output = Path(queue["outputs"]["original"]["path"])
            rows = list(read_json_lines(output))
            rows[0]["candidate_ids"] = list(reversed(rows[0]["candidate_ids"]))
            self._rewrite_output_and_meta(queue, rows)
            with self.assertRaisesRegex(ValueError, "lineage/order drift"):
                _validate_completed_order(
                    queue=queue,
                    order="original",
                    candidates_by_uid=candidates_by_uid,
                    expected_bank_ids=bank_ids,
                )

    def test_out_of_bank_match_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            queue, candidates_by_uid, bank_ids = self._fixture(Path(raw_root))
            output = Path(queue["outputs"]["original"]["path"])
            rows = list(read_json_lines(output))
            rows[0]["metric_id"] = "outside-bank"
            self._rewrite_output_and_meta(queue, rows)
            with self.assertRaisesRegex(ValueError, "invalid decision schema"):
                _validate_completed_order(
                    queue=queue,
                    order="original",
                    candidates_by_uid=candidates_by_uid,
                    expected_bank_ids=bank_ids,
                )

    def test_wrong_prompt_or_model_metadata_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            queue, candidates_by_uid, bank_ids = self._fixture(Path(raw_root))
            output = Path(queue["outputs"]["original"]["path"])
            meta_path = output.with_suffix(output.suffix + ".meta.json")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["model"] = "wrong-model"
            meta["prompt_sha256"] = "0" * 64
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "metadata does not bind"):
                _validate_completed_order(
                    queue=queue,
                    order="original",
                    candidates_by_uid=candidates_by_uid,
                    expected_bank_ids=bank_ids,
                )


def read_json_lines(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


if __name__ == "__main__":
    unittest.main()
