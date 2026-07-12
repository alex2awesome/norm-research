#!/usr/bin/env python3
"""Fresh mechanical verification of blind-v2 retrieval scope (finding 5).

The verifier plants mutually exclusive markers in a legacy-style raw ``text`` field and
the operative ``ctext`` field, builds a deterministic sealed compiler bundle, and then
executes retrieval from every emitted TRAIN alias. It closes the finding only for the
blind-v2 path; it makes no claim that historical pre-v2 retrieval artifacts were repaired.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any

try:
    from .blind_reconstruction_v2 import build_bundle
    from .split_ops_v2 import SplitScopedOps
except ImportError:  # direct-file execution
    from blind_reconstruction_v2 import build_bundle  # type: ignore[no-redef]
    from split_ops_v2 import SplitScopedOps  # type: ignore[no-redef]


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
DEFAULT_OUT = (
    ROOT
    / "outputs"
    / "metric_seam_pilot"
    / "reconstruction_v2"
    / "retrieval_finding5_verification.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify() -> dict[str, Any]:
    with TemporaryDirectory() as directory:
        tmp = Path(directory)
        items_path = tmp / "items.json"
        contract_path = tmp / "contract.json"
        items = [
            {
                "datapoint_id": f"source_id_{index}",
                "text": f"RAW_ONLY_SECRET_{index}",
                "ctext": f"CTEXT_ONLY_OPERATIVE_{index} shared retrieval vocabulary",
            }
            for index in range(8)
        ]
        items_path.write_text(json.dumps(items))
        contract_path.write_text(
            json.dumps(
                {
                    "construct_definition": "Synthetic retrieval-scope verification only.",
                    "cf_probes": [
                        {
                            "text_pos": "operative relation is present",
                            "text_neg": "operative relation is absent",
                            "why": "label-free synthetic contrast",
                            "corpus_phenomenon": "synthetic verifier fixture",
                            "probe_type": "genuine_contrast",
                            "channel": "CODE",
                        }
                    ],
                    "discrimination_checks": {"min_std": 0.01, "max_frac_at_mode": 0.9},
                }
            )
        )
        bundle, provenance = build_bundle(
            task="synthetic_retrieval_audit",
            aspect_id="a0",
            items_path=items_path,
            contract_path=contract_path,
            train_count=5,
            split_seed=7,
            capabilities={"base", "retrieval"},
        )
        encoded = json.dumps(bundle, sort_keys=True)
        corpus = {item["item_key"]: item["ctext"] for item in bundle["train_items"]}
        owner = SplitScopedOps(corpus, {"base", "retrieval"})

        all_hits: dict[str, list[str]] = {}
        self_exclusion_holds = True
        train_only_hits = True
        attacker_cannot_cancel_self_exclusion = True
        for item_key, ctext in corpus.items():
            ops = owner.for_item(item_key)
            hits = ops.retrieve_similar(ctext, k=20)
            hit_keys = [key for _, key in hits]
            all_hits[item_key] = hit_keys
            self_exclusion_holds &= item_key not in hit_keys
            train_only_hits &= set(hit_keys) <= set(corpus)
            attacker_hits = [
                key for _, key in ops.retrieve_similar(ctext, k=20, exclude_id="not_the_self")
            ]
            attacker_cannot_cancel_self_exclusion &= item_key not in attacker_hits

        heldout_rejected = False
        try:
            owner.for_item("heldout_0001")
        except KeyError:
            heldout_rejected = True

        assertions = {
            "bundle_uses_ctext_markers": "CTEXT_ONLY_OPERATIVE_" in encoded,
            "bundle_excludes_raw_text_markers": "RAW_ONLY_SECRET_" not in encoded,
            "bundle_excludes_original_identifiers": "source_id_" not in encoded,
            "bundle_count_is_train_only": len(bundle["train_items"]) == 5,
            "provenance_records_three_heldout": provenance["partition"]["heldout_count"] == 3,
            "retrieval_hits_train_aliases_only": train_only_hits,
            "retrieval_unconditionally_excludes_self": self_exclusion_holds,
            "alternate_exclude_id_cannot_cancel_self_exclusion": (
                attacker_cannot_cancel_self_exclusion
            ),
            "heldout_alias_cannot_bind_ops": heldout_rejected,
        }
        if not all(assertions.values()):
            raise AssertionError(f"blind-v2 retrieval-scope verification failed: {assertions}")

    return {
        "schema_version": "metric-seam-retrieval-finding5-verification-v1",
        "date": "2026-07-12",
        "finding": "retrieval indexed raw text and the full corpus, including sealed split",
        "scope": "blind_reconstruction_v2_only",
        "verdict": "PASS",
        "formal_closure": (
            "finding 5 is closed for the blind-v2 compiler/worker/SplitScopedOps path only"
        ),
        "historical_scope": (
            "historical pre-v2 retrieval artifacts are unchanged and are not closed by this check"
        ),
        "fixture": {"corpus_count": 8, "train_count": 5, "heldout_count": 3},
        "assertions": assertions,
        "retrieval_hits_by_train_alias": all_hits,
        "implementation_sha256": {
            "blind_reconstruction_v2.py": sha256(HERE / "blind_reconstruction_v2.py"),
            "_blind_worker_v2.py": sha256(HERE / "_blind_worker_v2.py"),
            "split_ops_v2.py": sha256(HERE / "split_ops_v2.py"),
            "verification_script": sha256(Path(__file__)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = verify()
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.check:
        print(rendered, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
