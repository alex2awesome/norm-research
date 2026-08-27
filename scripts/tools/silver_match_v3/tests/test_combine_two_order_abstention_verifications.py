import json

import pytest

from scripts.tools.silver_match_v3.combine_two_order_abstention_verifications import (
    combine,
)
from scripts.tools.silver_match_v3.common import read_jsonl


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _fixture(tmp_path):
    audits = _write(
        tmp_path / "audits.jsonl",
        [
            {
                "norm_uid": uid,
                "task": "t",
                "corpus": "c",
                "row": index,
                "bank_source_sha256": "sha",
                "provisional_decision": "NO_CANDIDATE_FITS",
                "rescue_exhaustive": True,
                "rescue_bank_count": 88,
                "rescue_coverage_repeats": 2,
                "rescue_reincludes_primary": True,
            }
            for index, uid in enumerate(("keep", "disagree", "possible"))
        ],
    )

    def row(uid, order, decision="NO_CANDIDATE_FITS", possible=False):
        return {
            "norm_uid": uid,
            "task": "t",
            "corpus": "c",
            "bank_source_sha256": "sha",
            "provisional_decision": "NO_CANDIDATE_FITS",
            "decision": "POSSIBLE_EXACT_BANK_MATCH" if possible else decision,
            "confirmed_decision": None if possible else decision,
            "possible_exact_bank_match": possible,
            "confidence": "high",
            "prompt_sha256": "p" * 64,
            "model": "/model",
            "order_mode": order,
            "parse_error": None,
        }

    original = _write(
        tmp_path / "original.jsonl",
        [
            row("keep", "original"),
            row("disagree", "original"),
            row("possible", "original"),
        ],
    )
    hashed = _write(
        tmp_path / "hashed.jsonl",
        [
            row("keep", "hashed"),
            row("disagree", "hashed", decision="GENERIC_VERDICT"),
            row("possible", "hashed", possible=True),
        ],
    )
    return audits, original, hashed


def test_two_order_abstention_consensus_fails_closed(tmp_path):
    audits, original, hashed = _fixture(tmp_path)
    output = tmp_path / "combined.jsonl"
    report = combine(
        audits_path=audits,
        original_path=original,
        hashed_path=hashed,
        output_path=output,
    )
    rows = {row["norm_uid"]: row for row in read_jsonl(output)}
    assert report["complete"] is True
    assert rows["keep"]["confirmed_decision"] == "NO_CANDIDATE_FITS"
    assert rows["keep"]["strict_two_order_abstention"] is True
    assert rows["disagree"]["decision"] == "UNRESOLVED_ABSTENTION"
    assert rows["possible"]["possible_exact_bank_match"] is True


def test_two_order_abstention_requires_exact_coverage(tmp_path):
    audits, original, hashed = _fixture(tmp_path)
    hashed.write_text("\n".join(hashed.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="coverage mismatch"):
        combine(
            audits_path=audits,
            original_path=original,
            hashed_path=hashed,
            output_path=tmp_path / "combined.jsonl",
        )
