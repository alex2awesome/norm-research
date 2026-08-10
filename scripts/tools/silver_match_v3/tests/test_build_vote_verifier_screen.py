from pathlib import Path

from scripts.tools.silver_match_v3.build_union_verifier_screen import _index
from scripts.tools.silver_match_v3.build_vote_verifier_screen import build
from scripts.tools.silver_match_v3.common import write_jsonl


def test_vote_screen_counts_individual_order_views(tmp_path: Path) -> None:
    primary = tmp_path / "primary.jsonl"
    write_jsonl(
        primary,
        [
            {
                "norm_uid": uid,
                "task": "press-releases",
                "decision": "MATCH",
                "metric_id": "a1",
                "candidate_bank_source_sha256": "bank",
            }
            for uid in ("u1", "u2")
        ],
    )
    variants = []
    for name, confirmations in (("a", {"u1"}), ("b", {"u2"})):
        paths = []
        for order in ("original", "hashed"):
            path = tmp_path / f"{name}.{order}.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "norm_uid": uid,
                        "task": "press-releases",
                        "order_mode": order,
                        "primary_metric_id": "a1",
                        "candidate_bank_source_sha256": "bank",
                        "decision": (
                            "CONFIRM_MATCH" if uid in confirmations else "REJECT_MATCH"
                        ),
                        "metric_id": "a1" if uid in confirmations else None,
                        "confidence": "high",
                        "parse_error": None,
                    }
                    for uid in ("u1", "u2")
                ],
            )
            paths.append(path)
        variants.append((name, *paths))
    result = build(
        task="press-releases",
        primary_path=primary,
        variants=variants,
        minimum_confirmations=2,
        output_root=tmp_path / "screen",
    )
    assert result["selected_count"] == 2
    assert set(_index(tmp_path / "screen/screened_primary.jsonl")) == {"u1", "u2"}
