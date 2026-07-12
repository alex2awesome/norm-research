#!/usr/bin/env python3
"""Unsupervised full-paper science certificate probe.

The source JSONL files contain a historical accept/reject field named ``y``.  This probe
never reads that field.  It measures only represented-evidence coverage and a conservative
code-native relation: whether a specific numeric token stated in an abstract recurs in the
paper's extracted methods/results/evaluation body.

Positive recurrence is a replayable cross-section certificate.  Zero recurrence is *not* a
failure certificate because formatting, unit conversion, extraction gaps, and paraphrase can
all hide a valid relation.  The program therefore reports zero-match cases as unresolved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_FULLPAPER = REPO_ROOT / "datasets" / "peer-review" / "peer_review_fullpaper_evidence.jsonl"
DEFAULT_CV = REPO_ROOT / "datasets" / "peer-review" / "peer_review_cv_evidence.jsonl"
DEFAULT_OUT = (
    REPO_ROOT / "outputs" / "metric_seam_pilot" / "technical_replay_v2" / "fullpaper_probe.json"
)

_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d+)"
    r"(?:\s?(?:%|percent|x|×))?",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def specific_numeric_tokens(text: str) -> set[str]:
    """Extract conservative, normalized numeric claims.

    We retain decimals, percentages, multipliers, and integers >=10.  Bare years are
    excluded, as are single-digit counts that generate pervasive accidental matches.
    This is deliberately narrower than semantic numerical equivalence.
    """

    tokens: set[str] = set()
    for match in _NUMBER_RE.finditer(text or ""):
        raw = (
            match.group(0)
            .strip()
            .lower()
            .replace(",", "")
            .replace(" ", "")
            .replace("percent", "%")
            .replace("×", "x")
        )
        number = re.match(r"\d+(?:\.\d+)?", raw)
        if not number:
            continue
        value = float(number.group(0))
        suffix = raw[number.end() :]
        if not suffix and value.is_integer() and 1900 <= value <= 2100:
            continue
        if not suffix and "." not in raw and value < 10:
            continue
        tokens.add(raw)
    return tokens


def numeric_recurrence(abstract: str, body: str) -> dict[str, Any]:
    claimed = specific_numeric_tokens(abstract)
    body_tokens = specific_numeric_tokens(body)
    matched = claimed & body_tokens
    fraction = len(matched) / len(claimed) if claimed else None
    return {
        "claimed_tokens": sorted(claimed),
        "matched_tokens": sorted(matched),
        "match_fraction": fraction,
        "certificate": "positive_recurrence" if matched else "unresolved",
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _median(values: Iterable[int]) -> float | None:
    values = list(values)
    return float(statistics.median(values)) if values else None


def run_probe(fullpaper_path: Path, cv_path: Path) -> dict[str, Any]:
    fullpaper = _read_jsonl(fullpaper_path)
    cv_rows = _read_jsonl(cv_path)
    fullpaper_ids = [str(row["paper_id"]) for row in fullpaper]
    cv_ids = [str(row["paper_id"]) for row in cv_rows]
    if len(set(fullpaper_ids)) != len(fullpaper_ids):
        raise ValueError("duplicate paper_id in full-paper evidence")
    if fullpaper_ids != cv_ids:
        raise ValueError("full-paper and claim/body evidence must have identical ordered IDs")

    aspects = sorted({aid for row in fullpaper for aid in (row.get("ev") or {})})
    evidence_surface = {}
    for aid in aspects:
        values = [str((row.get("ev") or {}).get(aid) or "") for row in fullpaper]
        abstracts = [str(row.get("abstract") or "") for row in fullpaper]
        evidence_surface[aid] = {
            "n_rows": len(values),
            "n_nonempty": sum(bool(value.strip()) for value in values),
            "n_differs_from_abstract": sum(
                value.strip() != abstract.strip() for value, abstract in zip(values, abstracts)
            ),
            "n_longer_than_abstract": sum(
                len(value) > len(abstract) for value, abstract in zip(values, abstracts)
            ),
            "median_evidence_chars": _median(len(value) for value in values),
        }

    body_nonempty = 0
    eligible = 0
    positive = 0
    majority = 0
    complete = 0
    unresolved_zero = 0
    fractions: list[float] = []
    certificates: list[dict[str, Any]] = []
    n_body_numeric = 0
    n_body_figure_table = 0
    n_body_proof = 0
    fig_table = re.compile(r"\b(?:figure|fig\.?|table|tab\.?)\s*\d", re.IGNORECASE)
    proof = re.compile(
        r"\b(?:proof|theorem|lemma|proposition|corollary|convergence bound)\b", re.IGNORECASE
    )

    for row in cv_rows:
        abstract = str(row.get("abstract") or "")
        body = str(row.get("body") or "")
        if body.strip():
            body_nonempty += 1
        if specific_numeric_tokens(body):
            n_body_numeric += 1
        n_body_figure_table += int(bool(fig_table.search(body)))
        n_body_proof += int(bool(proof.search(body)))
        relation = numeric_recurrence(abstract, body)
        if not body.strip() or not relation["claimed_tokens"]:
            continue
        eligible += 1
        fraction = float(relation["match_fraction"])
        fractions.append(fraction)
        if relation["matched_tokens"]:
            positive += 1
            if len(certificates) < 12:
                certificates.append(
                    {
                        "paper_id": str(row["paper_id"]),
                        "claimed_tokens": relation["claimed_tokens"],
                        "matched_tokens": relation["matched_tokens"],
                    }
                )
        else:
            unresolved_zero += 1
        majority += int(fraction >= 0.5)
        complete += int(math.isclose(fraction, 1.0))

    # Executable invariance checks freeze what the normalization does and does not claim.
    invariance_checks = {
        "comma_normalization": specific_numeric_tokens("1,200 samples")
        == specific_numeric_tokens("1200 samples"),
        "percent_spelling_normalization": specific_numeric_tokens("12.3 percent")
        == specific_numeric_tokens("12.3%"),
        "multiplication_symbol_normalization": specific_numeric_tokens("2.5× faster")
        == specific_numeric_tokens("2.5x faster"),
        "bare_year_excluded": not specific_numeric_tokens("published in 2024"),
        "single_digit_excluded": not specific_numeric_tokens("we use 5 folds"),
        "missing_body_abstains": numeric_recurrence("improves 12.3%", "")["certificate"]
        == "unresolved",
    }
    if not all(invariance_checks.values()):
        raise AssertionError(f"numeric recurrence invariance failure: {invariance_checks}")

    return {
        "schema_version": "technical-fullpaper-probe-v1",
        "external_supervision": "none",
        "ignored_source_fields": ["y"],
        "sources": {
            "fullpaper": {
                "path": str(fullpaper_path.relative_to(REPO_ROOT)),
                "sha256": sha256(fullpaper_path),
                "n_rows": len(fullpaper),
            },
            "claim_body": {
                "path": str(cv_path.relative_to(REPO_ROOT)),
                "sha256": sha256(cv_path),
                "n_rows": len(cv_rows),
            },
        },
        "represented_evidence": {
            "aspect_surfaces": evidence_surface,
            "n_body_nonempty": body_nonempty,
            "n_body_with_specific_numbers": n_body_numeric,
            "n_body_with_figure_or_table_reference": n_body_figure_table,
            "n_body_with_proof_marker": n_body_proof,
        },
        "numeric_claim_body_certificate": {
            "relation": "specific numeric token in abstract recurs in extracted methods/results body",
            "relation_depth": 2,
            "relation_depth_label": "cross_section_relation",
            "n_total": len(cv_rows),
            "n_eligible": eligible,
            "n_positive_recurrence_certificate": positive,
            "n_majority_tokens_recur": majority,
            "n_all_tokens_recur": complete,
            "n_zero_match_unresolved": unresolved_zero,
            "eligible_fraction": eligible / len(cv_rows) if cv_rows else None,
            "positive_fraction_of_eligible": positive / eligible if eligible else None,
            "median_match_fraction": statistics.median(fractions) if fractions else None,
            "zero_match_interpretation": (
                "unresolved, not unsupported: formatting, conversion, extraction, or paraphrase may hide support"
            ),
            "sample_positive_certificates": certificates,
        },
        "metamorphic_invariance_checks": invariance_checks,
        "missing_historical_artifacts": [
            "datasets/peer-review/peer_review_code_scores_fullpaper.npz",
            "datasets/peer-review/peer_review_code_scores_abstract_matched.npz",
            "datasets/peer-review/peer_review_cv_scores.npz",
        ],
        "interpretation": {
            "pipeline_status": "selected",
            "selection_mode": "retrospective_seed",
            "utility": (
                "full-paper evidence makes a cross-section verifier executable and yields positive, "
                "replayable recurrence certificates without consulting accept/reject outcomes"
            ),
            "constructive_extension": (
                "evidence-surface extension over the abstract-only pipeline; it certifies recurrence "
                "but does not establish that zero-match claims are unsupported or that an LLM is wrong"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fullpaper", type=Path, default=DEFAULT_FULLPAPER)
    parser.add_argument("--claim-body", type=Path, default=DEFAULT_CV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = run_probe(args.fullpaper.resolve(), args.claim_body.resolve())
    if args.check:
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
