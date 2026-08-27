"""
validate_claims.py — post-hoc validator + summary stats for the extraction
output.

Reads outputs/.../claims.jsonl (rows = answers) and writes:
  - claims_flat.jsonl : one row per CLAIM, ready for check_claims.py
  - validate_summary.json : yield, retry breakdown, claim-type histogram

Also performs the binding-cue filter that the recipe demands:
"filter REFUTED claims whose quotes open with binding cues (if/let/define/
suppose) — definitions and context-dependent statements masquerade as wrong
claims." For code, we treat 'if you do X then Y' antecedents as non-checkable.
The filter is added as a feature (binding_cue_skip=True) so check_claims.py
can downgrade REFUTED verdicts.

Usage:
  python3 scripts/crse_claims/validate_claims.py \
      --claims outputs/crse_claims_pilot/claims.jsonl \
      --out-flat outputs/crse_claims_pilot/claims_flat.jsonl \
      --out-summary outputs/crse_claims_pilot/validate_summary.json
"""
from __future__ import annotations
import argparse, json, re
from collections import Counter
from pathlib import Path


BINDING_CUE_RE = re.compile(
    r"^\s*(?:if\b|let\b|define\b|suppose\b|assume\b|when\b|given\b|"
    r"in\s+case\b|provided\b)",
    re.IGNORECASE,
)
# Code-specific antecedent cues (e.g. "if you do X then Y") — we keep the
# claim but mark it for downstream conditional handling.
CONDITIONAL_CUE_RE = re.compile(
    r"^\s*(?:if\s+you\b|if\s+the\b|if\s+your\b|when\s+you\b|when\s+the\b)",
    re.IGNORECASE,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True)
    ap.add_argument("--out-flat", required=True)
    ap.add_argument("--out-summary", required=True)
    args = ap.parse_args()

    in_path = Path(args.claims)
    out_flat = Path(args.out_flat); out_flat.parent.mkdir(parents=True, exist_ok=True)
    out_summary = Path(args.out_summary)

    # Counters
    n_rows = 0
    n_validated = 0
    pass_used = Counter()
    final_errors = Counter()
    n_claims_total = 0
    type_dist = Counter()
    lang_dist = Counter()
    binding_skipped = 0
    conditional_marked = 0
    quote_exact_count = 0
    quote_loose_only = 0

    with in_path.open() as fin, out_flat.open("w") as fout:
        for line in fin:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            n_rows += 1
            ext = rec.get("extracted")
            pass_used[rec.get("final_pass") or "NONE"] += 1
            if not ext:
                final_errors[rec.get("final_error") or "unknown"] += 1
                continue
            n_validated += 1
            claims = ext.get("claims") or []
            for ci, c in enumerate(claims):
                n_claims_total += 1
                ctype = c.get("claim_type", "unknown")
                lang = c.get("language", "unknown")
                type_dist[ctype] += 1
                lang_dist[lang] += 1
                if c.get("quote_exact"):
                    quote_exact_count += 1
                elif c.get("quote_loose_ok"):
                    quote_loose_only += 1
                quote = c.get("source_quote", "") or ""
                # Binding-cue detection
                binding_cue = bool(BINDING_CUE_RE.match(quote))
                conditional = bool(CONDITIONAL_CUE_RE.match(quote))
                if binding_cue:
                    binding_skipped += 1
                if conditional:
                    conditional_marked += 1
                flat = {
                    "row_idx":      n_rows - 1,
                    "claim_idx":    ci,
                    "answer_id":    rec.get("answer_id"),
                    "question_id":  rec.get("question_id"),
                    "judgement":    rec.get("judgement"),
                    "split":        rec.get("split"),
                    "primary_tag":  rec.get("primary_tag"),
                    "answer_text":  rec.get("answer_text"),
                    "claim_type":   ctype,
                    "claim_text":   c.get("claim_text"),
                    "source_quote": quote,
                    "target":       c.get("target"),
                    "language":     lang,
                    "inline_code":  c.get("inline_code"),
                    "quote_exact":  bool(c.get("quote_exact")),
                    "quote_loose_ok": bool(c.get("quote_loose_ok")),
                    "binding_cue":  binding_cue,
                    "conditional":  conditional,
                }
                fout.write(json.dumps(flat) + "\n")

    summary = {
        "input_rows":               n_rows,
        "validated_rows":           n_validated,
        "validated_rate":           round(n_validated / max(1, n_rows), 4),
        "pass_used":                dict(pass_used),
        "final_errors":             dict(final_errors),
        "total_claims":             n_claims_total,
        "avg_claims_per_validated": round(n_claims_total / max(1, n_validated), 2),
        "claim_type_dist":          dict(type_dist),
        "language_dist":            dict(lang_dist),
        "quote_exact":              quote_exact_count,
        "quote_loose_only":         quote_loose_only,
        "binding_cue_count":        binding_skipped,
        "conditional_count":        conditional_marked,
    }
    out_summary.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
