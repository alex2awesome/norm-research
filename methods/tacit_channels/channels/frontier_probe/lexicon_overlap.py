"""Phase 0.2 — recombination vs novelty: vocabulary overlap of rescuing articulations.

For each extracted articulation (0.1 output), measure how much of its content vocabulary
already exists in the domain's own metric-bank lexicon. If successful rescues are always
recombination (low novel-token rate), articulation is bounded by known language; genuinely
novel formalization would show up as a high novel-rate among rescued cells relative to the
non-rescued contrast set.

Descriptive only — the LLM-judge pass (0.3) is the measurement of record
(feedback_llm_judges_do_all_measurement); this is the cheap text-statistics companion.

Usage:
  python -m methods.tacit_channels.channels.frontier_probe.lexicon_overlap \
      --rescues outputs/tacit_channels/frontier_probe/rescue_articulations.jsonl \
      --lexicon-glob "datasets/{domain}/online-rubrics/gpt-parsed/gpt-5-mini/*.json*" \
      --out outputs/tacit_channels/frontier_probe/lexicon_overlap.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict

from methods.tacit_channels.channels.common import read_jsonl, write_jsonl

_TOKEN = re.compile(r"[a-z][a-z\-']+")
_STOP = set("""a an the and or of to in on for with as by is are was were be been this that
these those it its if then than so not no yes does do did done have has had can could should
would may might will shall must about into over under between within without across new item
satisfy criterion guidance answer exactly following apply""".split())


def content_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall((text or "").lower()) if t not in _STOP and len(t) > 2}


def bigrams(text: str) -> set[tuple[str, str]]:
    toks = [t for t in _TOKEN.findall((text or "").lower()) if t not in _STOP]
    return set(zip(toks, toks[1:]))


def load_lexicon(pattern: str) -> set[str]:
    """Union of content tokens over every text field found in the matched lexicon files."""
    vocab: set[str] = set()

    def harvest(obj):
        if isinstance(obj, str):
            vocab.update(content_tokens(obj))
        elif isinstance(obj, dict):
            for v in obj.values():
                harvest(v)
        elif isinstance(obj, list):
            for v in obj:
                harvest(v)

    for path in glob.glob(pattern):
        try:
            if path.endswith(".jsonl"):
                for line in open(path):
                    if line.strip():
                        harvest(json.loads(line))
            else:
                harvest(json.load(open(path)))
        except (json.JSONDecodeError, OSError):
            continue
    return vocab


def analyze(rows: list[dict], lexicons: dict[str, set[str]]) -> list[dict]:
    out = []
    for row in rows:
        text = row.get("articulation_text")
        name = row.get("construct_name_text") or ""
        if not text:
            continue
        # Articulation-only content = what the arm ADDS beyond the bare construct name.
        added = content_tokens(text) - content_tokens(name)
        lex = lexicons.get(row["domain"], set())
        novel = added - lex
        out.append({
            "family": row["family"], "executor_job": row["executor_job"],
            "domain": row["domain"], "cell_id": row["cell_id"],
            "rescued": row["rescued"], "best_arm": row["best_arm"],
            "n_added_tokens": len(added),
            "novel_token_rate": (len(novel) / len(added)) if added else None,
            "novel_tokens": sorted(novel)[:40],
        })
    return out


def summarize(out_rows: list[dict]) -> dict:
    agg: dict = defaultdict(lambda: {"rescued": [], "contrast": []})
    for r in out_rows:
        if r["novel_token_rate"] is None:
            continue
        bucket = "rescued" if r["rescued"] else "contrast"
        agg[(r["family"], r["executor_job"], r["domain"])][bucket].append(r["novel_token_rate"])
    summary = {}
    for key, buckets in agg.items():
        summary["/".join(key)] = {
            b: (round(sum(v) / len(v), 4) if v else None, len(v))
            for b, v in buckets.items()}
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescues", required=True)
    ap.add_argument("--lexicon-glob", required=True,
                    help="glob with optional {domain} placeholder")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.rescues)
    domains = sorted({r["domain"] for r in rows})
    lexicons = {d: load_lexicon(args.lexicon_glob.format(domain=d)) for d in domains}
    for d in domains:
        print(f"lexicon[{d}]: {len(lexicons[d])} content tokens")

    out_rows = analyze(rows, lexicons)
    write_jsonl(args.out, out_rows)
    print(json.dumps(summarize(out_rows), indent=2))


if __name__ == "__main__":
    main()
