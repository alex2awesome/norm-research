"""Manual audit of an articulation-STaR iter's rationales.

Reads `rationales.jsonl`, `judge_diagnostics.jsonl`, `rationales_kept.jsonl`
out of a single iter dir and reports:

  1. label/category distributions + accuracy of weak and strong judges
  2. diversity metrics on the KEPT rationales (vocab, type-token ratio,
     unique bullet stems, top bigrams/trigrams)
  3. lexical mode-collapse indicators (most-repeated bullet stems, most
     common stylistic phrases per side)
  4. sampled rationales for eyeballing — random and bottom-K most common.

Usage:
    python -m methods.articulation_star.audit_rationales \\
        --task creative_writing --run_name explore_contrastive_cw --iter 0
"""
from __future__ import annotations

import argparse
import json
import re
import random
from collections import Counter
from pathlib import Path

from .config import LoopConfig


# ── parsing helpers ─────────────────────────────────────────────

_BULLET_RE = re.compile(r"^\s*-\s+(.+?)\s+—\s+(.+?)\s*$")


def _parse_bullets(text: str) -> dict[str, list[tuple[str, str]]]:
    """Return {'pos': [(content, norm), ...], 'neg': [...]}."""
    out: dict[str, list[tuple[str, str]]] = {"pos": [], "neg": []}
    section: str | None = None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("POSITIVE_ASPECTS"):
            section = "pos"
            continue
        if s.startswith("NEGATIVE_ASPECTS"):
            section = "neg"
            continue
        if section is None:
            continue
        m = _BULLET_RE.match(line)
        if m:
            out[section].append((m.group(1).strip(), m.group(2).strip()))
    return out


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _normalize_stem(s: str) -> str:
    """Cheap normalization for grouping similar bullet stems."""
    s = s.lower()
    s = re.sub(r'["\']', "", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Drop everything inside parentheses — often paper-specific.
    s = re.sub(r"\([^)]*\)", "", s).strip()
    return s


# ── main audit ──────────────────────────────────────────────────

def run(cfg: LoopConfig, iter_idx: int, n_samples: int = 6):
    d = cfg.iter_dir(iter_idx)
    diag = [json.loads(l) for l in (d / "judge_diagnostics.jsonl").open()]
    kept = [json.loads(l) for l in (d / "rationales_kept.jsonl").open()]

    print(f"\n{'='*70}\nAUDIT: {d}\n{'='*70}")

    # 1. Category distribution
    cat_counts = Counter(r["category"] for r in diag if "category" in r)
    print("\n[1] Category distribution (from contrastive judge):")
    for k in ["both_right", "only_strong_right", "only_weak_right",
              "both_wrong", "undecoded"]:
        n = cat_counts.get(k, 0)
        print(f"    {k:22s} {n:5d}  ({n / max(len(diag), 1) * 100:5.1f}%)")

    y_dist_diag = Counter(r["y"] for r in diag)
    y_dist_kept = Counter(r["y"] for r in kept)
    print(f"\n    label dist (all):  y=1={y_dist_diag[1]} y=0={y_dist_diag[0]}")
    print(f"    label dist (kept): y=1={y_dist_kept[1]} y=0={y_dist_kept[0]}")
    if kept:
        print(f"    kept positive rate = {y_dist_kept[1] / len(kept):.1%}  "
              f"(vs label prior {y_dist_diag[1] / len(diag):.1%})")

    # weak/strong individual accuracy
    if any("weak_pred" in r for r in diag):
        w_dec = [r for r in diag if r.get("weak_pred") is not None]
        s_dec = [r for r in diag if r.get("strong_pred") is not None]
        w_acc = sum(1 for r in w_dec if r["weak_pred"] == r["y"]) / max(len(w_dec), 1)
        s_acc = sum(1 for r in s_dec if r["strong_pred"] == r["y"]) / max(len(s_dec), 1)
        print(f"\n    weak judge acc:   {w_acc:.1%}  (decoded {len(w_dec)}/{len(diag)})")
        print(f"    strong judge acc: {s_acc:.1%}  (decoded {len(s_dec)}/{len(diag)})")
        # baseline = always-majority
        maj = max(y_dist_diag.values()) / sum(y_dist_diag.values())
        print(f"    majority baseline acc: {maj:.1%}")

    # 2. Diversity on kept rationales
    print("\n[2] Diversity in the KEPT rationales:")
    if not kept:
        print("    (empty — nothing to analyze)")
        return

    all_pos_stems: list[str] = []
    all_neg_stems: list[str] = []
    all_pos_norms: list[str] = []
    all_neg_norms: list[str] = []
    for r in kept:
        b = _parse_bullets(r["completion"])
        all_pos_stems.extend(_normalize_stem(c) for c, _ in b["pos"])
        all_neg_stems.extend(_normalize_stem(c) for c, _ in b["neg"])
        all_pos_norms.extend(_normalize_stem(n) for _, n in b["pos"])
        all_neg_norms.extend(_normalize_stem(n) for _, n in b["neg"])

    print(f"    n kept rationales:        {len(kept)}")
    print(f"    avg pos bullets/rationale: "
          f"{len(all_pos_stems) / len(kept):.2f}")
    print(f"    avg neg bullets/rationale: "
          f"{len(all_neg_stems) / len(kept):.2f}")
    print(f"    unique pos bullet stems:  {len(set(all_pos_stems))} of "
          f"{len(all_pos_stems)} (uniqueness {len(set(all_pos_stems)) / max(len(all_pos_stems),1):.1%})")
    print(f"    unique neg bullet stems:  {len(set(all_neg_stems))} of "
          f"{len(all_neg_stems)} (uniqueness {len(set(all_neg_stems)) / max(len(all_neg_stems),1):.1%})")
    print(f"    unique pos norms:         {len(set(all_pos_norms))}")
    print(f"    unique neg norms:         {len(set(all_neg_norms))}")

    # type-token ratio on completion text
    all_tokens = []
    for r in kept:
        all_tokens.extend(_tokenize(r["completion"]))
    print(f"    type-token ratio:         {len(set(all_tokens)) / max(len(all_tokens),1):.3f} "
          f"({len(set(all_tokens))} types / {len(all_tokens)} tokens)")

    # 3. Mode-collapse indicators
    print("\n[3] Most-repeated norms (kept rationales):")
    print("    --- top positive norms ---")
    for norm, c in Counter(all_pos_norms).most_common(8):
        print(f"      {c:4d}x  {norm[:120]}")
    print("    --- top negative norms ---")
    for norm, c in Counter(all_neg_norms).most_common(8):
        print(f"      {c:4d}x  {norm[:120]}")

    # 4. Samples
    print(f"\n[4] {n_samples} random kept rationales (eyeball):")
    random.seed(7)
    for r in random.sample(kept, min(n_samples, len(kept))):
        print(f"\n--- row_id={r['row_id']} sample_idx={r['sample_idx']} "
              f"y={r['y']} ---")
        print(r["completion"][:1500])
    print()


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="peer_review")
    ap.add_argument("--run_name", default="v0")
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--n_samples", type=int, default=6)
    return ap.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    run(LoopConfig(task=a.task, run_name=a.run_name), a.iter, a.n_samples)
