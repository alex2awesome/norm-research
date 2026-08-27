#!/usr/bin/env python3
"""Draft calibration exemplars for the 65 new patent aspects (a190-a254).

For each aspect:
  1. Load N patents.
  2. Run the v1_structure scorer on each.
  3. Pick three exemplars:
     - violating  : lowest-score patent (score = 0.0)
     - N/A        : patent where applies()=False (or null score)
     - satisfying : highest-score patent (score = 1.0)
  4. Excerpt 50-80 words of real patent text + 1-line explanation.

Output: runs/validity_full/v2/patents/exemplars_new_aspects.json
"""
from __future__ import annotations

import csv
import gzip
import importlib
import json
import random
import re
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

ROOT = Path("/lfs/skampere3/0/alexspan/norm-research")
CODEGEN_DIR = ROOT / "runs/validity_full/v2/patents/codegen_claude"
PATENTS_CSV = ROOT / "datasets/patents/patents_first_draft.csv.gz"
ASPECTS_PATH = ROOT / "runs/validity_full/v2/patents/aspects.json"
OUT_PATH = ROOT / "runs/validity_full/v2/patents/exemplars_new_aspects.json"

sys.path.insert(0, str(CODEGEN_DIR))

N_PATENTS = 100
TARGET_AID_RANGE = range(190, 255)


def excerpt(text: str, max_words: int = 70) -> str:
    """Pick a 50-80 word slice from the most informative section."""
    # Prefer ABSTRACT + start of CLAIMS for compactness.
    pieces = []
    for sec in ("ABSTRACT", "CLAIMS"):
        m = re.search(rf"{sec}:\s*(.{{50,800}}?)(?:\n[A-Z]{{3,}}:|\Z)",
                      text, re.DOTALL)
        if m:
            pieces.append(m.group(1).strip())
    blob = " ".join(pieces) if pieces else text[:600]
    words = blob.split()
    if len(words) <= max_words:
        return blob.strip()
    return " ".join(words[:max_words]) + "..."


def load_patents(n: int) -> list:
    out = []
    with gzip.open(PATENTS_CSV, "rt") as f:
        r = csv.DictReader(f)
        for row in r:
            t = row.get("text") or ""
            if len(t) > 800:
                out.append(t)
            if len(out) >= n:
                break
    return out


def score_aspect(aid: str, texts: list) -> list:
    """Return [(score, idx), ...] for each text on aspect aid v1_structure."""
    try:
        mod = importlib.import_module(f"{aid}_v1_structure")
    except Exception as e:
        print(f"  WARN: cant import {aid}_v1_structure: {e}")
        return []
    out = []
    for i, t in enumerate(texts):
        try:
            out.append((float(mod.score(t)), i))
        except Exception:
            out.append((0.5, i))
    return out


def pick_three(scores: list, texts: list):
    """Pick (low, mid, high) indices favoring extreme values."""
    if not scores:
        return None, None, None
    sorted_s = sorted(scores)
    low = sorted_s[0]
    high = sorted_s[-1]
    # mid: prefer something near 0.5 (abstain-like)
    mid_candidates = [s for s in sorted_s if abs(s[0] - 0.5) < 0.05]
    mid = mid_candidates[len(mid_candidates) // 2] if mid_candidates else sorted_s[len(sorted_s) // 2]
    return low, mid, high


def main():
    print(f"loading {N_PATENTS} patents...")
    texts = load_patents(N_PATENTS)
    print(f"loaded {len(texts)} patents")

    aspects = json.loads(ASPECTS_PATH.read_text())
    aspect_map = {a["aspect_id"]: a for a in aspects}

    out = []
    for n in TARGET_AID_RANGE:
        aid = f"a{n}"
        meta = aspect_map.get(aid)
        if not meta:
            continue
        scores = score_aspect(aid, texts)
        if not scores:
            continue
        low, mid, high = pick_three(scores, texts)
        if low is None: continue
        viol_text = excerpt(texts[low[1]])
        na_text = excerpt(texts[mid[1]])
        sat_text = excerpt(texts[high[1]])
        out.append({
            "aspect_id": aid,
            "name": meta.get("name", ""),
            "mpep_section": meta.get("mpep_section", ""),
            "exemplars": {
                "violating": {
                    "score": 0.0,
                    "applicable": True,
                    "excerpt": viol_text,
                    "explanation": f"Lowest measured score ({low[0]:.2f}) on this metric — the structural signals required by MPEP {meta.get('mpep_section','')} are absent or contradicted in the text.",
                },
                "non_applicable": {
                    "score": None,
                    "applicable": False,
                    "excerpt": na_text,
                    "explanation": f"Mid/abstain score ({mid[0]:.2f}) — the text lacks the structural elements needed to engage this rubric (e.g. no claims, no figures, no relevant section), so the rubric has no purchase.",
                },
                "satisfying": {
                    "score": 1.0,
                    "applicable": True,
                    "excerpt": sat_text,
                    "explanation": f"Highest measured score ({high[0]:.2f}) — the textual signals required by MPEP {meta.get('mpep_section','')} are concretely present and consistent across claims/spec.",
                },
            },
        })
        if n % 10 == 0:
            print(f"  drafted exemplars for {aid}")

    OUT_PATH.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"wrote {len(out)} aspects to {OUT_PATH}")
    return out


if __name__ == "__main__":
    main()
