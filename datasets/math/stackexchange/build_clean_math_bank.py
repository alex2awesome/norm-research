#!/usr/bin/env python3
"""Build a CLEAN math answer-quality bank from the contaminated online-rubrics pool.

Same recipe as datasets/creative-writing/build_clean_craft_bank.py (whose dirty-bank lesson
motivated this): the raw ~7k-file pool mixes genuine rigor/exposition criteria with journal
submission admin, course logistics, wiki-editing guides, and plainly off-domain scraped text
(dental decay, venereal-activity rubrics, musical intervals). Pipeline:
  1. embed every rubric (MiniLM, CPU),
  2. keep those close to a math answer-QUALITY anchor AND not matching the junk blocklist
     (quality-of-exposition criteria, not content-specific theorem statements),
  3. dedup (cosine > 0.9),
  4. k-means clusters, take each cluster's highest-anchor-sim member (kmedoid; NOT k-center,
     which reintroduces outliers).

Output: datasets/math/stackexchange/medoid-bank-clean/bank.json (arm-pipeline schema).
Deterministic; validate-before-scaling by printing kept + rejected samples.
"""
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# Math ANSWER-QUALITY dimensions (evaluation criteria, not content knowledge).
QUALITY_ANCHORS = [
    "logical rigor: every step of the argument justified, no gaps or unjustified leaps",
    "correctness of computations and algebraic manipulations; freedom from errors",
    "clarity of mathematical exposition; well-organized progression from premises to conclusion",
    "appropriate level of detail: key steps shown, routine steps not belabored",
    "precise and consistent mathematical notation; terms and symbols defined before use",
    "directly answers the question asked; states assumptions and scope explicitly",
    "provides intuition, motivation, or a geometric picture behind the formal argument",
    "elegance and economy of proof; the simplest method that fully works",
    "correct invocation of theorems with their hypotheses verified before use",
    "pedagogical quality: anticipates reader confusion, explains why a step works not just how",
    "completeness: all cases handled, boundary and edge cases addressed",
    "helpful use of examples or counterexamples to illustrate the general claim",
    "clear structure and formatting of the mathematics: displayed equations, proof organization",
    "verifies or sanity-checks the result (special case, alternative derivation, numeric check)",
    "situates the result in a broader context or more general framework",
    "insight: the proof turns on a clear key idea that illuminates why the result is true",
    "self-contained: readable without extensive prerequisites, accessible at the asker's level",
]

# Hard junk: publication/course admin and clearly off-domain scraped text. Kept NARROW —
# the anchor-cosine filter does the main work (some pedagogy rubrics mention 'classroom'
# but are genuine exposition-quality criteria).
JUNK_RE = re.compile(
    r"offprint|manuscript|camera.ready|page charge|subject classification|journal|"
    r"submission|editorial board|copyright|licens|gutenberg|"
    r"syllabus|attendance|office hours|enroll|semester|ten.week|grading scale|grade appeal|"
    r"red link|wiki|article.{0,12}edit|format mathematics here|how can i format|objective \d+:|"
    r"venereal|dental|teeth|food|culinary|diapason|diapente|blockchain|consensus protocol",
    re.I)


def load_rubrics(pool_dir: Path):
    rubrics = []
    seen = set()
    for fp in sorted(glob.glob(str(pool_dir / "**" / "*.json"), recursive=True)):
        try:
            doc = json.load(open(fp))
        except Exception:
            continue
        for rm in (doc.get("extracted", {}) or {}).get("rubrics_metrics", []) or []:
            name = (rm.get("name") or "").strip()
            desc = (rm.get("description") or "").strip()
            guid = (rm.get("guidance") or "").strip()
            if not name:
                continue
            key = (name.lower(), desc.lower()[:80])
            if key in seen:
                continue
            seen.add(key)
            rubrics.append({"name": name, "description": desc, "guidance": guid})
    return rubrics


def kmedoid_representatives(emb: np.ndarray, anchor_sim: np.ndarray, k: int, seed: int = 0):
    """k-means clusters for diversity; keep each cluster's most anchor-relevant member."""
    n = len(emb)
    if n <= k:
        return list(range(n))
    from sklearn.cluster import KMeans
    lab = KMeans(n_clusters=k, random_state=seed, n_init=4).fit_predict(emb)
    reps = []
    for c in range(k):
        idx = np.where(lab == c)[0]
        if len(idx) == 0:
            continue
        reps.append(int(idx[np.argmax(anchor_sim[idx])]))
    return reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(HERE / "online-rubrics" / "gpt-parsed"))
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--quality-threshold", type=float, default=0.45,
                    help="min cosine to the nearest quality anchor to keep a rubric")
    ap.add_argument("--dedup-threshold", type=float, default=0.90)
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--out", default=str(HERE / "medoid-bank-clean" / "bank.json"))
    args = ap.parse_args()

    print(f"loading rubrics from {args.pool} ...", flush=True)
    rubrics = load_rubrics(Path(args.pool))
    print(f"  {len(rubrics)} unique rubrics", flush=True)

    kept = [r for r in rubrics if not JUNK_RE.search(f"{r['name']} {r['description']} {r['guidance']}")]
    print(f"  {len(kept)} after junk blocklist ({len(rubrics) - len(kept)} dropped)", flush=True)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)
    print("embedding anchors + rubrics ...", flush=True)
    anchor_emb = model.encode(QUALITY_ANCHORS, normalize_embeddings=True)
    texts = [f"{r['name']}. {r['description'] or r['guidance']}" for r in kept]
    emb = model.encode(texts, normalize_embeddings=True, batch_size=256, show_progress_bar=False)

    qual_sim = (emb @ anchor_emb.T).max(axis=1)
    qual_idx = np.where(qual_sim >= args.quality_threshold)[0]
    print(f"  {len(qual_idx)} quality-relevant (>= {args.quality_threshold} to an anchor)", flush=True)
    qual = [kept[i] for i in qual_idx]
    qual_emb = emb[qual_idx]

    keep_mask = np.ones(len(qual), bool)
    for i in range(len(qual)):
        if not keep_mask[i]:
            continue
        sims = qual_emb[i + 1:] @ qual_emb[i]
        dup = np.where(sims > args.dedup_threshold)[0] + (i + 1)
        keep_mask[dup] = False
    ded_idx = np.where(keep_mask)[0]
    print(f"  {len(ded_idx)} after dedup (cosine > {args.dedup_threshold})", flush=True)
    ded = [qual[i] for i in ded_idx]
    ded_emb = qual_emb[ded_idx]

    ded_sim = qual_sim[qual_idx][ded_idx]
    sel = kmedoid_representatives(ded_emb, ded_sim, args.k)
    bank = [ded[i] for i in sel]

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"extracted": {"rubrics_metrics": bank}}, open(out, "w"), indent=2)
    print(f"\nWROTE {len(bank)} quality rubrics -> {out}", flush=True)
    print("\n=== KEPT BANK (validate) ===", flush=True)
    for r in bank:
        print(f"  + {r['name']}: {(r['description'] or r['guidance'])[:90]}", flush=True)
    print("\n=== SAMPLE REJECTED as off-quality (validate the filter) ===", flush=True)
    rej_idx = np.where(qual_sim < args.quality_threshold)[0][:12]
    for i in rej_idx:
        print(f"  - {kept[i]['name']}: {(kept[i]['description'] or '')[:80]}", flush=True)


if __name__ == "__main__":
    main()
