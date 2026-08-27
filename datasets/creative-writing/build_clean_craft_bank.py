#!/usr/bin/env python3
"""Build a CLEAN creative-writing craft bank from the contaminated online-rubrics pool.

The medoid bank (coverage-selected over the raw ~72k pool) is ~50% off-domain — coverage
selection PREFERS outliers (Gutenberg license text, navigation manuals, bacteriology), so
V(bank) is a noise floor. This selects only genuine creative-writing CRAFT criteria:
  1. embed every rubric (MiniLM, CPU),
  2. keep those close to craft ANCHOR concepts AND not matching a junk blocklist,
  3. dedup (cosine > 0.9),
  4. coverage-select k via k-center greedy over the CLEAN subset (spans craft dimensions).

Output: datasets/creative-writing/medoid-bank-clean/bank.json (same schema as the dirty bank).
Deterministic (fixed seed); validate-before-scaling by printing the kept set + rejected sample.
"""
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# Genuine creative-writing evaluation dimensions — the craft anchors.
CRAFT_ANCHORS = [
    "vivid and distinctive character development; believable, consistent motivation",
    "dialogue that reveals character and advances the story; natural but not slavish speech",
    "narrative structure: rising tension, turning point, satisfying resolution",
    "pacing: scenes move with purpose; no sagging middle or rushed ending",
    "prose style: precise word choice, economy, controlled rhythm and sentence variety",
    "point-of-view control and consistency; a distinctive narrative voice",
    "show don't tell: dramatize emotion and change through scene rather than summary",
    "concrete sensory imagery that grounds the setting and mood",
    "originality of premise; a fresh angle rather than cliche or generic trope",
    "emotional resonance; the story earns its feeling and lands an ending",
    "thematic depth and subtext beneath the literal surface",
    "world-building and vivid setting that serve the story",
    "tonal consistency; humor, irony, or tension sustained deliberately",
    "narrative tension and stakes that pull the reader forward",
    "figurative language: apt metaphor and image that reveal rather than decorate",
]

# Hard junk blocklist — clearly not creative-writing craft (regardless of embedding).
JUNK_RE = re.compile(
    r"gutenberg|copyright|license|licens|trademark|arable|permanent crops|latitude|longitude|"
    r"bearing|meridian|navigation|sterilis|platinum loop|petri|IRB|informed consent|"
    r"North Carolina|Confederation|mercantile|importation|bounties|tariff|GDP|census|"
    r"submission (guidelines|email)|manuscript format|word[- ]count requirement|"
    r"reviewer|Kirkus choose|how .* chooses|agency|querying|submit to|eligibilit",
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


def kmedoid_representatives(emb: np.ndarray, craft_sim: np.ndarray, k: int, seed: int = 0):
    """Cluster into k groups and take the MOST CRAFT-RELEVANT member of each cluster.

    k-center/farthest-point selection reintroduces the outlier problem (it picks the weird
    extremes — the original medoid bank's failure). k-means clusters give diversity across
    craft dimensions; picking each cluster's highest craft_sim member keeps the bank central
    and strongly on-domain rather than spanning to the fringe."""
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
        reps.append(int(idx[np.argmax(craft_sim[idx])]))   # best craft member of the cluster
    return reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(HERE / "online-rubrics" / "gpt-parsed"))
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--craft-threshold", type=float, default=0.45,
                    help="min cosine to the nearest craft anchor to keep a rubric")
    ap.add_argument("--dedup-threshold", type=float, default=0.90)
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--out", default=str(HERE / "medoid-bank-clean" / "bank.json"))
    args = ap.parse_args()

    print(f"loading rubrics from {args.pool} ...", flush=True)
    rubrics = load_rubrics(Path(args.pool))
    print(f"  {len(rubrics)} unique rubrics", flush=True)

    # hard junk filter first (cheap)
    kept = [r for r in rubrics if not JUNK_RE.search(f"{r['name']} {r['description']} {r['guidance']}")]
    print(f"  {len(kept)} after junk blocklist ({len(rubrics) - len(kept)} dropped)", flush=True)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)
    print("embedding craft anchors + rubrics (CPU) ...", flush=True)
    anchor_emb = model.encode(CRAFT_ANCHORS, normalize_embeddings=True)
    texts = [f"{r['name']}. {r['description'] or r['guidance']}" for r in kept]
    emb = model.encode(texts, normalize_embeddings=True, batch_size=256, show_progress_bar=False)

    # craft relevance = max cosine to any anchor
    craft_sim = (emb @ anchor_emb.T).max(axis=1)
    craft_idx = np.where(craft_sim >= args.craft_threshold)[0]
    print(f"  {len(craft_idx)} craft-relevant (>= {args.craft_threshold} to an anchor)", flush=True)
    craft = [kept[i] for i in craft_idx]
    craft_emb = emb[craft_idx]

    # dedup (greedy: keep first, drop anything cosine > threshold to a kept one)
    keep_mask = np.ones(len(craft), bool)
    for i in range(len(craft)):
        if not keep_mask[i]:
            continue
        sims = craft_emb[i + 1:] @ craft_emb[i]
        dup = np.where(sims > args.dedup_threshold)[0] + (i + 1)
        keep_mask[dup] = False
    ded_idx = np.where(keep_mask)[0]
    print(f"  {len(ded_idx)} after dedup (cosine > {args.dedup_threshold})", flush=True)
    ded = [craft[i] for i in ded_idx]
    ded_emb = craft_emb[ded_idx]

    # coverage-select k representative (central, high-craft) criteria across clusters
    ded_sim = craft_sim[craft_idx][ded_idx]
    sel = kmedoid_representatives(ded_emb, ded_sim, args.k)
    bank = [ded[i] for i in sel]

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"extracted": {"rubrics_metrics": bank}}, open(out, "w"), indent=2)
    print(f"\nWROTE {len(bank)} craft rubrics -> {out}", flush=True)
    print("\n=== KEPT BANK (validate) ===", flush=True)
    for r in bank:
        print(f"  + {r['name']}: {(r['description'] or r['guidance'])[:90]}", flush=True)
    print("\n=== SAMPLE REJECTED as non-craft (validate the filter) ===", flush=True)
    rej_idx = np.where(craft_sim < args.craft_threshold)[0][:12]
    for i in rej_idx:
        print(f"  - {kept[i]['name']}: {(kept[i]['description'] or '')[:80]}", flush=True)


if __name__ == "__main__":
    main()
