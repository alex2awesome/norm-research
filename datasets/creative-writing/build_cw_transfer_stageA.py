#!/usr/bin/env python3
"""cw_transfer_v1 Stage A: pooled CW preference corpus for LoRA pretraining.

Question: can pooled CW data buy a real dense model for the small verdict/curation
cells, where the standard-recipe arm is at chance (RoyalRoad T .4986 on 651
selection-free rows) or thin (Wigleaf T .6054)?

STAGE A CORPORA (neither contains RoyalRoad or Wigleaf by construction):
  litbench   LitBench-Train chosen/rejected pairs -> two binary rows per pair
             (chosen=1, rejected=0), grouped by prompt
  wp         writingprompts_modeling_clean, already binary, grouped by prompt_id

HARD LEAKAGE GUARD: every Stage-A text is hashed (normalised) and checked against
every RoyalRoad and Wigleaf text. Any collision is DROPPED and counted. Stage B
evaluates on those two cells, so a single shared story would contaminate the
transfer claim. The guard also reports near-duplicates via a first-200-char hash.

Splits are stable-hash by GROUP (prompt), never a seeded shuffle. Stage A has its
own eval split purely to verify the pretrain learned anything; it is never mixed
with any Stage-B cell evaluation, and NO POOLED AUC ACROSS SOURCES is ever emitted.

  python datasets/creative-writing/build_cw_transfer_stageA.py --pilot 24000
"""
import argparse, gzip, hashlib, json, os, re
from pathlib import Path
import pandas as pd

REPO = Path(os.environ.get("NR_REPO", "/lfs/skampere3/0/alexspan/norm-research"))
CW = REPO / "datasets/creative-writing"
OUT = CW / "cw_transfer_v1/stageA"
WS = re.compile(r"\s+")


def norm_hash(t, n=None):
    s = WS.sub(" ", str(t)).strip().lower()
    if n:
        s = s[:n]
    return hashlib.sha1(s.encode()).hexdigest()


def gsplit(g):
    b = int(hashlib.md5(("stageA::" + str(g)).encode()).hexdigest(), 16) % 1000
    return "train" if b < 900 else "eval"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=24000,
                    help="rows to keep for the pilot (0 = full corpus)")
    ap.add_argument("--sources", default="wp,litbench")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- forbidden text hashes: every RoyalRoad + Wigleaf row -----------------
    forbid_full, forbid_head = set(), set()
    for p in (CW / "royalroad_stubs/va/population.csv.gz",
              CW / "royalroad_stubs/va_expanded/population.csv.gz",
              CW / "wigleaf/va/population.csv.gz"):
        if p.exists():
            d = pd.read_csv(p)
            for t in d.text.astype(str):
                forbid_full.add(norm_hash(t))
                forbid_head.add(norm_hash(t, 200))
    print(f"[guard] {len(forbid_full)} forbidden texts (RoyalRoad + Wigleaf)")

    frames, prov = [], {}
    if "wp" in a.sources:
        wp = pd.read_csv(CW / "writingprompts_modeling_clean.csv.gz")
        wp = pd.DataFrame({"text": wp.text.astype(str),
                           "judgement": wp.judgement.astype(int),
                           "group": "wp_" + wp.prompt_id.astype(str),
                           "source": "wp"})
        prov["wp_rows_raw"] = int(len(wp))
        frames.append(wp)
    if "litbench" in a.sources:
        lb = pd.read_csv(CW / "LitBench-Train.csv.gz")
        prov["litbench_pairs_raw"] = int(len(lb))
        g = "lb_" + lb.prompt.astype(str).map(lambda s: norm_hash(s, 300)[:16])
        rows = pd.concat([
            pd.DataFrame({"text": lb.chosen_story.astype(str), "judgement": 1,
                          "group": g, "source": "litbench"}),
            pd.DataFrame({"text": lb.rejected_story.astype(str), "judgement": 0,
                          "group": g, "source": "litbench"})], ignore_index=True)
        frames.append(rows)

    df = pd.concat(frames, ignore_index=True)
    prov["rows_before_guard"] = int(len(df))

    h_full = df.text.map(norm_hash)
    h_head = df.text.map(lambda t: norm_hash(t, 200))
    bad = h_full.isin(forbid_full) | h_head.isin(forbid_head)
    prov["rows_dropped_by_leakage_guard"] = int(bad.sum())
    df = df[~bad].copy()
    df = df[df.text.str.len() >= 200]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    prov["rows_after_guard_and_dedup"] = int(len(df))
    print(f"[guard] dropped {prov['rows_dropped_by_leakage_guard']} leaking rows; "
          f"{len(df)} remain")

    df["split"] = df["group"].map(gsplit)
    if a.pilot and len(df) > a.pilot:
        # stable, group-coherent subsample: keep whole groups in hash order
        order = sorted(df.group.unique(), key=lambda g: hashlib.md5(("pilot::"+g).encode()).hexdigest())
        keep, n = set(), 0
        for g in order:
            k = int((df.group == g).sum())
            keep.add(g); n += k
            if n >= a.pilot:
                break
        df = df[df.group.isin(keep)].reset_index(drop=True)
    df["row_id"] = [norm_hash(t)[:20] for t in df.text]
    df = df.drop_duplicates(subset=["row_id"]).reset_index(drop=True)

    (OUT / "split").mkdir(parents=True, exist_ok=True)
    cols = ["text", "judgement", "group", "row_id"]
    df[cols].to_csv(OUT / "data.csv", index=False)
    for sp in ("train", "eval"):
        s = df[df.split == sp]
        s[cols].to_csv(OUT / f"split/{sp}.csv", index=False)
    # the trainer wants a test file; Stage A never reports it, so mirror eval
    df[df.split == "eval"][cols].to_csv(OUT / "split/test.csv", index=False)

    man = {"design_id": "cw_transfer_v1", "stage": "A",
           "purpose": "LoRA pretrain on pooled CW preference data; NEVER evaluated "
                      "against a Stage-B cell and never pooled across sources at readout",
           "sources": a.sources, "provenance": prov,
           "n": int(len(df)), "pos_rate": round(float(df.judgement.mean()), 4),
           "n_groups": int(df.group.nunique()),
           "by_source": df.groupby("source").size().to_dict(),
           "pos_rate_by_source": df.groupby("source").judgement.mean().round(4).to_dict(),
           "split_counts": df.split.value_counts().to_dict(),
           "split_rule": 'md5("stageA::"+group)%1000 -> <900 train / eval; group = prompt',
           "leakage_guard": "normalised full-text sha1 AND first-200-char sha1 checked "
                            "against every RoyalRoad (v1 + rr_v2_k24) and Wigleaf row; "
                            "collisions dropped",
           "no_royalroad_or_wigleaf_rows": True}
    (OUT / "manifest.json").write_text(json.dumps(man, indent=2))
    print(json.dumps({k: man[k] for k in
                      ("n", "pos_rate", "n_groups", "by_source", "pos_rate_by_source",
                       "split_counts", "provenance")}, indent=1))
    print("BUILD_CW_TRANSFER_STAGEA_DONE")


if __name__ == "__main__":
    main()
