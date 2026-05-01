"""
Build cluster partitions over funded ERC projects for ERCEA score-aggregate
requests. Four partitions share the same project universe so within-cluster
score variance can be decomposed across clustering axes:

  - random              null baseline (shuffle within stratum)
  - semantic_tfidf      TF-IDF over `objective`, agglomerative
  - rhetorical          hedging/claim/length features over `objective`
  - budget_duration     2-D: ec_max_contribution + duration days

Stratification key: (panel, call_year). Synergy (ERC-SyG), PoC, LVG, and
any project without a panel assignment are excluded — they don't fit the
single-panel review model this experiment is about.

Target cluster size: 5. Within each stratum we form ceil(n/5) clusters
of size 4–6. Strata with n < 5 are kept as a single cluster and flagged.

Output long-format CSV, one row per (project × partition):

  partition, cluster_id, project_id, panel, funding_scheme, call_year

cluster_id format:  {partition}__{panel}__{year}__{kkk}
e.g.  semantic_tfidf__PE5__2018__003
"""

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

EXCLUDE_SCHEMES = {"ERC-SyG", "ERC-POC", "ERC-POC-LS", "ERC-LVG"}
TARGET_CLUSTER_SIZE = 5
RANDOM_SEED = 20260415

CALL_YEAR_RE = re.compile(r"(\d{4})")

HEDGE_WORDS = {
    "may", "might", "could", "possibly", "potentially", "suggest", "suggests",
    "indicate", "indicates", "likely", "perhaps", "appears", "seems",
    "hypothesize", "hypothesise", "propose", "proposes",
}
CLAIM_WORDS = {
    "prove", "proves", "demonstrate", "demonstrates", "novel", "first",
    "unprecedented", "breakthrough", "revolutionary", "transform",
    "transforms", "paradigm", "fundamental", "groundbreaking", "establish",
    "establishes",
}
METHOD_WORDS = {
    "method", "methods", "methodology", "approach", "technique", "framework",
    "pipeline", "algorithm", "model", "experiment", "experiments",
    "measurement", "measurements", "analysis", "data",
}
VISION_WORDS = {
    "vision", "goal", "aim", "aims", "ambition", "ambitious", "challenge",
    "understand", "understanding", "discover", "explore", "reveal", "grand",
}


def extract_call_year(row: dict) -> str:
    for col in ("master_call", "sub_call", "topics"):
        v = row.get(col) or ""
        m = CALL_YEAR_RE.search(v)
        if m:
            y = int(m.group(1))
            if 2006 <= y <= 2025:
                return str(y)
    v = row.get("start_date") or ""
    m = CALL_YEAR_RE.match(v)
    if m:
        return m.group(1)
    return "UNK"


def rhetorical_features(text: str) -> np.ndarray:
    if not text:
        return np.zeros(7, dtype=float)
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    n = max(len(tokens), 1)
    hedge = sum(t in HEDGE_WORDS for t in tokens) / n
    claim = sum(t in CLAIM_WORDS for t in tokens) / n
    method = sum(t in METHOD_WORDS for t in tokens) / n
    vision = sum(t in VISION_WORDS for t in tokens) / n
    n_sent = max(text.count(".") + text.count("!") + text.count("?"), 1)
    avg_sent_len = n / n_sent
    qs = text.count("?")
    return np.array(
        [hedge, claim, method, vision, math.log1p(n), avg_sent_len, qs],
        dtype=float,
    )


def parse_duration_days(row: dict) -> float:
    sd, ed = row.get("start_date"), row.get("end_date")
    if not sd or not ed:
        return np.nan
    try:
        s = pd.to_datetime(sd, errors="coerce")
        e = pd.to_datetime(ed, errors="coerce")
        if pd.isna(s) or pd.isna(e):
            return np.nan
        return float((e - s).days)
    except Exception:
        return np.nan


def n_clusters_for(n: int) -> int:
    if n <= TARGET_CLUSTER_SIZE:
        return 1
    return max(1, round(n / TARGET_CLUSTER_SIZE))


def merge_singletons(
    labels: list[int], rng: np.random.Generator
) -> list[int]:
    """Reassign singletons to a random non-singleton cluster in the same stratum.

    If every cluster has size 1 (stratum of singletons only), merge all into
    cluster 0. If there's exactly one multi-project cluster, singletons fold
    into it.
    """
    if not labels:
        return labels
    counts: dict[int, int] = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    singletons = {lab for lab, c in counts.items() if c == 1}
    non_singletons = [lab for lab, c in counts.items() if c > 1]
    if not singletons:
        return labels
    if not non_singletons:
        return [labels[0]] * len(labels)
    out = list(labels)
    for i, lab in enumerate(out):
        if lab in singletons:
            out[i] = int(rng.choice(non_singletons))
    return out


def assign_random(ids: list[str], rng: np.random.Generator) -> list[int]:
    n = len(ids)
    k = n_clusters_for(n)
    perm = rng.permutation(n)
    labels = np.empty(n, dtype=int)
    # Distribute as evenly as possible into k buckets.
    buckets = np.array_split(perm, k)
    for bi, idxs in enumerate(buckets):
        labels[idxs] = bi
    return labels.tolist()


def assign_agglomerative(features: np.ndarray, n: int) -> list[int]:
    k = n_clusters_for(n)
    if k == 1:
        return [0] * n
    clusterer = AgglomerativeClustering(n_clusters=k, linkage="ward")
    return clusterer.fit_predict(features).tolist()


def build_partition_semantic(stratum_rows: list[dict]) -> list[int]:
    texts = [(r.get("objective") or "").strip() or " " for r in stratum_rows]
    n = len(texts)
    if n < 2:
        return [0] * n
    try:
        vec = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=1,
        )
        X = vec.fit_transform(texts).toarray()
    except ValueError:
        return [0] * n
    return assign_agglomerative(X, n)


def build_partition_rhetorical(stratum_rows: list[dict]) -> list[int]:
    feats = np.stack(
        [rhetorical_features(r.get("objective") or "") for r in stratum_rows]
    )
    n = feats.shape[0]
    if n < 2 or np.allclose(feats.std(axis=0), 0):
        return [0] * n
    feats = StandardScaler().fit_transform(feats)
    return assign_agglomerative(feats, n)


def build_partition_budget(stratum_rows: list[dict]) -> list[int]:
    budget = np.array(
        [float(r.get("ec_max_contribution") or 0) for r in stratum_rows],
        dtype=float,
    )
    dur = np.array([parse_duration_days(r) for r in stratum_rows], dtype=float)
    # Fill NaNs with medians
    if np.isnan(dur).any():
        med = np.nanmedian(dur)
        dur = np.where(np.isnan(dur), med if not np.isnan(med) else 0.0, dur)
    feats = np.column_stack([np.log1p(budget), dur])
    n = feats.shape[0]
    if n < 2 or np.allclose(feats.std(axis=0), 0):
        return [0] * n
    feats = StandardScaler().fit_transform(feats)
    return assign_agglomerative(feats, n)


PARTITIONERS = {
    "random": None,  # handled specially (needs rng)
    "semantic_tfidf": build_partition_semantic,
    "rhetorical": build_partition_rhetorical,
    "budget_duration": build_partition_budget,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in",
        dest="in_path",
        default="/lfs/skampere3/0/alexspan/norm-research/datasets/grant-funding/erc/erc_projects.csv",
    )
    ap.add_argument(
        "--out-dir",
        dest="out_dir",
        default="/lfs/skampere3/0/alexspan/norm-research/datasets/grant-funding/erc",
        help="Directory for per-partition cluster files (erc_clusters__<partition>.csv)",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.in_path, dtype=str, keep_default_na=False)
    print(f"[load] {len(df):,} projects")

    # Exclude schemes without single-panel evaluation
    df = df[~df["funding_scheme"].isin(EXCLUDE_SCHEMES)]
    df = df[df["panel"].str.len() > 0]
    print(f"[filter] after scheme/panel filter: {len(df):,}")

    df["call_year"] = df.apply(lambda r: extract_call_year(r.to_dict()), axis=1)
    print(f"[strata] distinct panels: {df['panel'].nunique()}, years: {sorted(df['call_year'].unique())[:3]}..{sorted(df['call_year'].unique())[-3:]}")

    strata = df.groupby(["panel", "call_year"])
    print(f"[strata] {len(strata)} panel x year strata")

    rng = np.random.default_rng(RANDOM_SEED)
    # rows per partition, post-singleton-merge
    per_partition_rows: dict[str, list[dict]] = {p: [] for p in PARTITIONERS}
    merge_stats = defaultdict(int)

    for (panel, year), sdf in strata:
        srows = sdf.to_dict("records")
        n = len(srows)

        partition_labels: dict[str, list[int]] = {}
        partition_labels["random"] = assign_random(
            [r["id"] for r in srows], rng
        )
        for pname, fn in PARTITIONERS.items():
            if pname == "random":
                continue
            partition_labels[pname] = fn(srows)

        for pname, labels in partition_labels.items():
            labels = merge_singletons(labels, rng)
            merge_stats[pname]  # no-op to keep key present
            # Relabel to 0..k-1 in first-appearance order for stable cluster_ids
            remap: dict[int, int] = {}
            for lab in labels:
                if lab not in remap:
                    remap[lab] = len(remap)
            for r, lab in zip(srows, labels):
                ki = remap[lab]
                cid = f"{pname}__{panel}__{year}__{ki:03d}"
                per_partition_rows[pname].append({
                    "cluster_id": cid,
                    "project_id": r["id"],
                    "panel": panel,
                    "funding_scheme": r.get("funding_scheme", ""),
                    "call_year": year,
                })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[write] per-partition files:")
    for pname, rows in per_partition_rows.items():
        out_path = out_dir / f"erc_clusters__{pname}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["cluster_id", "project_id", "panel",
                            "funding_scheme", "call_year"],
                quoting=csv.QUOTE_ALL,
            )
            w.writeheader()
            w.writerows(rows)
        sizes = pd.DataFrame(rows).groupby("cluster_id").size()
        print(
            f"  {out_path.name:40s}  rows={len(rows):,}  clusters={len(sizes):,}  "
            f"size mean={sizes.mean():.2f}  median={sizes.median():.0f}  "
            f"min={sizes.min()}  max={sizes.max()}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
