"""mine_clusters — harvest a rich candidate-criteria pool from the curated rubric-cluster hierarchy.

The leaf-rubric clustering (notes/2026-05-19__structural-metrics.md) organizes 53K human-authored
rubric forms (11 tasks) into a granularity hierarchy. Two levels are materialized on disk and are
DIRECT candidate-criteria sources — no decomposition, no GPU:

  * **L0** = "same rubric, restated"   — `clusters_<task>.json`  ({provenance_key: cluster_id}).
    One representative canonical text per L0 cluster → the FINEST atomic units.
  * **R1** = "different rubrics, same principle" — `r1_families_<task>.json`
    ({families: [{family_id, name, description, cluster_ids}], cluster_to_family}). Each R1 family
    is an LLM-WRITTEN name+description → a READY-MADE atomic criterion, the coarse/legible backbone.

WHY THIS EXISTS (theory §6.5 / missing-impact §6.7c): GEPA-only mining decomposes a single optimized
prompt and collapses Ω to ~3-6 (the under-mining bug). The cluster hierarchy is a CORPUS-COVERAGE
source — every human-authored norm is represented — so unioning it with GEPA's OPTIMIZATION-RELEVANCE
source lets |Ω| reach the ~8-25 the tail-bound needs. The two are complementary: GEPA anchors in what
is causally relevant to THIS executor X; the clusters guarantee coverage of the normative universe.
(Side-channel criteria — length/format — are NOT in the corpus; those still need adversarial_saturation.)

ZERO GPU. CLI:
  python -m methods.metric_implementer.experiments.mine_clusters --task creative-writing
  python -m methods.metric_implementer.experiments.mine_clusters --task peer-review --levels R1,L0 --pool-max 120
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Local mirror of sk3 /lfs/.../norm_embed/match_out (see notes/2026-05-19__structural-metrics.md).
_STRUCT_DIR = str(_REPO_ROOT / "outputs" / "analyses" / "structural_metrics")
_CANON = str(_REPO_ROOT / "outputs" / "analyses" / "canon_all_real_forms.jsonl")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()


def _dedup(crits):
    """Same 8-token-prefix dedup the GEPA harvester uses, so union sources de-dupe consistently."""
    seen, out = set(), []
    for c in crits:
        k = " ".join(_norm(c).split()[:8])
        if k and k not in seen:
            seen.add(k)
            out.append(c)
    return out


def r1_criteria(task: str):
    """R1 families → 'name: description' strings. The coverage backbone (LLM-written, legible)."""
    fn = os.path.join(_STRUCT_DIR, f"r1_families_{task}.json")
    if not os.path.exists(fn):
        return []
    d = json.load(open(fn))
    fams = d.get("families", []) if isinstance(d, dict) else []
    out = []
    for f in fams:
        name, desc = (f.get("name") or "").strip(), (f.get("description") or "").strip()
        if not desc:
            continue
        out.append(f"{name}: {desc}".strip(": ").strip() if name else desc)
    return out


# --------------------------------------------------------------------------------------------
# R2 = "different principles, same metric" — `outputs/hierarchy/<task>_<bucket>_r2_expanded.json`.
# Each merged_group is ONE R2 METRIC: a family with a holistic merged_description (→ GEPA seed) and
# all_leaves (→ the family's child criteria). Buckets general/specific/hyper_specific OVERLAP — fix
# ONE bucket per run to avoid double-counting the same cluster.
# --------------------------------------------------------------------------------------------

_HIER_DIR = str(_REPO_ROOT / "outputs" / "hierarchy")


def _expanded_groups(task: str, bucket: str, level: str):
    """merged_groups from ``<task>_<bucket>_<level>_expanded.json`` — R2 and R3 SHARE this format
    (each merged_group: merged_name, merged_description, all_leaves:[{key,name}]). Returns
    [{group_idx, merged_name, merged_description, total_leaf_rubrics, all_leaves}], or [] if absent.
    `group_idx` is the position in `merged_groups` — pass it to the matching `_children`."""
    fn = os.path.join(_HIER_DIR, f"{task}_{bucket}_{level}_expanded.json")
    if not os.path.exists(fn):
        return []
    d = json.load(open(fn))
    out = []
    for i, g in enumerate(d.get("merged_groups", []) or []):
        out.append({"group_idx": i, "merged_name": g.get("merged_name", ""),
                    "merged_description": g.get("merged_description", ""),
                    "total_leaf_rubrics": g.get("total_leaf_rubrics", 0),
                    "all_leaves": g.get("all_leaves", []) or []})
    return out


def r2_groups(task: str, bucket: str = "specific"):
    """All R2 merged_groups in one specificity bucket. `group_idx` → pass to `r2_children`."""
    return _expanded_groups(task, bucket, "r2")


def r3_groups(task: str, bucket: str = "general"):
    """R3 merged_groups — CW's coarsest MERGED level (fewer, broader groups; e.g. 70 vs R2's 371).
    Same on-disk format as R2, different level suffix."""
    return _expanded_groups(task, bucket, "r3")


def r2_criteria(task: str, bucket: str = "specific"):
    """R2 groups → 'merged_name: merged_description' strings (parallels r1_criteria)."""
    return [f"{g['merged_name']}: {g['merged_description']}".strip(": ").strip()
            for g in r2_groups(task, bucket) if g["merged_description"]]


def _expanded_children(task: str, bucket: str, level: str, group_idx: int):
    """Scoped child-criteria pool for ONE R2/R3 metric = its `all_leaves` names (the family's own
    criteria). Source (c) scoped to the family — the coverage backbone for the certificate."""
    groups = _expanded_groups(task, bucket, level)
    if not (0 <= group_idx < len(groups)):
        return []
    return [lf.get("name", "").strip() for lf in groups[group_idx]["all_leaves"]
            if lf.get("name") and len(lf.get("name", "")) > 4]


def r2_children(task: str, bucket: str, group_idx: int):
    """Scoped child-criteria pool for ONE R2 metric = its `all_leaves` names."""
    return _expanded_children(task, bucket, "r2", group_idx)


def r3_children(task: str, bucket: str, group_idx: int):
    """Scoped child-criteria pool for ONE R3 metric = its `all_leaves` names."""
    return _expanded_children(task, bucket, "r3", group_idx)


def _l0_key_to_canon(task: str):
    """Stream canon_all_real_forms.jsonl, yield (key, canonical) for `task` only."""
    if not os.path.exists(_CANON):
        return
    with open(_CANON) as f:
        for line in f:
            try:
                o = json.loads(line)
            except Exception:
                continue
            if o.get("task") != task:
                continue
            key, canon = o.get("key"), o.get("canonical")
            if key and canon:
                yield key, canon


def l0_reps(task: str):
    """One representative canonical text per L0 cluster (the median-length member — avoids
    degenerate short/long restatements). Joins clusters_<task>.json (key→cid) × canon (key→text)."""
    fn = os.path.join(_STRUCT_DIR, f"clusters_{task}.json")
    if not os.path.exists(fn) or not os.path.exists(_CANON):
        return []
    key2cid = json.load(open(fn))            # {provenance_key: cluster_id}
    cid2texts: dict = {}
    n_hit = 0
    for key, canon in _l0_key_to_canon(task):
        cid = key2cid.get(key)
        if cid is None:
            continue
        cid2texts.setdefault(cid, []).append(canon)
        n_hit += 1
    reps = []
    for cid, texts in cid2texts.items():
        if not texts:
            continue
        texts.sort(key=len)
        reps.append(texts[len(texts) // 2])  # median-length member = the representative
    return reps, n_hit


def _cid2rep(task: str) -> dict:
    """``{cluster_id: representative canonical text}`` — the median-length member per L0 cluster.
    Used to materialize R1 family children from their `cluster_ids`."""
    fn = os.path.join(_STRUCT_DIR, f"clusters_{task}.json")
    if not os.path.exists(fn) or not os.path.exists(_CANON):
        return {}
    key2cid = json.load(open(fn))
    cid2texts: dict = {}
    for key, canon in _l0_key_to_canon(task):
        cid = key2cid.get(key)
        if cid is None:
            continue
        cid2texts.setdefault(cid, []).append(canon)
    out = {}
    for cid, texts in cid2texts.items():
        texts.sort(key=len)
        out[cid] = texts[len(texts) // 2]
    return out


# --------------------------------------------------------------------------------------------
# R1 = "different rubrics, same principle" — `r1_families_<task>.json`
# ({families:[{family_id, name, description, cluster_ids}]}). Each family is an LLM-WRITTEN
# name+description → a metric whose children are its `cluster_ids`' L0 representative texts.
# NOTE for CW: R1 over-fragmented (~1700 families, median 1 leaf) → degenerate as standalone
# metrics (≈ L0 atoms); SAMPLE rather than enumerate, or use R3 for a real coarse level.
# --------------------------------------------------------------------------------------------

def r1_groups(task: str):
    """R1 families as metric groups: [{group_idx, merged_name, merged_description,
    total_leaf_rubrics, all_leaves:[{name}]}]. `merged_description` is the family description (→ M_i /
    GEPA seed); children = the L0 representative texts of the family's `cluster_ids`."""
    fn = os.path.join(_STRUCT_DIR, f"r1_families_{task}.json")
    if not os.path.exists(fn):
        return []
    d = json.load(open(fn))
    fams = d.get("families", []) if isinstance(d, dict) else []
    cid2rep = _cid2rep(task)
    out = []
    for i, f in enumerate(fams):
        name = (f.get("name") or "").strip()
        desc = (f.get("description") or "").strip()
        cids = f.get("cluster_ids", []) or []
        leaves = [{"name": cid2rep[c]} for c in cids if c in cid2rep]
        if not leaves and name:                 # no canon rep resolved → seed with the family name
            leaves = [{"name": name}]
        out.append({"group_idx": i, "merged_name": name, "merged_description": desc,
                    "total_leaf_rubrics": len(cids), "all_leaves": leaves})
    return out


def r1_children(task: str, group_idx: int):
    """Scoped child-criteria pool for ONE R1 family = its `cluster_ids`' L0 representative texts."""
    groups = r1_groups(task)
    if not (0 <= group_idx < len(groups)):
        return []
    return [lf.get("name", "").strip() for lf in groups[group_idx]["all_leaves"]
            if lf.get("name") and len(lf.get("name", "")) > 4]


def mine(task: str, levels=("R1", "L0"), pool_max: int | None = None, want_provenance: bool = False,
         bucket: str = "specific"):
    """Union the requested levels → dedup'd candidate pool. `want_provenance` returns
    [{criterion, level}] so the pool COMPOSITION (which source each came from) is inspectable —
    useful for diagnosing whether Ω is GEPA- or corpus-driven and for relevance-weighting. `bucket`
    selects the R2 specificity hierarchy (general/specific/hyper_specific — they OVERLAP, fix one)."""
    pool, l0_hit = [], None
    if "R2" in levels:
        for c in r2_criteria(task, bucket):
            pool.append((c, "R2"))
    if "R1" in levels:
        for c in r1_criteria(task):
            pool.append((c, "R1"))
    if "L0" in levels:
        reps, l0_hit = l0_reps(task)
        for c in reps:
            pool.append((c, "L0"))
    # de-dupe by text (across levels too — an R1 family often restates its member L0 cluster)
    seen, dedup = set(), []
    for c, lvl in pool:
        k = " ".join(_norm(c).split()[:8])
        if k and k not in seen:
            seen.add(k)
            dedup.append((c, lvl))
    if pool_max:
        dedup = dedup[:pool_max]
    if want_provenance:
        return [{"criterion": c, "level": lvl} for c, lvl in dedup], {"l0_keys_matched": l0_hit}
    return [c for c, _ in dedup], {"l0_keys_matched": l0_hit, "n_levels": list(levels)}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mine_clusters", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--levels", default="R1,L0", help="comma list of {R1,L0}")
    ap.add_argument("--pool-max", type=int, default=None)
    ap.add_argument("--out", default=None, help="write criteria one-per-line (else just print stats)")
    a = ap.parse_args(argv)
    levels = tuple(x.strip().upper() for x in a.levels.split(",") if x.strip())
    pool, meta = mine(a.task, levels=levels, pool_max=a.pool_max, want_provenance=False)
    by = {}
    # recount per level (post-dedup) for the readout
    prov, _ = mine(a.task, levels=levels, pool_max=a.pool_max, want_provenance=True)
    by = {}
    for p in prov:
        by[p["level"]] = by.get(p["level"], 0) + 1
    print(f"[{a.task}] levels={list(levels)} → {len(pool)} candidate criteria "
          f"(by level: {by}; L0 canon-keys matched: {meta.get('l0_keys_matched')})")
    for c in pool[:6]:
        print(f"  - {c[:110]}")
    if a.out:
        with open(a.out, "w") as f:
            for c in pool:
                f.write(f"- {c}\n")
        print(f"wrote {len(pool)} criteria to {a.out}")


if __name__ == "__main__":
    main()
