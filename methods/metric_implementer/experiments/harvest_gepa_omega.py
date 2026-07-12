"""Harvest a REAL mined Ω from the GEPA optimizer registry (recon_channel --induce gepa lineage).

Parse every evolved version's rubric (criteria -> checks -> name/description), collect the LEAF atomic
criteria, dedup by normalized text -> the criterion set the optimizer actually discovered. Writes one Ω
file per metric-family and a pooled file. ZERO GPU.

WHERE THIS SITS IN THE §6.5 PIPELINE (revised 2026-06-22, theory notes/2026-06-18):
  1. MINE the *semantic diffs* across the WHOLE lineage (the criteria each accepted mutation ADDED), not
     just the surviving winners -- harvesting every distinct leaf criterion across all versions, as here,
     is the coarse text-level approximation of that diff pool R_pool.
  2. ORTHOGONALIZE -> Ω: the TEXT dedup below (`_dedup`, 8-token-prefix) only collapses near-identical
     STRINGS. The principled step is BEHAVIORAL: score each candidate's per-item signal X_e and keep only
     units NOT already explained by Ω (Shannon-CMI filter) -- run
     `experiments.orthogonalize.orthogonalization_filter` on the scored signals downstream of this file.
  3. CANONICALIZE: assemble subsets with a FIXED section order Format -> Semantics -> Negative Constraints
     (the compiler C(S)), so f(S)=R(C(S)) is a set function.
This module is step 1's text harvest; it does NOT do the behavioral orthogonalization (step 2).

  HOME=/lfs python -m methods.metric_implementer.experiments.harvest_gepa_omega \
      --registry /lfs/skampere3/0/alexspan/tmp_vinfo/gepa_registry --out-dir outputs/.../real_omega
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import re


def _parse_body(b):
    if isinstance(b, dict):
        return b
    if not isinstance(b, str):
        return None
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(b)
        except Exception:
            pass
    return None


_DIAG = re.compile(r"counterfactual|invariance|delta=|moved the score|edited excerpt|\bMISS\b", re.I)


_JUNK_KEYS = ("worked_examples", "scores", "steps", "examples", "score_levels", "levels", "anchors")


def _criteria(obj, out):
    """STRUCTURAL extraction, RECURSIVE: collect criterion nodes (name+description) from `criteria[]`
    and `checks[]` lists ANYWHERE in the rubric tree (handles the schema variation that made 4 families
    return 0), while pruning the contaminating subtrees deterministically:
      * `worked_examples` -> GEPA optimizer-diagnostic leaks (category C)
      * `scores`/`levels` -> score-scale ANCHORS (category B)
      * `steps`           -> procedural sub-steps
    The top-level rubric name/description (TITLE, category D) is skipped because it is never inside a
    `criteria`/`checks` list.
    """
    if isinstance(obj, dict):
        for key in ("criteria", "checks"):
            lst = obj.get(key)
            if isinstance(lst, list):
                for c in lst:
                    if isinstance(c, dict):
                        name, desc = c.get("name", ""), c.get("description", "")
                        if desc and len(desc) > 12 and not _DIAG.search(f"{name} {desc}"):
                            out.append(f"{name}: {desc}".strip(": ").strip() if name else desc)
                        _criteria(c, out)
        for k, v in obj.items():
            if k in _JUNK_KEYS or k in ("criteria", "checks"):
                continue
            # SCHEMA B: criteria stored as TOP-LEVEL NAMED DIMENSIONS {dimname: {description: ...}}.
            # GEPA emits this for some lineages (e.g. aigner runs 1&2: legibility/self-containment/...);
            # the criteria[]-only pass silently dropped them, undercounting Ω. Capture them here.
            if isinstance(v, dict) and isinstance(v.get("description"), str) and len(v["description"]) > 12 \
                    and not _DIAG.search(f"{k} {v['description']}"):
                out.append(f"{k}: {v['description']}")
            if isinstance(v, (dict, list)):
                _criteria(v, out)
    elif isinstance(obj, list):
        for x in obj:
            _criteria(x, out)


def _steps(obj, out):
    """FINE-granularity extraction: the procedural `steps` under each check (e.g. 'Check that all
    mathematical symbols are used correctly'). These are the most atomic units GEPA's rubric exposes;
    prunes diagnostic-leak strings. Used to test Ω=criteria vs Ω=steps (the granularity-floor experiment)."""
    if isinstance(obj, dict):
        for key in ("criteria", "checks"):
            lst = obj.get(key)
            if isinstance(lst, list):
                for c in lst:
                    if isinstance(c, dict):
                        for s in (c.get("steps") or []):
                            if isinstance(s, str) and len(s) > 12 and not _DIAG.search(s):
                                out.append(s.strip())
                        _steps(c, out)
        for k, v in obj.items():
            if k not in _JUNK_KEYS and k not in ("criteria", "checks") and isinstance(v, (dict, list)):
                _steps(v, out)
    elif isinstance(obj, list):
        for x in obj:
            _steps(x, out)


def _norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def _dedup(crits):
    seen, out = set(), []
    for c in crits:
        k = " ".join(_norm(c).split()[:8])
        if k and k not in seen:
            seen.add(k)
            out.append(c)
    return out


def _family(metric):
    return re.sub(r"_\d+$", "", metric)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="/lfs/skampere3/0/alexspan/tmp_vinfo/gepa_registry")
    ap.add_argument("--level", default="criteria", choices=["criteria", "steps"],
                    help="criteria = high/mid dimensions (criteria+checks); steps = fine procedural units")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    if a.out_dir is None:
        a.out_dir = ("outputs/metric_implementer_scale/real_omega" if a.level == "criteria"
                     else "outputs/metric_implementer_scale/real_omega_steps")
    extract = _criteria if a.level == "criteria" else _steps
    os.makedirs(a.out_dir, exist_ok=True)

    fam_raw, fam_versions = {}, {}
    for vf in sorted(glob.glob(f"{a.registry}/metrics/*/versions/v*__prompt.json")):
        metric = os.path.basename(os.path.dirname(os.path.dirname(vf)))
        fam = _family(metric)
        try:
            obj = _parse_body(json.load(open(vf)).get("body"))
        except Exception:
            obj = None
        if not obj:
            continue
        out = []
        extract(obj, out)
        fam_raw.setdefault(fam, []).extend(out)
        fam_versions[fam] = fam_versions.get(fam, 0) + 1

    print(f"{len(fam_raw)} metric-families with evolved lineage\n")
    print(f"{'family':52s}{'versions':>9s}{'raw':>6s}{'distinct':>9s}")
    summary = {}
    for fam in sorted(fam_raw):
        uniq = _dedup(fam_raw[fam])
        summary[fam] = len(uniq)
        short = fam.replace("claude-parsed__", "")[:50]
        print(f"{short:52s}{fam_versions[fam]:>9d}{len(fam_raw[fam]):>6d}{len(uniq):>9d}")
        with open(os.path.join(a.out_dir, f"{short}.txt"), "w") as f:
            f.write(f"Criteria mined from real GEPA lineage of {short}:\n")
            for c in uniq:
                f.write(f"- {c}\n")
    # pooled across all families (global Ω)
    allu = _dedup([c for v in fam_raw.values() for c in v])
    with open(os.path.join(a.out_dir, "_pooled.txt"), "w") as f:
        f.write("Pooled criteria mined from ALL real GEPA lineage:\n")
        for c in allu:
            f.write(f"- {c}\n")
    print(f"\npooled distinct criteria across all families: {len(allu)}")
    print(f"wrote per-family + _pooled.txt to {a.out_dir}")
    # show a few real examples
    print("\nsample distinct criteria (first family):")
    f0 = sorted(fam_raw)[0]
    for c in _dedup(fam_raw[f0])[:8]:
        print(f"  - {c[:100]}")


if __name__ == "__main__":
    main()
