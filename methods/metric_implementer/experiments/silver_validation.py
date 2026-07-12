"""Sorensen-style validation of our label-free information certificate against silver human-norm labels.

Sorensen et al. 2022 (ACL) select prompt TEMPLATES by a label-free MI score and validate that per-template
MI correlates (Pearson R) with per-template accuracy. OUR analog: the unit is a METRIC; the label-free
score is the certificate's information content (OPT_Ω, T=H_M, g1, recovery R); the "accuracy" analog is
SILVER SALIENCE — how often real human comments/reviews invoke that metric (from the weak-label pipeline).
The validation is a per-metric rank correlation, per task, with controls Sorensen did not need.

DATA (all verified 2026-07-03):
  - Silver: sk3:/lfs/skampere3/0/alexspan/data/bge_pertask/<task>/matches_joined_<task>.jsonl
    lines {"row","doc","norm","top10":[a-ids],"top10_names":[{id,name}]}. PRE-JOINED — salience is just
    counting a-ids in top10 (rank-ordered best-first). Positional-join risk already eliminated upstream.
    Optional GOLD (5 tasks): matches_<task>.json {"id","aspects":[3 a-ids]} — human, not CE; cleaner.
  - Metric id space: catalog a{N}. a{N} index == <task>_general_r2_expanded.json merged_group[N].
    R2->R3 rollup: r3_expanded merged_group.source_r2_cluster_ids (bare ints -> "a"+str(id)).
  - Certificate: per-metric rows with opt_omega_bits, H_M, gains(->g1), eps_bits_adv, name(==merged_name).

SCORES on the x-axis (label-free instrument):
  OPT_Ω (primary checklist recovery), T=H_M (=I(M_ω;X)), g1 (best single criterion),
  and if present recovery R (reconstruction I(M_ω;M'); may EXCEED OPT_Ω — F-class escape, a real result).

CONTROLS (the point of the exercise):
  1. permutation null on the salience<->score pairing (1000x) -> empirical p, guards shared-size confound.
  2. partial Spearman(sal, OPT | log total_leaf_rubrics, base_rate) -> "attention" not "cluster is big".
  3. reliability/attenuation ceiling: r_silver via split-half of the norm corpus (salience on each half,
     correlate rankings); observed rho <= sqrt(r_MI * r_silver); report disattenuated rho_true.
  4. coverage ceiling (capture-recapture): fraction of silver norm-mass on a-ids with NO scored metric =
     a floor on unexplainable human attention; Chao2/Lincoln-Petersen on {Ω-discovered} vs {silver-invoked}
     metric sets -> est. total human-relevant metrics + coverage. The ONE genuine upper bound on overlap.

CPU-only. Reads local files (pass --data-dir with the per-task artifacts + r2/r3 + cert copied down), or
--sk3 to scp them. Emits per-task JSON + a summary table.

Usage (local, CW already available):
  python -m methods.metric_implementer.experiments.silver_validation \
    --tasks creative_writing --level R3 \
    --matches /tmp/cw_smoke/matches_joined_creative_writing.jsonl \
    --r3 /tmp/cw_smoke/creative-writing_general_r3_expanded.json \
    --cert notebooks/data/prompt_optimality_20260703/cert_8b_v2.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr, rankdata


# --------------------------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------------------------
def load_matches(path):
    """matches_joined lines -> list of {row, doc, norm, top10}."""
    return [json.loads(l) for l in open(path)]


def _norm_name(s):
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()


def load_r2_index(catalog_path, r2_path):
    """a{N} -> {name, description, size}. The catalog defines a{id}->name (authoritative); we NAME-JOIN
    to r2_expanded for description+size. NOTE: a{N} index != r2 group index N (verified: CW 368/368 by
    name but only 227/368 by index) — the join MUST be by name, never by positional index."""
    cat = {}
    for l in open(catalog_path):
        if ":" in l:
            aid, nm = l.split(":", 1)
            cat[aid.strip()] = nm.strip()
    mg = json.load(open(r2_path))["merged_groups"]
    by_name = {_norm_name(g["merged_name"]): g for g in mg}
    out = {}
    for aid, nm in cat.items():
        g = by_name.get(_norm_name(nm))
        # PARENT KEY = the r2_expanded merged_name (the EXACT string the certificate keys metrics by,
        # since the cert is produced by run_alpha_probe seeded from r2_groups=r2_expanded). The catalog
        # name is only the join bridge; using it as the key would miss cert rows on any surface-form diff.
        out[aid] = {"name": (g["merged_name"] if g else nm),
                    "description": (g.get("merged_description", "") if g else ""),
                    "size": (g.get("total_leaf_rubrics", 0) if g else 0),
                    "joined": g is not None}
    return out


def load_r3_rollup(path):
    """r3_expanded: a{N} -> R3 merged_name (via source_r2_cluster_ids), plus per-R3 size."""
    mg = json.load(open(path))["merged_groups"]
    a2r3, r3_size = {}, {}
    for g in mg:
        nm = g["merged_name"]
        r3_size[nm] = g.get("total_leaf_rubrics", 0)
        for cid in g["source_r2_cluster_ids"]:
            a2r3[f"a{cid}"] = nm
    return a2r3, r3_size


def load_cert(path):
    """certificate rows -> {metric_name: {opt, hm, g1, eps, R?}}."""
    rows = json.load(open(path))
    out = {}
    for r in rows:
        if not r.get("gains"):
            continue
        out[r["name"]] = {
            "opt": float(r["opt_omega_bits"]),
            "hm": float(r["H_M"]),
            "g1": float(r["gains"][0]),
            "eps": float(r.get("eps_bits_adv", 0.0)),
            "R": float(r["recovery_bits"]) if "recovery_bits" in r else None,
        }
    return out


# --------------------------------------------------------------------------------------------
# salience
# --------------------------------------------------------------------------------------------
def salience(matches, topk, rank_weighted=False):
    """a{N} -> silver salience. rank_weighted uses 1/rank instead of a flat count."""
    c = Counter()
    for m in matches:
        for rank, a in enumerate(m["top10"][:topk], start=1):
            c[a] += (1.0 / rank) if rank_weighted else 1.0
    return c


def gold_salience(gold_path):
    """matches_<task>.json -> a{N} count over the 3-aspect gold sets (human, not CE)."""
    g = json.load(open(gold_path))
    recs = g if isinstance(g, list) else g.get("records", g.get("data", []))
    c = Counter()
    for r in recs:
        for a in r.get("aspects", []):
            c[a] += 1.0
    return c


def rollup_salience(sal_a, a2parent):
    """roll a{N} salience up to parent metric name; return (parent_salience, unmapped_mass)."""
    par, unmapped = Counter(), 0.0
    for a, v in sal_a.items():
        if a in a2parent:
            par[a2parent[a]] += v
        else:
            unmapped += v
    return par, unmapped


# --------------------------------------------------------------------------------------------
# controls
# --------------------------------------------------------------------------------------------
def perm_null(sal, score, n=1000, seed=0):
    """empirical p that Spearman(sal, score) exceeds a random re-pairing of salience to metrics."""
    obs = spearmanr(sal, score).correlation
    rng = np.random.default_rng(seed)
    s = np.asarray(sal, float)
    null = np.array([spearmanr(rng.permutation(s), score).correlation for _ in range(n)])
    # two-sided by |rho|
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n + 1)
    return float(obs), float(p), float(np.mean(null)), float(np.std(null))


def partial_spearman(y, x, covars):
    """partial Spearman(x, y | covars): rank everything, OLS-residualize x and y on covars, correlate."""
    def resid(v):
        vr = rankdata(v)
        C = np.column_stack([rankdata(c) for c in covars] + [np.ones(len(vr))])
        beta, *_ = np.linalg.lstsq(C, vr, rcond=None)
        return vr - C @ beta
    return float(pearsonr(resid(x), resid(y))[0])   # pearson on ranks-residuals = partial spearman


def split_half_reliability(matches, topk, a2parent, metrics, seed=0):
    """r_silver: split norms in half, salience each half (rolled to parent, restricted to `metrics`),
    correlate the two rankings (Spearman-Brown corrected)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(matches))
    h1 = [matches[i] for i in idx[: len(idx) // 2]]
    h2 = [matches[i] for i in idx[len(idx) // 2:]]
    s1, _ = rollup_salience(salience(h1, topk), a2parent)
    s2, _ = rollup_salience(salience(h2, topk), a2parent)
    v1 = np.array([s1.get(m, 0.0) for m in metrics])
    v2 = np.array([s2.get(m, 0.0) for m in metrics])
    r = spearmanr(v1, v2).correlation
    # Spearman-Brown: reliability of the FULL (double-length) instrument from a half-split correlation
    return float(2 * r / (1 + r)) if r > -1 else float("nan")


def coverage_ceiling(sal_a, a2parent, scored_metrics):
    """capture-recapture on 'which metrics matter'. List A = Ω-discovered (scored_metrics). List B =
    silver-invoked (parents with nonzero salience). Returns coverage frac + Chao/Lincoln-Petersen N_hat
    (est. total human-relevant metrics) + fraction of silver MASS with no scored parent."""
    par, unmapped = rollup_salience(sal_a, a2parent)
    total_mass = sum(sal_a.values())
    invoked = {m for m, v in par.items() if v > 0}
    scored = set(scored_metrics)
    both = invoked & scored
    lincoln_petersen = (len(invoked) * len(scored) / len(both)) if both else float("inf")
    # NB at R2 a2parent covers EVERY catalog a-id, so `unmapped` (no-parent mass) is trivially 0;
    # the real mass ceiling is mass on parents that exist but were never cert-SCORED:
    mass_unscored = sum(v for m, v in par.items() if m not in scored)
    return {
        "n_scored": len(scored),
        "n_silver_invoked": len(invoked),
        "n_overlap": len(both),
        "coverage_of_silver_by_scored": len(both) / len(invoked) if invoked else float("nan"),
        "silver_mass_unmapped_frac": unmapped / total_mass if total_mass else float("nan"),
        "silver_mass_on_unscored_frac": mass_unscored / total_mass if total_mass else float("nan"),
        "N_hat_lincoln_petersen": lincoln_petersen,
    }


# --------------------------------------------------------------------------------------------
# per-task run
# --------------------------------------------------------------------------------------------
def run_task(task, matches_path, cert_path, r2_path=None, r3_path=None, gold_path=None,
             flip_path=None, level="R3", topks=(1, 3, 10), n_perm=1000):
    matches = load_matches(matches_path)
    cert = load_cert(cert_path)

    # r_MI = instrument (form-orbit) reliability per metric = 1 - flip_rate (the fraction of probes whose
    # binarized verdict flips across Φ-reformulations). Real, from the sigs npz; falls back to 0.9 if absent.
    flip = json.load(open(flip_path)) if flip_path and Path(flip_path).exists() else {}
    r_MI_by_name = {nm: (1.0 - v["flip_rate"]) for nm, v in flip.items()
                    if v.get("flip_rate") is not None}

    if level == "R2":
        idx = load_r2_index(r2_path, r3_path)   # r2_path=catalog.txt, r3_path=r2_expanded.json here
        a2parent = {a: idx[a]["name"] for a in idx}
        parent_size = {idx[a]["name"]: idx[a]["size"] for a in idx}
    elif level == "R3":
        a2parent, parent_size = load_r3_rollup(r3_path)
    else:
        raise ValueError(level)

    scored = list(cert.keys())
    opt = np.array([cert[m]["opt"] for m in scored])
    hm = np.array([cert[m]["hm"] for m in scored])
    g1 = np.array([cert[m]["g1"] for m in scored])
    size = np.array([parent_size.get(m, 0) for m in scored], float)
    base = hm  # H_M is a monotone proxy for base-rate distance from 0.5; used as the base-rate covar

    out = {"task": task, "level": level, "n_matches": len(matches), "n_scored": len(scored),
           "by_topk": {}, "gold": None, "coverage": None}

    for topk in topks:
        sal_a = salience(matches, topk)
        par, unmapped = rollup_salience(sal_a, a2parent)
        sal = np.array([par.get(m, 0.0) for m in scored])
        _tot = sum(sal_a.values())
        rec = {"nonzero": int((sal > 0).sum()), "unmapped_mass": unmapped,
               "unmapped_frac": unmapped / _tot if _tot else float("nan")}
        for label, sc in (("OPT", opt), ("T_HM", hm), ("g1", g1)):
            obs, p, mu, sd = perm_null(sal, sc, n=n_perm)
            rec[f"spearman_{label}"] = obs
            rec[f"perm_p_{label}"] = p
        # partial-out size + base-rate on the primary (OPT)
        try:
            rec["partial_OPT_given_size_base"] = partial_spearman(sal, opt, [np.log1p(size), base])
        except Exception as e:
            rec["partial_OPT_given_size_base"] = None
            rec["partial_error"] = str(e)[:100]
        # reliability + attenuation ceiling (uses OPT correlation)
        r_silver = split_half_reliability(matches, topk, a2parent, scored)
        rec["r_silver_splithalf"] = r_silver
        # r_MI = mean form-orbit reliability over scored metrics (real, from flip rates); else 0.9 placeholder.
        rmis = [r_MI_by_name[m] for m in scored if m in r_MI_by_name]
        r_MI = float(np.mean(rmis)) if rmis else 0.9
        rec["r_MI_orbit"] = r_MI
        rec["r_MI_source"] = f"flip_rate ({len(rmis)}/{len(scored)} metrics)" if rmis else "placeholder 0.9"
        ceiling = float(np.sqrt(max(r_MI, 0) * max(r_silver, 0))) if r_silver == r_silver else float("nan")
        rec["attenuation_ceiling_rho"] = ceiling
        rec["rho_true_disattenuated_OPT"] = (rec["spearman_OPT"] / ceiling) if ceiling and ceiling > 0 else None
        out["by_topk"][str(topk)] = rec

    # coverage ceiling (top-10 salience, the most inclusive)
    out["coverage"] = coverage_ceiling(salience(matches, 10), a2parent, scored)

    # gold salience (5 tasks) — cleaner human signal
    if gold_path and Path(gold_path).exists():
        gsal_a = gold_salience(gold_path)
        gpar, gunmapped = rollup_salience(gsal_a, a2parent)
        gsal = np.array([gpar.get(m, 0.0) for m in scored])
        obs, p, _, _ = perm_null(gsal, opt, n=n_perm)
        try:
            gpartial = partial_spearman(gsal, opt, [np.log1p(size), base])
        except Exception:
            gpartial = None
        _gtot = sum(gsal_a.values())
        out["gold"] = {"nonzero": int((gsal > 0).sum()), "unmapped_mass": gunmapped,
                       "gold_total_mass": _gtot,
                       "unmapped_frac": gunmapped / _gtot if _gtot else float("nan"),
                       "gold_mass_on_unscored_frac": (sum(v for m, v in gpar.items() if m not in set(scored)) / _gtot) if _gtot else float("nan"),
                       "spearman_OPT": obs, "perm_p_OPT": p,
                       "spearman_g1": spearmanr(gsal, g1).correlation,
                       "partial_OPT_given_size_base": gpartial,
                       "spearman_gold_vs_size": spearmanr(gsal, size).correlation}
    return out


def _fmt(o):
    lines = [f"\n=== {o['task']} ({o['level']}) — {o['n_matches']} norms, {o['n_scored']} scored metrics ==="]
    for tk, r in o["by_topk"].items():
        lines.append(
            f"  top-{tk}: rho(OPT)={r['spearman_OPT']:+.3f} (p={r['perm_p_OPT']:.3f})  "
            f"rho(g1)={r['spearman_g1']:+.3f}  partial(OPT|size,base)={r['partial_OPT_given_size_base']}  "
            f"nonzero={r['nonzero']}/{o['n_scored']}")
        lines.append(
            f"          r_silver(split-half)={r['r_silver_splithalf']:.3f}  "
            f"attenuation-ceiling={r['attenuation_ceiling_rho']:.3f}  "
            f"rho_true={r['rho_true_disattenuated_OPT']}")
    c = o["coverage"]
    lines.append(f"  COVERAGE: silver-invoked {c['n_silver_invoked']} metrics, overlap {c['n_overlap']}, "
                 f"coverage {c['coverage_of_silver_by_scored']:.2f}, unmapped mass "
                 f"{c['silver_mass_unmapped_frac']:.2f}, N_hat(LP)={c['N_hat_lincoln_petersen']:.0f}")
    if o["gold"]:
        g = o["gold"]
        lines.append(f"  GOLD: rho(OPT)={g['spearman_OPT']:+.3f} (p={g['perm_p_OPT']:.3f}) "
                     f"partial(OPT|size,base)={g['partial_OPT_given_size_base']:+.3f} "
                     f"[gold-vs-size={g['spearman_gold_vs_size']:+.2f}] rho(g1)={g['spearman_g1']:+.3f} "
                     f"nonzero={g['nonzero']}")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", default="creative_writing", help="comma-separated bge task names")
    p.add_argument("--level", default="R3", choices=["R2", "R3"])
    p.add_argument("--matches", help="matches_joined path (single-task mode)")
    p.add_argument("--r2", help="r2_expanded path (level R2)")
    p.add_argument("--r3", help="r3_expanded path (level R3)")
    p.add_argument("--cert", help="certificate json path")
    p.add_argument("--gold", default=None, help="matches_<task>.json gold path (optional)")
    p.add_argument("--flip", default=None, help="json {metric_name: {flip_rate}} for real r_MI (optional)")
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--out", default="/tmp/silver_validation")
    args = p.parse_args(argv)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    for task in args.tasks.split(","):
        o = run_task(task, args.matches, args.cert, r2_path=args.r2, r3_path=args.r3,
                     gold_path=args.gold, flip_path=args.flip, level=args.level, n_perm=args.n_perm)
        print(_fmt(o))
        json.dump(o, open(Path(args.out) / f"{task}_{args.level}.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
