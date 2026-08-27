#!/usr/bin/env python3
"""ANALYSIS 2 -- Good-Turing / missing-mass prototype for the Layer-3 closure pilot.

Motivated by the capture-recapture discipline from the prompt-scaling line: mining
moves the BOUND'S LEVEL, the audit has to give its WIDTH.  The pilot stopped on a
2-consecutive-sub-epsilon rule, which says "the last two draws were small"; it does
not say how much articulable signal was left unmined.  This script prototypes the
missing quantity.

(a) DECAY EXTRAPOLATION.  Fit the observed per-round marginal AUC gains on the
    honest (1,244 dense-held-out) population with simple decay laws and extrapolate
    the remaining recoverable mass sum_{r>=5} g_r, with a group-bootstrap CI.
    Also, as an independent corroborating series, extrapolate the per-round
    best-new-criterion alone-AUC excess from the round mechanism JSONs.

(b) REDUNDANCY / RECAPTURE CENSUS.  Embed the 56 post-gate mined criteria and the
    original 154-criterion A bank with BAAI/bge-large-en-v1.5, sweep a cosine
    threshold, and count (i) near-duplicate pairs among the mined criteria
    (cross-round recapture), (ii) mined criteria that duplicate something already in
    the bank (leak past the anti-duplication instruction), and (iii) Chao1 on
    concept-species using singleton/doubleton cluster counts.

(c) is a design, written up in notes/2026-08-06__closure-swap-and-missing-mass.md.

HONESTY NOTES carried into the output JSON:
  * The bootstrap CI in (a) is READOUT sampling noise only.  It does NOT include
    proposer variance -- a different proposer, or GEPA-iterated phrasing, would move
    the gains far more than the CI width.  Treat it as a lower bound on uncertainty.
  * The pilot's rounds were SEQUENTIAL and each proposer was shown the current bank
    and told not to duplicate it.  Recapture was therefore suppressed by design, so
    the Chao1 number here is a LOWER bound on richness computed from an invalid
    sampling design.  It is reported to exercise the machinery, not as an estimate.
    (c) exists precisely to fix that with P independent proposers per round.

CPU only.  Requires round_preds_all.npz.  Usage: python missing_mass.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.special import zeta

HERE = Path(__file__).resolve().parent
ROUNDS = ["round0", "round1", "round2", "round3", "round4"]
N_BOOT = 2000
SEED = 0
THRESHOLDS = [0.70, 0.75, 0.78, 0.80, 0.85, 0.90]


# ============================================================ (a) decay fits ==
def fit_geometric(g):
    """g_r = a * lam^(r-1), r = 1..R.  Returns (a, lam, remaining, resid)."""
    r = np.arange(1, len(g) + 1)

    def f(p):
        return p[0] * p[1] ** (r - 1) - g

    sol = least_squares(f, x0=[max(g[0], 1e-5), 0.7], bounds=([0.0, 0.01], [1.0, 0.98]))
    a, lam = sol.x
    remaining = a * lam ** len(g) / (1.0 - lam)
    return {"a": float(a), "lam": float(lam), "remaining": float(remaining),
            "next_round_gain": float(a * lam ** len(g)),
            "sse": float(np.sum(sol.fun ** 2)),
            "at_bound": bool(lam <= 0.0101 or lam >= 0.9799)}


def fit_saturating_level(V):
    """V_r = Vinf - c*lam^r, r = 0..R.  Remaining = Vinf - V_R."""
    r = np.arange(len(V))

    def f(p):
        return (p[0] - p[1] * p[2] ** r) - V

    sol = least_squares(f, x0=[V[-1] + 0.005, max(V[-1] - V[0], 1e-4), 0.7],
                        bounds=([V[-1] - 0.05, 0.0, 0.01], [V[-1] + 0.5, 1.0, 0.98]))
    Vinf, c, lam = sol.x
    R = len(V) - 1
    return {"Vinf": float(Vinf), "c": float(c), "lam": float(lam),
            "remaining": float(Vinf - V[-1]),
            "next_round_gain": float(c * lam ** R * (1 - lam)),
            "sse": float(np.sum(sol.fun ** 2)),
            "at_bound": bool(lam <= 0.0101 or lam >= 0.9799)}


def fit_powerlaw(g):
    """g_r = a * r^(-b).  Remaining = a * zeta(b, R+1) (Hurwitz), needs b > 1."""
    r = np.arange(1, len(g) + 1)

    def f(p):
        return p[0] * r ** (-p[1]) - g

    sol = least_squares(f, x0=[max(g[0], 1e-5), 2.0], bounds=([0.0, 1.05], [1.0, 25.0]))
    a, b = sol.x
    remaining = float(a * zeta(b, len(g) + 1))
    return {"a": float(a), "b": float(b), "remaining": remaining,
            "next_round_gain": float(a * (len(g) + 1) ** (-b)),
            "sse": float(np.sum(sol.fun ** 2)), "at_bound": bool(b <= 1.06 or b >= 24.9)}


def auc_from_weights(y, p, w):
    """Weighted AUC via the pairwise definition (ties counted 0.5)."""
    P = np.where(y == 1)[0]
    N = np.where(y == 0)[0]
    s = np.sign(p[P][:, None] - p[N][None, :])
    c = (s > 0) + 0.5 * (s == 0)
    wp, wn = w[P], w[N]
    return float((wp @ c @ wn) / (wp.sum() * wn.sum()))


def part_a():
    z = np.load(HERE / "round_preds_all.npz", allow_pickle=True)
    held = z["held"].astype(bool)
    y = z["y"][held]
    nt = z["ntitle"][held]
    va = np.array([z[f"va_nl_{r}"][held] for r in ROUNDS])

    levels = np.array([auc_from_weights(y, va[r], np.ones(len(y))) for r in range(5)])
    gains = np.diff(levels)

    pt = {
        "honest_VA_nl_levels": levels.tolist(),
        "honest_per_round_gains": gains.tolist(),
        "geometric_on_gains": fit_geometric(gains),
        "saturating_on_levels": fit_saturating_level(levels),
        "powerlaw_on_gains": fit_powerlaw(gains),
    }

    # ---- group bootstrap of the READOUT (prediction vectors held fixed) -------
    rng = np.random.default_rng(SEED)
    uniq, codes = np.unique(nt, return_inverse=True)
    ng = len(uniq)
    keys = ["rem_geometric", "rem_saturating", "rem_powerlaw",
            "g5_geometric", "g5_saturating", "g5_powerlaw",
            "lam_geom", "gain_sum_observed"]
    boot = {k: [] for k in keys}
    ok_geom = []          # non-degenerate geometric fits only
    nbound = {"geometric": 0, "saturating": 0, "powerlaw": 0}
    for _ in range(N_BOOT):
        draw = rng.integers(0, ng, size=ng)
        mult = np.bincount(draw, minlength=ng).astype(float)
        w = mult[codes]
        if w[y == 1].sum() == 0 or w[y == 0].sum() == 0:
            continue
        lv = np.array([auc_from_weights(y, va[r], w) for r in range(5)])
        gg = np.diff(lv)
        fg, fs, fp = fit_geometric(gg), fit_saturating_level(lv), fit_powerlaw(gg)
        boot["rem_geometric"].append(fg["remaining"])
        boot["rem_saturating"].append(fs["remaining"])
        boot["rem_powerlaw"].append(fp["remaining"])
        boot["g5_geometric"].append(fg["next_round_gain"])
        boot["g5_saturating"].append(fs["next_round_gain"])
        boot["g5_powerlaw"].append(fp["next_round_gain"])
        boot["lam_geom"].append(fg["lam"])
        boot["gain_sum_observed"].append(float(gg.sum()))
        if not fg["at_bound"]:
            ok_geom.append(fg["remaining"])
        for k, f in (("geometric", fg), ("saturating", fs), ("powerlaw", fp)):
            nbound[k] += int(f["at_bound"])

    def ci(a):
        a = np.array(a, dtype=float)
        a = a[np.isfinite(a)]
        return {"median": float(np.median(a)), "lo": float(np.percentile(a, 2.5)),
                "hi": float(np.percentile(a, 97.5)),
                "q25": float(np.percentile(a, 25)), "q75": float(np.percentile(a, 75)),
                "mean": float(a.mean()), "p_gt_005": float((a > 0.005).mean()),
                "n": int(len(a))}

    nd = len(boot["rem_geometric"])
    pt["bootstrap"] = {
        "n_draws": nd,
        "note": "group (ntitle) bootstrap of the 1,244-row honest readout with the "
                "per-round prediction vectors HELD FIXED; captures readout sampling "
                "noise only, NOT proposer variance (which is larger)",
        "remaining_geometric": ci(boot["rem_geometric"]),
        "remaining_saturating": ci(boot["rem_saturating"]),
        "remaining_powerlaw_NOT_IDENTIFIED": ci(boot["rem_powerlaw"]),
        "remaining_geometric_nondegenerate_fits_only": ci(ok_geom),
        "next_round_gain_g5_geometric": ci(boot["g5_geometric"]),
        "next_round_gain_g5_saturating": ci(boot["g5_saturating"]),
        "lambda_geometric": ci(boot["lam_geom"]),
        "observed_4round_gain_sum": ci(boot["gain_sum_observed"]),
        "fits_hitting_a_bound": {k: v / max(nd, 1) for k, v in nbound.items()},
        "identification_warning": (
            "a 4-point gain series does not identify a decay rate: the geometric "
            "lambda bootstrap runs to BOTH bounds, so the upper end of every "
            "remaining-mass CI is an artefact of lambda -> 1 replicates rather than "
            "evidence of a large hidden tail.  The power-law fit is at-bound in the "
            "point fit itself and must not be quoted as an estimate.  The "
            "decision-relevant statistic is the predicted NEXT-round gain g5, which "
            "is far better conditioned than the infinite-tail sum."
        ),
    }

    # ---- lambda sensitivity band (lambda is the badly-identified parameter) ---
    a_hat = pt["geometric_on_gains"]["a"]
    pt["lambda_sensitivity_remaining"] = {
        f"lam={lam}": float(a_hat * lam ** 4 / (1 - lam)) for lam in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    }
    pt["lambda_sensitivity_anchored_on_g4"] = {
        f"lam={lam}": float(gains[-1] * lam / (1 - lam)) for lam in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    }
    pt["lambda_sensitivity_anchored_on_g3"] = {
        f"lam={lam}": float(gains[-2] * lam ** 2 / (1 - lam)) for lam in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    }

    # ---- independent series: best-new-criterion alone-AUC excess per round ----
    keys = {1: "per_A_criterion_MONITOR", 2: "per_A2_criterion_MONITOR",
            3: "per_A3_criterion_MONITOR", 4: "per_A4_criterion_MONITOR"}
    per_round = {}
    for r in (1, 2, 3, 4):
        m = json.loads((HERE / f"round{r}_mechanism.json").read_text())[keys[r]]
        aucs = np.array([v["alone_AUC_vs_y_MONITOR"] for v in m.values()])
        rhos = np.array([v["rho_vs_dense_MONITOR"] for v in m.values()])
        exc = np.abs(aucs - 0.5)
        per_round[f"round{r}"] = {
            "n_criteria": int(len(aucs)),
            "best_alone_AUC": float(aucs.max()),
            "best_excess": float(exc.max()),
            "mean_excess": float(exc.mean()),
            "median_excess": float(np.median(exc)),
            "sum_excess": float(exc.sum()),
            "frac_excess_below_.01": float((exc < 0.01).mean()),
            "best_rho_vs_dense": float(np.abs(rhos).max()),
            "mean_abs_rho_vs_dense": float(np.abs(rhos).mean()),
        }
    best_exc = np.array([per_round[f"round{r}"]["best_excess"] for r in (1, 2, 3, 4)])
    sum_exc = np.array([per_round[f"round{r}"]["sum_excess"] for r in (1, 2, 3, 4)])
    # geometric decay of the per-round criterion-value series (r indexed 1..4)
    def geom_series(v):
        rr = np.arange(1, len(v) + 1)
        sol = least_squares(lambda p: p[0] * p[1] ** (rr - 1) - v,
                            x0=[v[0], 0.8], bounds=([0.0, 0.01], [10.0, 0.995]))
        a, lam = sol.x
        # rounds until best-of-k excess falls under .01 (i.e. nothing individually readable)
        n_more = float(np.log(0.01 / max(a, 1e-9)) / np.log(lam)) if lam < 1 else np.inf
        return {"a": float(a), "lam": float(lam), "sse": float(np.sum(sol.fun ** 2)),
                "rounds_until_excess_below_.01": n_more,
                "remaining_sum_beyond_r4": float(a * lam ** 4 / (1 - lam))}
    # ---- value-without-increment: do new criteria carry signal that fails to add? --
    blk_key = {1: "A_block_alone", 2: "A2_block_alone", 3: "A3_block_alone",
               4: "A4_block_alone"}
    curve = json.loads((HERE / "round4_results.json").read_text())["closure_curve"]
    conv = {}
    for r in (1, 2, 3, 4):
        m = json.loads((HERE / f"round{r}_mechanism.json").read_text())[blk_key[r]]
        alone = m["nl_MONITOR_all"]
        gain_mon = curve[f"VA_nl_gain_MONITOR_r{r}"]
        conv[f"round{r}"] = {
            "new_block_alone_AUC_MONITOR": alone,
            "new_block_alone_excess": alone - 0.5,
            "stack_gain_MONITOR": gain_mon,
            "stack_gain_honest": float(gains[r - 1]),
            "conversion_alone_excess_to_stack_gain_MONITOR": gain_mon / (alone - 0.5),
            "conversion_alone_excess_to_stack_gain_honest": float(gains[r - 1]) / (alone - 0.5),
        }
    pt["value_without_increment"] = {
        "per_round": conv,
        "reading": "every round's new block carries real standalone signal (alone AUC "
                   ".553-.650 on MONITOR) yet converts only a few percent of it into a "
                   "stack increment.  Saturation here is REDUNDANCY saturation -- the "
                   "proposer keeps finding readable criteria that the existing bank "
                   "already spans -- not exhaustion of readable criteria.",
    }

    pt["criterion_value_series"] = {
        "per_round": per_round,
        "best_excess_geometric": geom_series(best_exc),
        "sum_excess_geometric": geom_series(sum_exc),
        "note": "alone-AUC excess = |alone AUC - .5| on MONITOR, from the round "
                "mechanism JSONs.  This is a per-CRITERION value series, independent "
                "of the stacked VA_nl gain series above; it is a corroborating "
                "diagnostic, not a second estimate of the same quantity.",
    }
    return pt


# ================================================= (b) redundancy / recapture ==
BOILER = re.compile(
    r"^(composite:\s*|score\s+0-10\s*(on|for|how)?\s*|score\s+the\s+)", re.I)


def load_concepts():
    bank = [json.loads(l) for l in open(HERE / "ref" / "rubrics_154.jsonl") if l.strip()]
    bank_recs = [{"set": "bank154", "id": f"bank:{b['rubric_id']}", "round": 0,
                  "name": b["name"], "text": b["description"]} for b in bank]

    mined = []
    for r in (1, 2, 3, 4):
        prop = {c["id"]: c for c in
                json.loads((HERE / f"round{r}_proposals_blinded.json").read_text())["criteria"]}
        routing = json.loads((HERE / f"round{r}_routing_final.json").read_text())
        rep = json.loads((HERE / f"round{r}_score_report.json").read_text())
        collapsed = {k for k, v in rep["per_criterion"].items() if v.get("collapsed")}
        for x in routing["final"]:
            if x["final_route"] == "A" and x["blind_id"] not in collapsed:
                c = prop[x["blind_id"]]
                mined.append({"set": "mined56", "id": f"r{r}:{x['blind_id']}", "round": r,
                              "name": c["name"],
                              "text": BOILER.sub("", c["instruction"]).strip()})
    return bank_recs, mined


def embed(texts, batch=16):
    import torch
    from transformers import AutoModel, AutoTokenizer
    name = "BAAI/bge-large-en-v1.5"
    tok = AutoTokenizer.from_pretrained(name)
    mod = AutoModel.from_pretrained(name).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True,
                      max_length=256, return_tensors="pt")
            h = mod(**enc).last_hidden_state[:, 0]  # bge = CLS pooling
            out.append(torch.nn.functional.normalize(h, dim=-1).numpy())
    return np.concatenate(out)


def single_linkage(S, tau):
    """Cluster ids by single linkage at cosine >= tau (union-find)."""
    n = S.shape[0]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] >= tau:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    lab = np.array([find(i) for i in range(n)])
    _, lab = np.unique(lab, return_inverse=True)
    return lab


def chao1(sizes):
    """Chao1 (+ bias-corrected) from a cluster-size (species-frequency) vector."""
    sizes = np.asarray(sizes)
    S = int(len(sizes))
    f1 = int((sizes == 1).sum())
    f2 = int((sizes == 2).sum())
    classic = S + (f1 ** 2) / (2 * f2) if f2 > 0 else float("nan")
    bc = S + f1 * (f1 - 1) / (2 * (f2 + 1))
    n = int(sizes.sum())
    return {"S_obs": S, "f1_singletons": f1, "f2_doubletons": f2, "n_proposals": n,
            "chao1": float(classic) if f2 > 0 else None,
            "chao1_bias_corrected": float(bc),
            "good_turing_missing_mass": f1 / n if n else float("nan")}


def bank_recapture_census():
    """The ONE place in this pilot with a usable capture-recapture design.

    The "154-criterion A bank" was not authored as 154 distinct concepts: it was
    produced by clustering aspects harvested from many independent source rubrics,
    and the merge left EXACT repeats in place.  Exact-name/description repeats are
    therefore genuine RECAPTURES of the same concept by independent draws, which is
    exactly the structure Good-Turing and Chao1 need and exactly the structure the
    pilot's sequential, anti-duplication-instructed mining destroys.

    Two frequency vectors are available:
      (i)  exact-string repeat counts across the 154 delivered criteria;
      (ii) `n_aspects_in_cluster` -- how many source aspects the bank builder
           merged into each criterion (184 aspects -> 154 criteria).
    """
    import sys
    sys.path.insert(0, str(HERE))
    import closure_lib as L

    rows = [json.loads(l) for l in open(HERE / "ref" / "rubrics_154.jsonl") if l.strip()]
    from collections import Counter, defaultdict

    by_name = defaultdict(list)
    for r in rows:
        by_name[r["name"]].append(r["rubric_id"])
    sizes = np.array([len(v) for v in by_name.values()])
    n_distinct_desc = len({r["description"] for r in rows})

    aspects = np.array([r["n_aspects_in_cluster"] for r in rows])

    # effective feature count actually entering round 0, after the degeneracy screen
    pop = L.load_population()
    _, split, _, _ = L.load_splits()
    fitm = split == "fit_mine"
    keepA, _ = L.clean_fit(pop["A"][fitm])
    names = pop["a_names"]
    kept_names = [names[j] for j in keepA]

    # are duplicate-name columns literally the same numbers, or independent scorings?
    ident, tot = 0, 0
    for n, idx in by_name.items():
        if len(idx) < 2:
            continue
        cols = [i for i, nm in enumerate(names) if nm == n]
        for a in range(len(cols)):
            for b in range(a + 1, len(cols)):
                x, y = pop["A"][:, cols[a]], pop["A"][:, cols[b]]
                m = ~(np.isnan(x) | np.isnan(y))
                tot += 1
                ident += int(np.allclose(x[m], y[m]))

    return {
        "n_delivered_criteria": len(rows),
        "n_distinct_names": int(len(by_name)),
        "n_distinct_descriptions": n_distinct_desc,
        "repeat_group_size_distribution": dict(sorted(Counter(sizes.tolist()).items())),
        "duplicate_column_pairs": tot,
        "duplicate_column_pairs_bit_identical": ident,
        "duplicate_columns_are_copies_not_rescorings": bool(tot > 0 and ident == tot),
        "A_columns_surviving_degeneracy_screen": int(len(keepA)),
        "distinct_concepts_among_surviving_columns": int(len(set(kept_names))),
        "effective_round0_bank_size": int(len(set(kept_names))),
        "chao1_on_exact_name_recapture": chao1(sizes),
        "chao1_on_source_aspect_clusters": chao1(aspects),
        "reading": (
            "the round-0 A bank is 154 delivered criteria but only "
            f"{len(by_name)} distinct concepts; after the column-degeneracy screen "
            f"{len(keepA)} columns survive carrying only {len(set(kept_names))} "
            "distinct concepts.  All duplicate-name column pairs are bit-identical "
            "copies, not independent re-scorings, so they add zero information and "
            "the nonlinear stack sees a far smaller bank than the headline number."
        ),
    }


def part_b():
    bank, mined = load_concepts()
    recs = bank + mined
    name_txt = [r["name"] for r in recs]
    full_txt = [f"{r['name']}. {r['text']}" for r in recs]

    E_name = embed(name_txt)
    E_full = embed(full_txt)

    is_mined = np.array([r["set"] == "mined56" for r in recs])
    rounds = np.array([r["round"] for r in recs])
    ids = [r["id"] for r in recs]

    out = {"n_bank154": int((~is_mined).sum()), "n_mined56": int(is_mined.sum()),
            "embedder": "BAAI/bge-large-en-v1.5 (CLS pooling, L2-normalised)",
            "variants": {}}

    for vname, E in (("name_only", E_name), ("name_plus_definition", E_full)):
        S = E @ E.T
        np.fill_diagonal(S, -1.0)
        Sm = S[np.ix_(is_mined, is_mined)]
        Smb = S[np.ix_(is_mined, ~is_mined)]
        Sb = S[np.ix_(~is_mined, ~is_mined)]
        mined_rounds = rounds[is_mined]
        iu = np.triu_indices(Sm.shape[0], k=1)
        cross_round = mined_rounds[iu[0]] != mined_rounds[iu[1]]

        v = {
            "similarity_distribution": {
                "mined_vs_mined": {"median": float(np.median(Sm[iu])),
                                    "p90": float(np.percentile(Sm[iu], 90)),
                                    "p99": float(np.percentile(Sm[iu], 99)),
                                    "max": float(Sm[iu].max())},
                "mined_vs_bank154": {"median": float(np.median(Smb)),
                                      "p90": float(np.percentile(Smb, 90)),
                                      "p99": float(np.percentile(Smb, 99)),
                                      "max": float(Smb.max())},
                "bank154_vs_bank154": {
                    "median": float(np.median(Sb[np.triu_indices(Sb.shape[0], k=1)])),
                    "p90": float(np.percentile(Sb[np.triu_indices(Sb.shape[0], k=1)], 90)),
                    "p99": float(np.percentile(Sb[np.triu_indices(Sb.shape[0], k=1)], 99)),
                    "max": float(Sb[np.triu_indices(Sb.shape[0], k=1)].max())},
            },
            "threshold_sweep": {},
        }
        for tau in THRESHOLDS:
            dup_pairs = int((Sm[iu] >= tau).sum())
            dup_pairs_cross = int(((Sm[iu] >= tau) & cross_round).sum())
            mined_hits_bank = (Smb.max(axis=1) >= tau)
            lab = single_linkage(Sm + np.eye(Sm.shape[0]) * -1, tau) if Sm.shape[0] else np.array([])
            sizes = np.bincount(lab)
            # union clustering (bank + mined) -> how many mined join a bank species
            Sall = S.copy()
            lab_all = single_linkage(Sall, tau)
            bank_species = set(lab_all[~is_mined].tolist())
            mined_in_bank_species = int(np.isin(lab_all[is_mined], list(bank_species)).sum())
            v["threshold_sweep"][f"{tau}"] = {
                "mined_mined_pairs_ge_tau": dup_pairs,
                "mined_mined_pairs_ge_tau_CROSS_ROUND": dup_pairs_cross,
                "mined_mined_pair_rate": dup_pairs / len(iu[0]),
                "n_mined_with_a_bank_match": int(mined_hits_bank.sum()),
                "frac_mined_with_a_bank_match": float(mined_hits_bank.mean()),
                "mined_species_single_linkage": int(len(sizes)),
                "chao1_on_mined": chao1(sizes),
                "n_mined_absorbed_into_a_bank154_species": mined_in_bank_species,
                "frac_mined_absorbed": mined_in_bank_species / int(is_mined.sum()),
            }
        # top near-duplicate pairs for eyeballing
        order = np.argsort(-Sm[iu])[:15]
        mined_ids = [i for i, m in zip(ids, is_mined) if m]
        v["top_mined_pairs"] = [
            {"a": mined_ids[iu[0][k]], "b": mined_ids[iu[1][k]], "cos": float(Sm[iu][k]),
             "a_name": name_txt[np.where(is_mined)[0][iu[0][k]]],
             "b_name": name_txt[np.where(is_mined)[0][iu[1][k]]]}
            for k in order]
        bank_ids = [i for i, m in zip(ids, is_mined) if not m]
        bo = np.dstack(np.unravel_index(np.argsort(-Smb, axis=None)[:15], Smb.shape))[0]
        v["top_mined_to_bank_pairs"] = [
            {"mined": mined_ids[a], "bank": bank_ids[b], "cos": float(Smb[a, b]),
             "mined_name": name_txt[np.where(is_mined)[0][a]],
             "bank_name": name_txt[np.where(~is_mined)[0][b]]}
            for a, b in bo]
        out["variants"][vname] = v

    # ---- threshold calibration from the pilot's planted probe pairs -----------
    # planted probes are lexical look-alikes of a real criterion but conceptually
    # distinct; a defensible duplicate threshold must sit ABOVE their similarity.
    probes = []
    r4 = {c["id"]: c for c in
          json.loads((HERE / "round4_proposals_blinded.json").read_text())["criteria"]}
    for a, b, why in (("P10", "P05", "planted 'bare mention of ablation' vs real "
                                     "'ablation establishes necessity'"),
                      ("P24", "P17", "planted notation probe vs its substantive pair")):
        if a in r4 and b in r4:
            ea = embed([r4[a]["name"], r4[b]["name"]])
            ef = embed([f"{r4[a]['name']}. {BOILER.sub('', r4[a]['instruction'])}",
                        f"{r4[b]['name']}. {BOILER.sub('', r4[b]['instruction'])}"])
            probes.append({"pair": f"r4:{a} vs r4:{b}", "why": why,
                           "a_name": r4[a]["name"], "b_name": r4[b]["name"],
                           "cos_name_only": float(ea[0] @ ea[1]),
                           "cos_name_plus_definition": float(ef[0] @ ef[1])})
    out["planted_probe_calibration"] = {
        "pairs": probes,
        "reading": "these pairs were designed to be lexically similar and "
                   "conceptually distinct; a duplicate threshold at or below their "
                   "cosine would call genuinely distinct concepts duplicates",
    }
    out["bank154_internal_recapture"] = bank_recapture_census()
    out["design_caveats"] = [
        "the pilot's proposer was shown the current bank each round and instructed "
        "not to duplicate it, so recapture among the 56 mined criteria is suppressed "
        "BY DESIGN; the mined-set Chao1 figures exercise the machinery rather than "
        "estimate concept richness",
        "the 154-criterion A bank is a GENERAL scientific-reporting rubric bank "
        "(CONSORT / PRISMA / STROBE / ARRIVE / TIDieR / CHEERS items) while the mined "
        "criteria are ML-abstract specific, so the near-zero mined-to-bank duplication "
        "rate is partly a domain mismatch and is NOT by itself evidence of proposer "
        "novelty",
        "bge cosine is compressed: the bank's own internal median is ~.58-.63, so an "
        "absolute threshold from another corpus does not transfer; the planted-probe "
        "pairs are used here to place a floor under any defensible threshold",
    ]
    return out


def main():
    res = {"analysis": "Good-Turing / missing-mass prototype for the Layer-3 closure pilot"}
    print("part (a) decay extrapolation ...", flush=True)
    res["part_a_decay_extrapolation"] = part_a()
    print(json.dumps({k: v for k, v in res["part_a_decay_extrapolation"].items()
                      if k != "criterion_value_series"}, indent=2)[:4000])
    print("part (b) redundancy census ...", flush=True)
    res["part_b_redundancy_census"] = part_b()
    for vn, v in res["part_b_redundancy_census"]["variants"].items():
        print(f"-- {vn}:", json.dumps(v["similarity_distribution"], indent=1))
        for tau, t in v["threshold_sweep"].items():
            print(f"   tau={tau}: mined-mined pairs={t['mined_mined_pairs_ge_tau']} "
                  f"(cross-round {t['mined_mined_pairs_ge_tau_CROSS_ROUND']}), "
                  f"mined w/ bank match={t['n_mined_with_a_bank_match']}, "
                  f"species={t['mined_species_single_linkage']}, "
                  f"chao1bc={t['chao1_on_mined']['chao1_bias_corrected']:.1f}, "
                  f"GT missing mass={t['chao1_on_mined']['good_turing_missing_mass']:.3f}")
    print(json.dumps(res["part_b_redundancy_census"]["planted_probe_calibration"], indent=1))
    (HERE / "missing_mass.json").write_text(json.dumps(res, indent=2))
    print("wrote", HERE / "missing_mass.json")


if __name__ == "__main__":
    main()
