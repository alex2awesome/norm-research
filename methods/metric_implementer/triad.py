"""CrowdTruth x G-theory triad layer: item difficulty, judge quality, words-vs-reader.

Lit grounding (design doc 8/9): CrowdTruth (Aroyo & Welty 2015; Dumitrache et al.
arXiv:1808.06080) models worker/unit/annotation quality as a mutually-weighted fixed
point; Generalizability Theory (Cronbach, Gleser, Nanda & Rajaratnam 1972) decomposes
score variance over crossed facets and is Cronbach's own successor to classical test
theory. Mapping onto our setting: judge config (model x pass) ~ worker, datapoint ~ unit,
metric ~ annotation task.

Three deliverables per (metric version x judge panel):

1. **UQS per item** -> the difficulty profile. Low-UQS items are where high-quality
   judges still split; they feed hard-item selection (counterfactual bases, oracle
   stratification, reconstruction shown-pairs) via ``stratified_order``.
2. **WQS per judge config** -> the anchor becomes a *measured* worker, never an assumed
   oracle. A panel where Sonnet's WQS collapses on some criterion disqualifies it as
   anchor for that criterion.
3. **Variance components** (item / judge / item x judge / pass residual) -> the
   ``words_share`` thickness index: thin criteria put variance in the item main effect
   (the words decide, every reader sees it), thick criteria in the item x judge
   interaction (the reader decides). The judge main effect is leniency/severity --
   calibratable, therefore NOT thickness.

Caveats carried from the lit: LLM judges violate CrowdTruth's worker-independence
assumption (shared weights within a family, shared pretraining across families) -- WQS is
meaningful *relative to the panel*, optimistic in absolute terms. Variance components use
a balanced-design approximation (items dropped unless every panel model scored them).

All selection here is in the CALIBRATE channel: no outcome labels are touched.
"""

from __future__ import annotations

import itertools
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .artifact import MetricArtifact
from .backends import LLMBackend
from .judges import PromptJudge

# --------------------------------------------------------------------------------------
# 1. Long table: the pooled score record every other function consumes
# --------------------------------------------------------------------------------------


def build_long_table(artifact: MetricArtifact, texts: List[str], ids: List[str],
                     panel: Dict[str, LLMBackend], cfg, passes: int = 3,
                     anchor: Optional[Tuple[str, LLMBackend]] = None,
                     version_id: Optional[str] = None,
                     n_data: Optional[int] = None,
                     run_ts: Optional[str] = None) -> pd.DataFrame:
    """Score every item with every panel judge `passes` times (plus the anchor once).

    Row schema — judge model × pass × (prompt, prompt-iteration) × item × score ×
    #datapoints: (judge, model, pass, metric_id, version_id, body_hash, n_data, run_ts,
    item_id, score). The prompt-identity and data-budget columns make every long CSV
    self-describing, so concatenating all triad runs for a metric yields the full prompt-
    evolution × data-scaling panel with no joins. Non-applicable / failed-parse items are
    NaN; ``judge`` is the panel short name, the anchor's is always ``"anchor"``.
    """
    assert artifact.kind == "prompt", "triad panels are for prompt judges"
    run_ts = run_ts or time.strftime("%Y-%m-%dT%H:%M:%S")
    meta = {"metric_id": artifact.metric_id, "version_id": version_id or "unversioned",
            "body_hash": artifact.body_hash,
            "n_data": len(texts) if n_data is None else n_data, "run_ts": run_ts}
    rows = []
    for name, backend in panel.items():
        pj = PromptJudge(backend, cfg)
        for p in range(passes):
            scores, _ = pj.score(artifact, texts)
            for iid, s in zip(ids, scores):
                rows.append({"judge": name, "model": backend.model, "pass": p,
                             "item_id": iid, "score": float(s), **meta})
    if anchor is not None:
        a_name, a_backend = anchor
        pj = PromptJudge(a_backend, cfg)
        scores, _ = pj.score(artifact, texts, temperature=0.3)
        for iid, s in zip(ids, scores):
            rows.append({"judge": "anchor", "model": a_backend.model, "pass": 0,
                         "item_id": iid, "score": float(s), **meta})
    return pd.DataFrame(rows)


def triad_history(registry, metric_id: str) -> pd.DataFrame:
    """The prompt-evolution × data-scaling panel: one row per triad run, joining each
    run's analysis (AQS, words_share, anchor agreement) to its version record's lineage
    (operator, parent, optimizer round) and budget (instruction tokens, data budget).
    Reads only the registry — no LLM calls. Tolerates pre-schema runs (missing fields
    come back NaN)."""
    import json as _json

    tdir = registry._mdir(metric_id) / "triad"
    rows = []
    for p in sorted(tdir.glob("*__triad.json")):
        d = _json.loads(p.read_text())
        vid = d.get("version_id")
        vrec = {}
        try:
            vrec = registry.get_version(metric_id, vid, d.get("kind", "prompt"))
        except Exception:
            pass
        anch = d.get("anchor_agreement") or {}
        best = {j: (e.get("overall") or {}).get("spearman") for j, e in anch.items()}
        hard = {j: (e.get("hard") or {}).get("spearman") for j, e in anch.items()}
        vbj = d.get("versions_by_judge") or {}
        rows.append({
            "run_ts": p.name.split("__")[1],
            "version_id": vid,
            "versions_by_judge": ",".join(f"{k}={v}" for k, v in vbj.items()) or None,
            "operator": (vrec.get("lineage") or {}).get("operator"),
            "parent": (vrec.get("lineage") or {}).get("parent_version"),
            "instruction_tokens": (vrec.get("budget") or {}).get("instruction_tokens"),
            "n_data": d.get("n_items"),
            "n_data_version": (vrec.get("budget") or {}).get("data_budget_n"),
            "panel": ",".join(m.split("/")[-1] for m in d.get("panel_models", [])),
            "passes": d.get("passes"),
            "aqs": (d.get("crowdtruth") or {}).get("aqs"),
            "words_share": (d.get("variance_components") or {}).get("words_share"),
            "noise_share": (d.get("variance_components") or {}).get("noise_share"),
            **{f"anchor_rho__{j}": v for j, v in best.items()},
            **{f"anchor_rho_hard__{j}": v for j, v in hard.items()},
            "cost_usd": d.get("cost_usd"),
        })
    return pd.DataFrame(rows).sort_values("run_ts").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# 2. CrowdTruth fixed point (adapted to scalar scores in [0,1])
# --------------------------------------------------------------------------------------


def crowdtruth(table: pd.DataFrame, max_iter: int = 100, tol: float = 1e-7) -> dict:
    """Mutually-weighted quality scores. Worker = one (judge, pass) configuration.

    agreement(a, b) = 1 - |a - b| for scores in [0,1].
    UQS(i) = WQS-weighted mean pairwise agreement among workers on item i.
    WQS(w) = UQS-weighted mean agreement between w and the WQS-weighted leave-one-out
             mean of the other workers (disagreeing on ambiguous items is not punished).
    """
    df = table.dropna(subset=["score"])
    workers = sorted(df.groupby(["judge", "pass"]).groups)
    items = sorted(df["item_id"].unique())
    widx = {w: k for k, w in enumerate(workers)}
    iidx = {i: k for k, i in enumerate(items)}
    W, I = len(workers), len(items)
    S = np.full((W, I), np.nan)
    for _, r in df.iterrows():
        S[widx[(r["judge"], r["pass"])], iidx[r["item_id"]]] = r["score"]
    M = ~np.isnan(S)
    S0 = np.nan_to_num(S)

    wqs = np.ones(W)
    uqs = np.ones(I)
    for _ in range(max_iter):
        # ---- UQS: weighted mean pairwise agreement per item -------------------------
        new_uqs = np.zeros(I)
        for i in range(I):
            num = den = 0.0
            ws = np.where(M[:, i])[0]
            for a, b in itertools.combinations(ws, 2):
                wt = wqs[a] * wqs[b]
                num += wt * (1.0 - abs(S0[a, i] - S0[b, i]))
                den += wt
            new_uqs[i] = num / den if den > 0 else np.nan
        # ---- WQS: UQS-weighted agreement with leave-one-out panel mean --------------
        tot = (wqs[:, None] * S0 * M).sum(axis=0)          # per item
        wsum = (wqs[:, None] * M).sum(axis=0)
        new_wqs = np.zeros(W)
        for w in range(W):
            num = den = 0.0
            for i in np.where(M[w])[0]:
                d = wsum[i] - wqs[w]
                if d <= 0 or np.isnan(new_uqs[i]):
                    continue
                loo = (tot[i] - wqs[w] * S0[w, i]) / d
                num += new_uqs[i] * (1.0 - abs(S0[w, i] - loo))
                den += new_uqs[i]
            new_wqs[w] = num / den if den > 0 else 0.0
        delta = np.nanmax(np.abs(new_wqs - wqs)) + np.nanmax(np.abs(new_uqs - uqs))
        wqs, uqs = new_wqs, new_uqs
        if delta < tol:
            break

    wqs_by_worker = {f"{j}#p{p}": float(wqs[widx[(j, p)]]) for j, p in workers}
    judges = sorted({j for j, _ in workers})
    wqs_by_judge = {j: float(np.mean([wqs[widx[(jj, p)]] for jj, p in workers if jj == j]))
                    for j in judges}
    return {
        "uqs": {items[i]: float(uqs[i]) for i in range(I)},
        "wqs_by_worker": wqs_by_worker,
        "wqs_by_judge": wqs_by_judge,
        "aqs": float(np.nanmean(uqs)),                     # annotation-task quality
    }


# --------------------------------------------------------------------------------------
# 3. G-study variance components + D-study (judge facet = model, passes = replicates)
# --------------------------------------------------------------------------------------


def variance_components(table: pd.DataFrame, exclude_judges: Tuple[str, ...] = ("anchor",)
                        ) -> dict:
    """Two-way crossed random-effects decomposition: item x judge with pass replicates.

    Balanced-design approximation: keeps only items scored by every panel judge; uses the
    harmonic-mean replicate count. Negative component estimates are clipped to 0 (standard
    G-theory practice). ``words_share`` = s2_item / (s2_item + s2_interaction): the
    fraction of construct-relevant variance decided by the words rather than the reader.
    """
    df = table[~table["judge"].isin(exclude_judges)].dropna(subset=["score"])
    judges = sorted(df["judge"].unique())
    cell = df.groupby(["item_id", "judge"])["score"]
    means = cell.mean().unstack()                          # items x judges
    counts = cell.count().unstack().reindex(means.index)
    full = means.dropna(axis=0)                            # items with all judges
    n_dropped = len(means) - len(full)
    ni, nj = full.shape
    if ni < 3 or nj < 2:
        return {"error": "not enough complete items/judges", "n_items": ni, "n_judges": nj}
    nr = float(1.0 / np.mean(1.0 / counts.loc[full.index].to_numpy().clip(min=1)))

    X = full.to_numpy()
    grand = X.mean()
    a = X.mean(axis=1) - grand                             # item effects
    b = X.mean(axis=0) - grand                             # judge effects
    E = X - X.mean(axis=1, keepdims=True) - X.mean(axis=0, keepdims=True) + grand

    ms_item = nj * nr * (a ** 2).sum() / (ni - 1)
    ms_judge = ni * nr * (b ** 2).sum() / (nj - 1)
    ms_int = nr * (E ** 2).sum() / ((ni - 1) * (nj - 1))
    within = df[df["item_id"].isin(full.index)].groupby(["item_id", "judge"])["score"] \
        .var(ddof=1).dropna()
    ms_res = float(within.mean()) if len(within) else 0.0  # pass-to-pass noise

    s2_res = max(0.0, ms_res)
    s2_int = max(0.0, (ms_int - s2_res) / nr)
    s2_item = max(0.0, (ms_item - ms_int) / (nj * nr))
    s2_judge = max(0.0, (ms_judge - ms_int) / (ni * nr))

    denom = s2_item + s2_int
    return {
        "n_items": ni, "n_judges": nj, "n_passes_harmonic": round(nr, 2),
        "n_items_dropped_incomplete": n_dropped,
        "sigma2_item": s2_item,                 # the words (every reader sees it)
        "sigma2_judge": s2_judge,               # leniency -- calibratable, not thickness
        "sigma2_interaction": s2_int,           # the reader (thickness lives here)
        "sigma2_residual": s2_res,              # pass noise
        "words_share": float(s2_item / denom) if denom > 0 else float("nan"),
        "noise_share": float(s2_res / (s2_item + s2_judge + s2_int + s2_res))
        if (s2_item + s2_judge + s2_int + s2_res) > 0 else float("nan"),
    }


def d_study(comp: dict, n_judges_grid=(1, 2, 3, 5), n_passes_grid=(1, 2, 3)) -> List[dict]:
    """Predicted generalizability Erho2 of a panel-mean score under alternative designs —
    the principled replacement for hand-picking reliability passes x items."""
    if "sigma2_item" not in comp:
        return []
    out = []
    for nj in n_judges_grid:
        for nr in n_passes_grid:
            den = comp["sigma2_item"] + comp["sigma2_interaction"] / nj + \
                comp["sigma2_residual"] / (nj * nr)
            out.append({"n_judges": nj, "n_passes": nr,
                        "E_rho2": float(comp["sigma2_item"] / den) if den > 0 else 1.0})
    return out


# --------------------------------------------------------------------------------------
# 4. Difficulty strata + selection policy (mixture, never easiest-first)
# --------------------------------------------------------------------------------------


def strata(uqs: Dict[str, float], n_strata: int = 3) -> Dict[str, str]:
    """Tercile labels by UQS: 'hard' = lowest-agreement third, then 'mid', 'easy'."""
    items = [k for k, v in uqs.items() if not np.isnan(v)]
    order = sorted(items, key=lambda k: uqs[k])
    labels = {}
    for rank, k in enumerate(order):
        tier = int(rank * n_strata / max(1, len(order)))
        labels[k] = ("hard", "mid", "easy")[min(tier, 2)]
    return labels


def stratified_order(ids: List[str], uqs: Dict[str, float], frac_hard: float = 0.6,
                     rng: Optional[np.random.Generator] = None) -> List[str]:
    """Reorder ids so head-of-list consumers (oracle subsample, CF bases, consistency,
    reconstruction shown-pairs) see a hard-oversampled MIXTURE: ~frac_hard from the hard
    stratum, the rest a random backbone. Easiest-first would inflate fidelity and hide
    the floor; hardest-only would bias every estimate. Items without a UQS go to the
    backbone."""
    rng = rng or np.random.default_rng(0)
    lab = strata(uqs)
    hard = [i for i in ids if lab.get(i) == "hard"]
    rest = [i for i in ids if lab.get(i) != "hard"]
    hard = list(rng.permutation(hard))
    rest = list(rng.permutation(rest))
    out = []
    while hard or rest:
        take_hard = hard and (not rest or rng.random() < frac_hard)
        out.append(hard.pop() if take_hard else rest.pop())
    return out


# --------------------------------------------------------------------------------------
# 5. Anchor agreement, difficulty-resolved (the frontier's per-stratum readout)
# --------------------------------------------------------------------------------------


def anchor_agreement(table: pd.DataFrame, item_strata: Dict[str, str]) -> dict:
    """Spearman + mean absolute agreement of each panel judge (pass-pooled mean) with the
    anchor, overall and per difficulty stratum. The hard-stratum number is the one that
    matters: easy-item agreement is where lawful-looking-but-wrong rubrics hide."""
    from scipy.stats import spearmanr

    df = table.dropna(subset=["score"])
    anchor = df[df["judge"] == "anchor"].set_index("item_id")["score"]
    if anchor.empty:
        return {}
    pooled = df[df["judge"] != "anchor"].groupby(["judge", "item_id"])["score"] \
        .mean().unstack(0)
    out = {}
    for j in pooled.columns:
        joined = pd.concat([pooled[j], anchor], axis=1, keys=["judge", "anchor"]).dropna()
        joined["stratum"] = [item_strata.get(i, "mid") for i in joined.index]
        ent = {}
        for name, sub in [("overall", joined)] + list(joined.groupby("stratum")):
            if len(sub) < 3:
                ent[name] = {"n": len(sub)}
                continue
            rho = spearmanr(sub["judge"], sub["anchor"]).statistic
            ent[name] = {"n": int(len(sub)),
                         "spearman": float(rho) if not np.isnan(rho) else None,
                         "mean_abs_agree": float(1 - (sub["judge"] - sub["anchor"])
                                                 .abs().mean())}
        out[j] = ent
    return out
