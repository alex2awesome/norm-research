"""E7 pilot analysis: bounded-articulability brackets from the registry.

Per (task x metric x judge tier) assemble the sandwich

    F_lo  <=  articulability(m; tier)  <=  F_hi

  F_lo  best achieved agreement-with-anchor at that tier:
        primary   = triad anchor_agreement (judge scores with its per-tier HEAD prompt
                    vs. Sonnet anchor on the canonical seed) — observed, conservative;
        secondary = best in-loop oracle_agreement.spearman across the scaling grid.
  F_hi  disattenuation ceiling sqrt(rho_tier) where rho_tier = the tier's pass-pass
        test-retest reliability from the triad long table. PILOT SIMPLIFICATION: anchor
        reliability is assumed ~1 (single pass at temp 0.3); the full E7 estimates it.

Plus the 2x2 words x reader interaction per focal metric from the scaling grid:
    I = [F(L_hi, strong) - F(L_lo, strong)] - [F(L_hi, weak) - F(L_lo, weak)]
which is the lack-of-fit of the additive (Chinchilla-null) model on a 2x2 grid —
the pilot estimate of thickness (formalization S6 / T7).

Pure read over outputs/; no LLM calls, zero spend.
Run:  python -m methods.metric_implementer.e7_brackets
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ImplementerConfig, apply_task_preset
from .registry import Registry

TASKS = ["law", "creative-writing", "code-review"]


def _pass_retest(long_csv: Path) -> dict:
    """Per-judge pass-pass Spearman + a score-collapse flag. A judge emitting <=2 distinct
    scores with ~0 spread has a spuriously perfect retest rho (the 06-12 pilot's gemma on
    code/edge: only {0.6,0.7}, std 0.045, rho=1.0) — that is collapse, not reliability."""
    df = pd.read_csv(long_csv)
    out = {}
    for judge, g in df[df.judge != "anchor"].groupby("judge"):
        wide = g.pivot_table(index="item_id", columns="pass", values="score")
        if wide.shape[1] < 2:
            continue
        cols = list(wide.columns)
        rhos = [wide[a].corr(wide[b], method="spearman")
                for i, a in enumerate(cols) for b in cols[i + 1:]]
        sc = g["score"].dropna()
        out[judge] = {
            "rho": float(np.nanmean(rhos)) if rhos else float("nan"),
            "n_distinct": int(sc.round(3).nunique()),
            "std": float(sc.std()),
            "n_items": int(wide.shape[0]), "n_passes": int(wide.shape[1]),
            "collapsed": bool(sc.round(3).nunique() <= 2 or sc.std() < 0.06),
        }
    return out


def _latest_triad(registry: Registry, metric_id: str):
    """Filename is ``{version_id}__{stamp}__triad.json`` — sort by the STAMP, not the
    whole name (version_id prefix would rank v008 after v000 regardless of date, picking
    a stale panel; that contaminated the cross-domain words_share comparison 2026-06-12)."""
    tdir = registry.root / "metrics" / metric_id / "triad"
    if not tdir.exists():
        return None, None
    js = sorted(tdir.glob("*__triad.json"), key=lambda p: p.name.split("__")[1])
    if not js:
        return None, None
    payload = json.loads(js[-1].read_text())
    long_csv = Path(str(js[-1]).replace("__triad.json", "__long.csv"))
    return payload, (long_csv if long_csv.exists() else None)


def _grid_cells(cfg, registry: Registry) -> pd.DataFrame:
    """Scaling CSV cells joined to each run's best in-loop scorecard."""
    runs = sorted(Path(cfg.runs_dir()).glob("scaling_*.csv"))
    if not runs:
        return pd.DataFrame()
    grid = pd.concat([pd.read_csv(p) for p in runs], ignore_index=True)
    grid = grid.drop_duplicates(subset="run_id", keep="last")
    best_oracle, best_rho = [], []
    for _, row in grid.iterrows():
        cards = [c for c in registry.scorecards(row["metric"])
                 if c.get("run_id") == row["run_id"]]
        promo = [c for c in cards if c.get("promotable")] or cards
        if promo:
            top = max(promo, key=lambda c: np.nan_to_num(c.get("fidelity_scalar"), nan=-1))
            best_oracle.append((top.get("oracle_agreement") or {}).get("spearman"))
            best_rho.append((top.get("reliability") or {}).get("rho"))
        else:
            best_oracle.append(np.nan)
            best_rho.append(np.nan)
    grid["best_oracle_spearman"] = pd.to_numeric(best_oracle, errors="coerce")
    grid["best_inloop_rho"] = pd.to_numeric(best_rho, errors="coerce")
    grid["tier"] = grid["judge_model"].str.split("/").str[-1]
    # the frontier at a budget is the BEST articulation <= that budget; the seed itself is
    # an articulation, so the frontier never drops below it (a 1-round optimizer can return
    # worse, but max-over-<=B is monotone by construction — formalization T1).
    grid["frontier"] = grid[["frontier_fidelity", "seed_fidelity"]].max(axis=1)
    return grid


# capability order: gemma-3-4b (weak) < llama-3.1-8b (strong). Alpha-sort would mis-rank
# ('g' < 'm' happens to hold here, but pin it explicitly so adding tiers can't silently flip).
_TIER_RANK = {"gemma-3-4b-it": 0, "llama-3.2-1b-instruct": -1,
              "llama-3.1-8b-instruct": 1, "llama-3.3-70b-instruct": 2}


def _interaction(grid: pd.DataFrame, metric: str, col: str) -> dict:
    g = grid[(grid.metric == metric) & (grid.kind == "prompt")]
    caps = sorted(g.token_cap.unique())
    tiers = sorted(g.tier.unique(), key=lambda t: _TIER_RANK.get(t, 99))
    if len(caps) < 2 or len(tiers) < 2:
        return {}
    weak, strong = tiers[0], tiers[-1]
    cell = {}
    for t in (weak, strong):
        for c in (caps[0], caps[-1]):
            v = g[(g.tier == t) & (g.token_cap == c)][col]
            cell[(t, c)] = float(v.max()) if len(v) else np.nan
    words_weak = cell[(weak, caps[-1])] - cell[(weak, caps[0])]
    words_strong = cell[(strong, caps[-1])] - cell[(strong, caps[0])]
    reader_lo = cell[(strong, caps[0])] - cell[(weak, caps[0])]
    reader_hi = cell[(strong, caps[-1])] - cell[(weak, caps[-1])]
    return {"weak_tier": weak, "strong_tier": strong,
            "caps": [int(caps[0]), int(caps[-1])],
            "cells": {f"{t}@{c}tok": round(v, 3) for (t, c), v in cell.items()},
            "words_effect_weak": round(words_weak, 3),
            "words_effect_strong": round(words_strong, 3),
            "reader_effect_loL": round(reader_lo, 3),
            "reader_effect_hiL": round(reader_hi, 3),
            "interaction": round(words_strong - words_weak, 3)}


def main() -> int:
    report = {"brackets": [], "interactions": []}
    for task in TASKS:
        cfg = ImplementerConfig()
        apply_task_preset(cfg, task)
        registry = Registry(cfg.registry_dir())
        grid = _grid_cells(cfg, registry)
        mdir = registry.root / "metrics"
        metrics = sorted(p.name for p in mdir.iterdir() if p.is_dir()) \
            if mdir.exists() else []
        for metric in metrics:
            payload, long_csv = _latest_triad(registry, metric)
            retest = _pass_retest(long_csv) if long_csv is not None else {}
            anchor_agree = (payload or {}).get("anchor_agreement") or {}
            words_share = ((payload or {}).get("variance_components") or {}) \
                .get("words_share")
            tiers = set(retest) | set(anchor_agree)
            if grid is not None and len(grid):
                tiers |= set(grid[grid.metric == metric].tier.unique())
            for tier in sorted(tiers):
                rt = retest.get(tier) or {}
                rho = rt.get("rho", np.nan)
                collapsed = bool(rt.get("collapsed"))
                f_hi = float(np.sqrt(rho)) if np.isfinite(rho) and rho > 0 else np.nan
                ov = (anchor_agree.get(tier) or {}).get("overall") \
                    if isinstance(anchor_agree.get(tier), dict) else None
                f_lo_triad = ov.get("spearman") if isinstance(ov, dict) else None
                f_lo_triad = float(f_lo_triad) if isinstance(f_lo_triad, (int, float)) \
                    and f_lo_triad is not None and np.isfinite(f_lo_triad) else np.nan
                sub = grid[(grid.metric == metric) & (grid.tier == tier)] \
                    if len(grid) else pd.DataFrame()
                f_lo_grid = float(sub.best_oracle_spearman.max()) if len(sub) else np.nan
                # F_lo = best construct-tracking (rank-agreement with the strong reference)
                # achieved at this tier, by triad anchor-agreement or grid oracle-agreement.
                f_lo = np.nanmax([f_lo_triad, f_lo_grid])
                # credibility flags (verification 2026-06-12): brackets are pilot-scale and
                # several endpoints are artifacts — never present a flagged bracket as a clean
                # measurement. F_hi from a collapsed judge is spurious; F_hi from rho stat-
                # indistinguishable-from-0 is uninformative; an inverted bracket whose ends are
                # both flagged is "data silent here", not a measured contradiction.
                flags = []
                if collapsed:
                    flags.append("Fhi_from_collapsed_judge")
                if np.isfinite(rho) and rt.get("n_items", 0) and rho < 0.45 \
                        and rt.get("n_passes", 0) <= 2:
                    flags.append("Fhi_rho_indistinguishable_from_0")
                if np.isfinite(f_lo) and f_lo_grid >= f_lo_triad and np.isfinite(f_lo_grid):
                    # grid oracle is a max over tiny applicable subsets (winner's curse)
                    flags.append("Flo_is_max_over_small_oracle_n")
                if np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo > f_hi:
                    flags.append("INVERTED_bracket_both_ends_suspect"
                                 if flags else "INVERTED_bracket")
                report["brackets"].append({
                    "task": task, "metric": metric, "tier": tier,
                    "F_lo": None if not np.isfinite(f_lo) else round(float(f_lo), 3),
                    "F_hi": None if not np.isfinite(f_hi) else round(f_hi, 3),
                    "rho_tier": None if not np.isfinite(rho) else round(rho, 3),
                    "F_lo_triad": None if not np.isfinite(f_lo_triad) else round(f_lo_triad, 3),
                    "F_lo_grid": None if not np.isfinite(f_lo_grid) else round(f_lo_grid, 3),
                    "words_share": words_share,
                    "words_share_floored": bool(words_share == 0.0),
                    "credibility_flags": flags or ["ok_pilot_scale"],
                })
            if len(grid) and metric in set(grid.metric):
                for col in ("frontier", "best_oracle_spearman"):
                    ia = _interaction(grid, metric, col)
                    if ia:
                        report["interactions"].append(
                            {"task": task, "metric": metric, "y": col, **ia})

    out = Path("outputs/metric_implementer/e7_pilot_brackets.json")
    out.write_text(json.dumps(report, indent=2, default=str))
    bdf = pd.DataFrame(report["brackets"])
    if len(bdf):
        bdf["flags"] = bdf["credibility_flags"].apply(
            lambda fs: "" if fs == ["ok_pilot_scale"] else ";".join(
                f.replace("_", " ")[:22] for f in fs))
        print("\n== Brackets: F_lo <= articulability(m; tier) <= F_hi  (PILOT, n~24) ==")
        print(bdf[["task", "metric", "tier", "F_lo", "F_hi", "rho_tier",
                   "words_share", "flags"]].to_string(index=False))
    for ia in report["interactions"]:
        print(f"\n== 2x2 interaction ({ia['task']}/{ia['metric']}, y={ia['y']}) ==")
        print(f"  cells {ia['cells']}")
        print(f"  words effect weak {ia['words_effect_weak']} | strong "
              f"{ia['words_effect_strong']} | interaction I = {ia['interaction']}")
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
