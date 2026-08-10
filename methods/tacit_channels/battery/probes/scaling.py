"""Scaling cluster: OSL asymptote classification, Dreyfus two-curve, differentiation floor.
All W0 — computed over every executor rung present for the (family, domain)."""
from __future__ import annotations

import numpy as np

from methods.tacit_channels.battery.registry import ProbeSpec, register
from methods.tacit_channels.channels.common import _rankdata, cell_stats

GAIN_ON, GAIN_OFF, GAP_MIN = 0.10, 0.05, 0.10


def classify_scaling(gains: list[float], gaps: list[float]) -> str:
    max_gain, max_gap = max(gains), max(gaps)
    if max_gap <= GAP_MIN:
        return "no_gap_anywhere"
    if max_gain >= GAIN_ON:
        return "eventually_articulable"
    if max_gain < GAIN_OFF:
        return "scaling_tacit"
    return "intermediate"


def compute_osl_asymptote(ctx) -> list[dict]:
    """Scaling-tacit(C) <=> articulation-gain ~ 0 at EVERY measured rung while the native gap
    persists. De-relativized: the pair is replaced by the capability axis."""
    tgt, _ = ctx.grid(ctx.target_job)
    per_rung = {r: dict() for r in ctx.executors()}
    for rung in ctx.executors():
        exe, em = ctx.grid(rung)
        for cell in ctx.cells():
            s = cell_stats(tgt, exe, em, cell)
            if s and s["gain"] is not None:
                per_rung[rung][cell] = s
    rows = []
    for cell in ctx.cells():
        gains = [per_rung[r][cell]["gain"] for r in per_rung if cell in per_rung[r]]
        gaps = [per_rung[r][cell]["gap"] for r in per_rung if cell in per_rung[r]]
        if len(gains) < 2:
            continue
        label = classify_scaling(gains, gaps)
        base = {"construct": cell, "rung": "_LADDER_", "domain": ctx.domain,
                "probe": "P-SCAL-1"}
        rows.append({**base, "statistic": "scaling_tacit",
                     "value": float(label == "scaling_tacit"), "label": label,
                     "n_rungs": len(gains)})
        rows.append({**base, "statistic": "max_articulation_gain",
                     "value": float(max(gains)), "label": label})
        rows.append({**base, "statistic": "max_native_gap",
                     "value": float(max(gaps)), "label": label})
    return rows


def _rung_size(job: str) -> float:
    """Nominal B-params from the job name (…_3b_/…_14b_…); orders rungs for slopes."""
    import re
    m = re.search(r"_(\d+)b_", job)
    return float(m.group(1)) if m else float("nan")


def compute_two_curve(ctx) -> list[dict]:
    """Dreyfus v0: per (cell, rung), name-based agreement minus best-articulation-based
    agreement — plus the per-construct SLOPE of that divergence across rungs (the Dreyfus
    claim proper: judgments outrun stated rules AS CAPABILITY GROWS). The slope, not the
    level, is the primary statistic — the level is gain-derived and duplicates P-CHAN-core.
    (v0 deviation noted: articulations are bank-mined, not the target's self-stated rubric.)"""
    tgt, _ = ctx.grid(ctx.target_job)
    rows = []
    per_cell: dict = {}
    for rung in ctx.executors():
        exe, em = ctx.grid(rung)
        size = _rung_size(rung)
        for cell in ctx.cells():
            s = cell_stats(tgt, exe, em, cell)
            if s is None or s["exec_name_rho"] is None or s["best_rho"] is None:
                continue
            div = float(s["exec_name_rho"] - s["best_rho"])
            rows.append({"construct": cell, "rung": rung, "domain": ctx.domain,
                         "probe": "P-SCAL-2", "statistic": "name_minus_articulation_rho",
                         "value": div})
            if not np.isnan(size):
                per_cell.setdefault(cell, []).append((size, div))
    from methods.tacit_channels.channels.common import spearman as _sp
    for cell, pairs in per_cell.items():
        if len(pairs) < 3:
            continue
        sizes, divs = zip(*sorted(pairs))
        slope = _sp(np.array(sizes), np.array(divs))
        if not np.isnan(slope):
            rows.append({"construct": cell, "rung": "_LADDER_", "domain": ctx.domain,
                         "probe": "P-SCAL-2", "statistic": "divergence_slope",
                         "value": float(slope), "n_rungs": len(pairs)})
    return rows


def compute_differentiation(ctx) -> list[dict]:
    """PC1 share of the construct x construct policy-correlation matrix, per rung (+target).
    High PC1 in a weak model = UNDIFFERENTIATION, not general competence."""
    rows = []
    for job in [ctx.target_job] + ctx.executors():
        vecs, _ = ctx.grid(job)
        mats = []
        for cell in ctx.cells():
            forms = [v for (c, a, f), v in vecs.items() if c == cell and a == "name"]
            if forms:
                mats.append(np.mean(forms, axis=0))
        if len(mats) < 10:
            continue
        R = np.vstack([_rankdata(m) for m in mats])
        C = np.nan_to_num(np.corrcoef(R), nan=0.0)
        np.fill_diagonal(C, 1.0)
        ev = np.sort(np.linalg.eigvalsh(C))[::-1]
        rows.append({"construct": "_AGGREGATE_", "rung": job, "domain": ctx.domain,
                     "probe": "P-SCAL-3", "statistic": "policy_pc1_share",
                     "value": float(ev[0] / len(C)), "n_constructs": len(mats)})
    return rows


register(ProbeSpec(
    id="P-SCAL-1",
    title="OSL scaling-asymptote classification (de-relativized tacitness)",
    cluster="scaling", catalog_refs=(),
    tacitness_direction="scaling_tacit=1: no measured capability can be TOLD this, though "
                        "every capability has the deficit",
    requires=("target_grid", "exec_grid"), wave=0, cost_class="free",
    falsifier="every construct becomes articulable at some rung -> tacitness was always "
              "just under-scaling",
    compute=compute_osl_asymptote, gates=("tier_stamp",),
))

register(ProbeSpec(
    id="P-SCAL-2",
    title="Dreyfus two-curve divergence (judgments vs stated-rubric predictions, across scale)",
    cluster="scaling", catalog_refs=("B21",),
    tacitness_direction="name-minus-articulation agreement GROWS with capability = expert "
                        "intuition outruns rules",
    requires=("target_grid", "exec_grid"), wave=0, cost_class="free",
    falsifier="the two curves track together at all scales",
    compute=compute_two_curve,
))

register(ProbeSpec(
    id="P-SCAL-3",
    title="Differentiation floor (policy PC1 share per rung)",
    cluster="scaling", catalog_refs=(),
    tacitness_direction="capability floor = differentiation floor (no policy-slot for an "
                        "articulation to configure below it)",
    requires=("target_grid", "exec_grid"), wave=0, cost_class="free",
    falsifier="rescue unlock does NOT co-occur with differentiation reaching target level "
              "(prereg the correlation before claiming)",
    compute=compute_differentiation,
))
