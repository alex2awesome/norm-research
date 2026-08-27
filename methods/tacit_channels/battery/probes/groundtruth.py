"""Ground-truth & discriminability cluster: ensemble reference, imitation game,
distributional recovery."""
from __future__ import annotations

import itertools

import numpy as np

from methods.tacit_channels.battery.registry import ProbeSpec, register
from methods.tacit_channels.channels.common import spearman


def compute_ensemble_reference(ctx) -> list[dict]:
    """Kripke/community criterion: (a) the target's internal form-consistency = how coherent
    the reference 'community' is; (b) executor agreement vs the ensemble mean vs the spread
    across single reference forms (single-form ground truth is philosophically insufficient —
    quantify how much it matters)."""
    rows = []
    tgt_raw = ctx.grid_raw_reps(ctx.target_job)
    tgt, _ = ctx.grid(ctx.target_job)
    for cell in ctx.cells():
        # every target name replicate (forms x reps)
        replicates = []
        for (c, a, f), vecs in tgt_raw.items():
            if c == cell and a == "name":
                replicates.extend(vecs)
        if len(replicates) < 2:
            continue
        pairs = [spearman(x, y) for x, y in itertools.combinations(replicates, 2)]
        pairs = [p for p in pairs if not np.isnan(p)]
        consistency = float(np.mean(pairs)) if pairs else None
        ensemble = np.mean(replicates, axis=0)
        base = {"construct": cell, "rung": ctx.target_job, "domain": ctx.domain,
                "probe": "P-GT-1"}
        if consistency is not None:
            rows.append({**base, "statistic": "target_form_consistency",
                         "value": consistency})
        for rung in ctx.executors():
            exe, _em = ctx.grid(rung)
            forms = [v for (c, a, f), v in exe.items() if c == cell and a == "name"]
            if not forms:
                continue
            vec = np.mean(forms, axis=0)
            rho_ens = spearman(vec, ensemble)
            singles = [spearman(vec, r) for r in replicates]
            singles = [s for s in singles if not np.isnan(s)]
            if np.isnan(rho_ens) or not singles:
                continue
            b2 = {"construct": cell, "rung": rung, "domain": ctx.domain, "probe": "P-GT-1"}
            rows.append({**b2, "statistic": "rho_vs_ensemble", "value": float(rho_ens)})
            rows.append({**b2, "statistic": "single_form_spread",
                         "value": float(max(singles) - min(singles))})
    return rows


def compute_distributional(ctx) -> list[dict]:
    """Hayek: does the executor recover population-LEVEL judgment structure (distribution
    shape), separately from item-level rank agreement?"""
    from scipy.stats import ks_2samp
    rows = []
    for rung in ctx.executors():
        exe, _em = ctx.grid(rung)
        for cell in ctx.cells():
            target = ctx.target_name_vector(cell)
            forms = [v for (c, a, f), v in exe.items() if c == cell and a == "name"]
            if target is None or not forms:
                continue
            vec = np.mean(forms, axis=0)
            ks = ks_2samp(vec, target).statistic
            qq = np.corrcoef(np.sort(vec), np.sort(target))[0, 1]
            base = {"construct": cell, "rung": rung, "domain": ctx.domain, "probe": "P-GT-3"}
            rows.append({**base, "statistic": "dist_ks", "value": float(ks)})
            if not np.isnan(qq):
                rows.append({**base, "statistic": "dist_qq_corr", "value": float(qq)})
    return rows


register(ProbeSpec(
    id="P-GT-1",
    title="Ensemble ground truth: target form-consistency + executor-vs-ensemble agreement",
    cluster="groundtruth", catalog_refs=("B18",),
    tacitness_direction="diagnostic (reference quality), feeds attenuation correction",
    requires=("target_grid", "exec_grid"), wave=0, cost_class="free",
    falsifier="single-form reference == ensemble reference everywhere -> community criterion "
              "adds nothing here",
    compute=compute_ensemble_reference,
    gates=("reliability_precondition",),
))

register(ProbeSpec(
    id="P-GT-2",
    title="Imitation Game: blinded judge distinguishes target vs trained executor",
    cluster="groundtruth", catalog_refs=("C30",),
    tacitness_direction="judge at chance = strong transfer criterion (beyond rho)",
    requires=("adapter_grid", "judge_batch"), wave=2, cost_class="judge",
    falsifier="judge beats chance easily even at high rho -> rho overstates transfer",
    compute=None, gates=("anchors", "experimenters_regress"),
))

register(ProbeSpec(
    id="P-GT-3",
    title="Distributional recovery (population-level shape vs item-level rank)",
    cluster="groundtruth", catalog_refs=("B23",),
    tacitness_direction="channels matching on rho but differing on distribution shape = "
                        "dispersed-knowledge signature",
    requires=("target_grid", "exec_grid"), wave=0, cost_class="free",
    falsifier="explicit channel matches reward channel on all distributional statistics",
    compute=compute_distributional,
))
