"""Generalization cluster: OOD/typicality decay, surface swap, scrambled, situational shift,
transfer narrowing."""
from __future__ import annotations

import numpy as np

from methods.tacit_channels.battery.artifacts import item_agreement
from methods.tacit_channels.battery.registry import ProbeSpec, register
from methods.tacit_channels.channels.common import spearman


def typicality_decay_stats(typicality: np.ndarray, agreement: np.ndarray) -> dict:
    slope = spearman(typicality, agreement)
    hi, lo = np.quantile(typicality, (2 / 3, 1 / 3))
    top = agreement[typicality >= hi]
    bot = agreement[typicality <= lo]
    return {
        "typicality_agreement_corr": None if np.isnan(slope) else float(slope),
        "typical_minus_atypical_agreement":
            float(np.mean(top) - np.mean(bot)) if len(top) and len(bot) else None,
    }


def compute_ood_decay(ctx) -> list[dict]:
    """Wittgenstein deviant-continuation / Bourdieu generativity, v0 on the name channel:
    does executor-target agreement decay off the typical core? (Channel-contrast version
    lands in W1 when articulation/adapter rows share the pass.)"""
    rows = []
    typ = ctx.item_typicality()
    for rung in ctx.executors():
        exe, _em = ctx.grid(rung)
        for cell in ctx.cells():
            target = ctx.target_name_vector(cell)
            forms = [v for (c, a, f), v in exe.items() if c == cell and a == "name"]
            if target is None or not forms:
                continue
            vec = np.mean(forms, axis=0)
            if len(typ) != len(vec) or vec.std() < 0.01:  # degenerate-vec guard (audit 07-23)
                continue
            stats = typicality_decay_stats(typ, item_agreement(vec, target))
            base = {"construct": cell, "rung": rung, "domain": ctx.domain,
                    "probe": "P-GEN-1"}
            rows.extend({**base, "statistic": k, "value": v}
                        for k, v in stats.items() if v is not None)
    return rows


register(ProbeSpec(
    id="P-GEN-1",
    title="OOD/typicality decay (deviant continuation; generativity)",
    cluster="generalization", catalog_refs=("B17", "C34"),
    tacitness_direction="channel-contrast at the boundary: explicit-rule channels diverge on "
                        "atypical items; installed/tacit channels persist",
    requires=("target_grid", "exec_grid", "item_embeddings"), wave=0, cost_class="free",
    falsifier="channel gaps uniform across typicality strata -> no rule-boundary effect "
              "(also answers Turner if agreement survives OOD)",
    compute=compute_ood_decay,
))

register(ProbeSpec(
    id="P-GEN-2",
    title="Surface-vocabulary swap (structure survives reformat)",
    cluster="generalization", catalog_refs=("A2",),
    tacitness_direction="rank-agreement retention across surface swap = abstraction, "
                        "not token memorization",
    requires=("transformed_items", "exec_grid", "target_grid"), wave=2, cost_class="build",
    falsifier="retention no better than fragment-statistics baseline (Perruchet critique)",
    compute=None, gates=("anchors",),
    notes="COMMITTED: GLM generation pass -> anchored content-preservation verification -> "
          "scoring rows; target's own retention = structure-preservation reference",
))

register(ProbeSpec(
    id="P-GEN-3",
    title="Structured-vs-scrambled interaction (Chase & Simon)",
    cluster="generalization", catalog_refs=("A11",),
    tacitness_direction="transfer sensitivity collapses on scrambled items = genuine "
                        "structural pattern use",
    requires=("transformed_items", "exec_grid", "target_grid"), wave=2, cost_class="build",
    falsifier="equal sensitivity on scrambled = surface-token keying (or construct is "
              "surface-keyed -- itself reportable)",
    compute=None,
    notes="COMMITTED: code-only sentence/clause shuffle; validity check = target's own "
          "signal collapses on scrambled",
))

register(ProbeSpec(
    id="P-GEN-4",
    title="Situational shift (newsworthiness case): install on region A of item space, "
          "evaluate on shifted region",
    cluster="generalization", catalog_refs=(),
    tacitness_direction="installed-policy robustness across covariate shift",
    requires=("adapter_grid", "item_embeddings"), wave=1, cost_class="score-rows",
    falsifier="installed policies never survive shift -> TK-local only",
    compute=None,
    notes="topic/style strata from item embeddings replace the iid split",
))

register(ProbeSpec(
    id="P-GEN-5",
    title="Transfer narrowing across training checkpoints (identical elements)",
    cluster="generalization", catalog_refs=("A9",),
    tacitness_direction="far-transfer plateaus/declines with training while near-transfer "
                        "rises = compilation trade-off",
    requires=("checkpoints", "exec_grid"), wave=3, cost_class="train",
    falsifier="breadth grows monotonically with training (naive more-data-more-general wins)",
    compute=None, gates=("item_disjoint", "controls_required"),
))
