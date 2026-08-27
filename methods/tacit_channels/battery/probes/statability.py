"""Statability cluster: confidence dissociations + multi-method statability + self-report
stability."""
from __future__ import annotations

import numpy as np

from methods.tacit_channels.battery.artifacts import item_agreement
from methods.tacit_channels.battery.registry import ProbeSpec, register
from methods.tacit_channels.channels.common import spearman


def zero_corr_stats(confidence: np.ndarray, agreement: np.ndarray) -> dict:
    """Dienes statistics from per-item confidence + per-item agreement."""
    conf_acc_corr = spearman(confidence, agreement)
    q = np.quantile(confidence, 0.25)
    guess_mask = confidence <= q
    return {
        "mean_item_agreement": float(np.mean(agreement)),
        "conf_acc_corr": None if np.isnan(conf_acc_corr) else float(conf_acc_corr),
        "guess_quartile_agreement": float(np.mean(agreement[guess_mask]))
        if guess_mask.any() else None,
    }


def compute_zero_corr_v0(ctx) -> list[dict]:
    """v0 confidence = |p_yes - .5| (log-odds magnitude) — free on existing grids.

    Tacit signature (Dienes): mean_item_agreement above chance (0.5) while conf_acc_corr ~ 0
    (and guess-quartile agreement still above chance)."""
    rows = []
    tgt, _ = ctx.grid(ctx.target_job)
    for rung in ctx.executors():
        exe, _em = ctx.grid(rung)
        for cell in ctx.cells():
            target = ctx.target_name_vector(cell)
            forms = [v for (c, a, f), v in exe.items() if c == cell and a == "name"]
            if target is None or not forms:
                continue
            vec = np.mean(forms, axis=0)
            if vec.std() < 0.01:   # degenerate executor vector: ranks/agreement meaningless
                continue           # (audit 2026-07-23: 4/90 humor cells at 7B; flagged, skipped)
            stats = zero_corr_stats(np.abs(vec - 0.5), item_agreement(vec, target))
            base = {"construct": cell, "rung": rung, "domain": ctx.domain,
                    "probe": "P-STAT-1"}
            rows.extend({**base, "statistic": k, "value": v}
                        for k, v in stats.items() if v is not None)
    return rows


register(ProbeSpec(
    id="P-STAT-1",
    title="Zero-correlation + guessing criteria (continuous tacitness; v0 = log-odds conf)",
    cluster="statability", catalog_refs=("A3", "A4"),
    tacitness_direction="above-chance agreement with conf_acc_corr~0 = knowledge without "
                        "metacognitive access",
    requires=("target_grid", "exec_grid"), wave=0, cost_class="free",
    falsifier="confidence tracks accuracy everywhere agreement>chance -> no metacognitive "
              "dissociation in models",
    compute=compute_zero_corr_v0,
    notes="v1 (W1): verbalized confidence elicitation replaces the log-odds proxy",
))

register(ProbeSpec(
    id="P-STAT-2",
    title="Multi-method statability battery (free-text + forced-choice + recon-MCQ + assay)",
    cluster="statability", catalog_refs=("A1", "A5", "A6", "B26"),
    tacitness_direction="performance transfers while ALL elicitation methods fail to "
                        "produce a transferring statement",
    requires=("adapter_grid", "elicitation"), wave=1, cost_class="elicit",
    falsifier="any single elicitation method reliably recovers a transferring rule "
              "(Shanks: prior 'tacit' claims were weak-instrument artifacts)",
    compute=None, gates=("multi_method_statability",),
))

register(ProbeSpec(
    id="P-STAT-3",
    title="Self-report instability vs behavioral stability (+ justification gap, "
          "concrete-vs-abstract)",
    cluster="statability", catalog_refs=("B24", "B25", "C38"),
    tacitness_direction="stable judgments + unstable/contradictory self-reports = "
                        "non-propositional capacity (confabulation signature)",
    requires=("elicitation", "exec_grid"), wave=1, cost_class="elicit",
    falsifier="self-report stability tracks behavioral stability -> Background/Network "
              "distinction collapses here",
    compute=None,
))
