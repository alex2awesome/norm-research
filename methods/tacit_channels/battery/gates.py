"""Validity disciplines encoded as GATES — applied to probe verdicts, not run as experiments.

Each gate: {description, catalog_refs, check(ctx, rows) -> list of violation strings}.
Gate violations never delete data; they STAMP rows (gate_flags) and block confirmatory
verdict language in the report layer.
"""
from __future__ import annotations

GATES = {
    "anchors": {
        "description": "every LLM-judge batch carries blinded known-label anchors and must "
                       "separate them before ingestion",
        "catalog_refs": (),
    },
    "multi_method_statability": {
        "description": "no 'unstatable' verdict from a single elicitation method; requires "
                       "free-text + forced-choice + recon-MCQ agreement",
        "catalog_refs": ("A5", "A6"),  # Shanks&StJohn; SRT-without-recognition subsumed
    },
    "g_control": {
        "description": "training-probe interpretations require an unrelated-task control "
                       "showing gains are domain-specific, not general capability",
        "catalog_refs": ("C37",),
    },
    "turner_phrasing": {
        "description": "verdicts phrased as behavior-correlation + generalization profile; "
                       "never 'policy-object transferred'",
        "catalog_refs": ("C35",),
    },
    "experimenters_regress": {
        "description": "judge-based instruments (imitation game etc.) carry the circularity "
                       "caveat: judge competence is not independently certified",
        "catalog_refs": ("C41",),
    },
    "reliability_precondition": {
        "description": "cross-construct slope claims require per-construct noise ceilings "
                       "(test-retest reps/forms) — attenuation-corrected twin reported",
        "catalog_refs": (),
    },
    "item_disjoint": {
        "description": "within-domain transfer stats valid only under train/eval item "
                       "disjointness (stable-hash halves, salt exp_gtk1)",
        "catalog_refs": (),
    },
    "tier_stamp": {
        "description": "every 'tacit' verdict carries its certification tier (0 bank-search / "
                       "1 optimized-search / 1.9 subspace-cap / 2 certified cap)",
        "catalog_refs": (),
    },
    "controls_required": {
        "description": "training-derived stats interpreted only as double differences vs "
                       "shuffled AND construct-permuted adapters",
        "catalog_refs": (),
    },
    "acceptance_freshness": {
        "description": "any scoring pass through the LoRA fork requires a passing zero-adapter "
                       "acceptance test on the current vLLM stack",
        "catalog_refs": (),
    },
}
