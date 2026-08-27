"""Frozen corpus-to-bank routing for the v3 silver rematch.

The old ``bge_pertask/<corpus>/catalog.txt`` files are deliberately not used.
Many of the 200-entry catalogs are orphaned placeholders and do not share the
metric universe used by the task-level bank/certificate.  Every corpus below is
routed to the current task hierarchy instead.
"""

from __future__ import annotations

from pathlib import Path


DEFAULT_HOME = Path("/lfs/skampere3/0/alexspan")
DEFAULT_DATA_ROOT = DEFAULT_HOME / "data" / "bge_pertask"
DEFAULT_HIERARCHY_ROOT = DEFAULT_HOME / "norm-research" / "outputs" / "hierarchy"
DEFAULT_OUTPUT_ROOT = DEFAULT_HOME / "data" / "silver_match_v3_20260712_faithful"


LEGACY_CORPUS_TO_TASK: dict[str, str] = {
    # Creative-writing feedback and comparative criticism.
    "creative_writing": "creative-writing",
    "litbench_rationales": "creative-writing",
    "wp_comments": "creative-writing",
    # Humor feedback.
    "humor": "humor",
    "humor_multi": "humor",
    # Press releases.
    "press_releases": "press-releases",
    # Code review. CRSE is the code-review Stack Exchange corpus.
    "code_review": "code-review",
    "crse": "code-review",
    # Mathematical answers, proof/code feedback, and editorials.
    "math": "math-stackexchange",
    "math_se": "math-stackexchange",
    "aops_forum": "math-stackexchange",
    "competition_editorials": "math-stackexchange",
    "mathlib": "math-stackexchange",
    # Scientific peer review.
    "peer_review": "peer-review",
    "pr_review_feedback": "peer-review",
    # Regulatory notice-and-comment.
    "notice_and_comment": "notice-and-comment",
    "nc_public_comments": "notice-and-comment",
    # Legal and administrative adjudication.
    "bva_opinions": "legal-outcome-prediction",
    "cavc_decisions": "legal-outcome-prediction",
    "courtlistener_opinions": "legal-outcome-prediction",
    "dol_arb": "legal-outcome-prediction",
    "law_se": "legal-outcome-prediction",
    "legaladvice_uk": "legal-outcome-prediction",
    "nlrb_decisions": "legal-outcome-prediction",
    "ptab_fwd": "legal-outcome-prediction",
    "reddit_supremecourt": "legal-outcome-prediction",
    "ttab_inter_partes": "legal-outcome-prediction",
}

# Backwards-compatible name used by the legacy-signal calibration manifest.
CORPUS_TO_TASK = LEGACY_CORPUS_TO_TASK


# Canonical, alias-free sources.  Each GEPA source is reconstructed by aligning
# the deploy record (full context/type/polarity) with its independent judge
# score (faithful/valid).  The old bge_pertask signals are samples or aliases
# of these sources and are not unioned into production.
CANONICAL_SOURCES: dict[str, dict] = {
    "aops_forum": dict(task="math-stackexchange", gepa="aops_forum", deploy="deploy_best_full.jsonl", score="score_best_full.jsonl"),
    "bva_opinions": dict(task="legal-outcome-prediction", gepa="bva_opinions", deploy="deploy_best_full.jsonl", score="score_best_full.jsonl"),
    "cavc_decisions": dict(task="legal-outcome-prediction", gepa="cavc_decisions", deploy="deploy_best_full.jsonl", score="score_best_full.jsonl"),
    "github_code_review": dict(task="code-review", gepa="code_review", deploy="deploy_round1_full.jsonl", score="score_round1_full.jsonl"),
    "competition_editorials": dict(task="math-stackexchange", gepa="competition_editorials", deploy="deploy_best_full.jsonl", score="score_best_full.jsonl"),
    "courtlistener_opinions": dict(task="legal-outcome-prediction", gepa="courtlistener_opinions", deploy="deploy_best_full.jsonl", score="score_best_full.jsonl"),
    "creative_writing": dict(task="creative-writing", gepa="creative_writing/wp_comments", deploy="deploy_round2_full.jsonl", score="score_round2_full.jsonl"),
    "crse": dict(task="code-review", gepa="crse", deploy="deploy_round1_full.jsonl", score="score_round1_full.jsonl"),
    "dol_arb": dict(task="legal-outcome-prediction", gepa="dol_arb", deploy="deploy_best_full.jsonl", score="score_best_full.jsonl"),
    "humor_multi": dict(task="humor", gepa="humor/standup_multi", deploy="deploy_round3_full.jsonl", score="score_round3_full.jsonl"),
    "law_se": dict(
        task="legal-outcome-prediction", gepa="law_se",
        deploy="deploy_best_full.jsonl.cap50000", score="score_best_full.jsonl",
        optional_segments=[dict(deploy="deploy_best_tail_after_50000.jsonl", score="score_best_tail_gemma.jsonl")],
        expected_unjudged_candidates=45048,
    ),
    "legaladvice_uk": dict(task="legal-outcome-prediction", gepa="legaladvice_uk", deploy="deploy_baseline_full.jsonl", score="score_baseline_full.jsonl"),
    "litbench_rationales": dict(task="creative-writing", gepa="litbench_rationales", deploy="deploy_baseline_full.jsonl", score="score_best_full.jsonl"),
    "math_se": dict(task="math-stackexchange", gepa="math_se", deploy="deploy_round3_full.jsonl", score="score_round3_full.jsonl"),
    "mathlib": dict(task="math-stackexchange", gepa="mathlib", deploy="deploy_best_full.jsonl", score="score_best_full.jsonl"),
    "nc_public_comments": dict(task="notice-and-comment", gepa="nc_public_comments", deploy="deploy_baseline_full.jsonl", score="score_best_full.jsonl"),
    "nlrb_decisions": dict(task="legal-outcome-prediction", gepa="nlrb_decisions", deploy="deploy_best_full.jsonl", score="score_best_full.jsonl"),
    "notice_and_comment": dict(task="notice-and-comment", gepa="notice_and_comment", deploy="deploy_best_full.jsonl", score="score_best_full.jsonl"),
    "press_releases": dict(task="press-releases", gepa="press_releases", deploy="deploy_round2_full.jsonl", score="score_round2_full.jsonl"),
    "ptab_fwd": dict(task="legal-outcome-prediction", gepa="ptab_fwd", deploy="deploy_best_full.jsonl", score="score_best_full.jsonl"),
    "reddit_supremecourt": dict(
        task="legal-outcome-prediction", gepa="reddit_supremecourt",
        deploy="deploy_best_full.jsonl.cap50000", score="score_best_full.jsonl",
        optional_segments=[dict(deploy="deploy_best_tail_after_50000.jsonl", score="score_best_tail_gemma.jsonl")],
        expected_unjudged_candidates=7857,
    ),
    "ttab_inter_partes": dict(task="legal-outcome-prediction", gepa="ttab_inter_partes", deploy="deploy_best_full.jsonl", score="score_best_full.jsonl"),
    "pr_review_feedback": dict(
        task="peer-review", type="review_feedback",
        signals="data/bge_pertask/pr_review_feedback/signals_pr_review_feedback.jsonl",
        reviews="outputs/review_norms/all_reviews.jsonl",
    ),
}


# Compatibility aliases are metadata only and never create duplicate rows.
CORPUS_ALIASES: dict[str, str] = {
    "code_review": "crse",
    "humor": "humor_multi",
    "math": "math_se",
    "peer_review": "pr_review_feedback",
    "wp_comments": "creative_writing",
}


TASK_TO_HIERARCHY: dict[str, str] = {
    task: f"{task}_general_r2_expanded.json"
    for task in sorted(set(LEGACY_CORPUS_TO_TASK.values()))
}


# Model snapshots already present on sk3.  Keeping paths explicit makes a run
# independent of mutable Hugging Face aliases.
BGE_ENCODER = (
    "/lfs/skampere3/0/shared_hf_cache/"
    "models--BAAI--bge-large-en-v1.5/snapshots/"
    "d4aa6901d3a41ba39fb536a557fa166f842b0e09"
)
BGE_RERANKER = (
    "/lfs/skampere3/0/shared_hf_cache/"
    "models--BAAI--bge-reranker-base/snapshots/"
    "2cfc18c9415c912f9d8155881c133215df768a70"
)
GEMMA4 = (
    "/lfs/skampere3/0/shared_hf_cache/"
    "models--google--gemma-4-31b-it/snapshots/"
    "3548789868c5356dbf307c98e6f609007b82b3eb"
)
