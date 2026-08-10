"""InfillConfig: hyperparameters for the gap-detecting tree + feature-infilling loop."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class InfillConfig:
    """Configuration for ``methods/metrics_tree_infilling``.

    Grouped by spec section. Defaults are chosen to match R ``partykit::glmtree`` /
    ``strucchange`` conventions where a counterpart exists (trim, alpha), and to be
    conservative elsewhere.
    """

    # ---- Honest split (spec §1) ---------------------------------------------------
    discover_fraction: float = 0.7      # 2-way: items_discover vs items_test
    # 3-way discover/select(guard)/confirm(test): the guard split drives gap-flagging + keep/drop;
    # the test split is touched ONLY to materialize features for the final, untouched power read.
    # discover = 1 - guard_fraction - test_fraction.
    guard_fraction: float = 0.14
    test_fraction: float = 0.30
    group_split_by_id: bool = False     # keep all rows sharing id_column in one split (no pair leak)
    random_seed: int = 42

    # ---- MOB instability test (spec §2, mob/mfluctuation.py) -----------------------
    alpha: float = 0.05                 # split iff Bonferroni-adjusted p < alpha
    mfluct_trim: float = 0.1            # sup-LM trimming window [trim, 1-trim] (R default)
    n_permutations: int = 999           # permutation-null replicates for the p-value
    bonferroni: bool = True             # adjust across candidate z's
    cov_estimator: str = "outer"        # "outer" (partykit default) | "info" (Fisher info)

    # ---- Tree growth (spec §2, mob/glmtree.py) ------------------------------------
    min_node_size: int = 20             # minimum items in a node to keep splitting / per child
    max_depth: Optional[int] = None     # None = grow until no node is unstable
    n_cutpoint_candidates: int = 64     # max distinct cutpoints scanned per numeric z (0=all)

    # ---- Gap flagging (spec §2.1, gaps.py) ----------------------------------------
    # A terminal node is a gap if its held-out fit is poor on EITHER criterion below.
    gap_deviance_per_item: float = 1.20   # mean held-out deviance/item above this => poor
    gap_auc_threshold: float = 0.55       # held-out AUC at/below this => near chance
    gap_min_test_items: int = 15          # need at least this many test items to judge a node

    # ---- Residualized contrast (spec §3, contrast.py) -----------------------------
    contrastive_pairs_k: int = 5          # k (pos,neg) pairs sampled from WRONG for the LLM
    wrong_resid_quantile: float = 0.66    # |resid| >= this quantile => WRONG
    right_resid_quantile: float = 0.33    # |resid| <= this quantile => RIGHT
    contrast_max_chars: int = 4000        # truncate each item text shown to the proposer
    proposer_max_examples: int = 6        # max pos/neg WRONG examples shown to the proposer

    # ---- Feature generation + distillation (spec §4, feature_gen.py) --------------
    seed_label_sample: int = 400          # items the LLM labels to fit the distilled scorer
    reliability_sample_size: int = 100    # items re-scored to estimate test-retest reliability
    distill_method: str = "fewshot"       # "fewshot" (frozen judge prompt) | "embedding"
    proposer_k_candidates: int = 4        # # distinct candidates the proposer returns (pick first non-redundant)
    enable_composite_proposer: bool = False  # §9 fix: also propose 2-primitive boolean composites for gaps

    # ---- Depth dial / pooling (spec §6, depth_dial.py) ----------------------------
    pool_cluster_min_nodes: int = 3       # cluster must span >= this many gap nodes to pool
    pool_cosine_threshold: float = 0.55   # gap-description cosine similarity for clustering
    embedding_model: str = "all-MiniLM-L6-v2"

    # ---- Keep/drop guards (spec §7, guards.py) ------------------------------------
    tau_redundant: float = 0.90           # drop if R^2(new | existing metrics) > tau
    min_deviance_drop_frac: float = 0.10  # gap-closure: deviance at N must drop >= this frac
    # Acceptance gate for a proposed feature (in addition to the redundancy guard):
    #   "gap_closure" -> keep iff it closes the targeted gap (deviance drop); legacy/default.
    #   "auc"         -> keep iff it raises GUARD bank-AUC (label~metric-levels LR, fit on discover)
    #                    by >= min_auc_gain — the direct north-star criterion.
    #   "either"      -> keep iff gap_closure OR auc gain (looser; lets power-improving features in
    #                    even when the local-deviance gate is coarse).
    #   "both"        -> keep iff gap_closure AND auc gain.
    acceptance_mode: str = "gap_closure"
    min_auc_gain: float = 0.005
    # Additional bits gate for the GLOBAL loop (guard log-loss reduction in bits/item must rise
    # by at least this much). 0 disables. The count certificates (notes/2026-07-01__metric-count-
    # certificates.md §4) compose only over bits, so certificate-bearing runs set this > 0.
    min_bits_gain: float = 0.0
    # How the global loop's acceptance gain is measured: "guard" = single guard-split delta
    # (legacy; SE ~0.06 AUC at n_guard=84 — a 0.02 gate is then below noise), "cv" = paired
    # 5-fold CV over pooled discover+guard (split-noise cancels; per-fold pairing shrinks the
    # SE well below a 0.02 gate at n~400). The final power read stays on the untouched test.
    acceptance_eval: str = "guard"
    # ---- Confirm stage (winner's-curse control; MCC §4.4) --------------------------
    # The 2026-07-02 runs showed the primary CV gate alone is anti-conservative under
    # selection: 20 gated proposals -> 1 CV survivor -> dead on the untouched test. Fix:
    # after the primary gate passes, re-estimate the gain with ``confirm_n_repeats`` FRESH-seed
    # paired-CV repeats and require (a) the fresh mean gain to clear the same delta floor and
    # (b) one-sided significance vs 0 under a Nadeau-Bengio-corrected t-test at
    # alpha = gate_alpha / m, m = gate_bonferroni_m (0 -> planned proposals = max_rounds).
    # confirm_n_repeats = 0 disables (legacy). Only active under acceptance_eval="cv".
    confirm_n_repeats: int = 0
    gate_alpha: float = 0.05
    gate_bonferroni_m: int = 0
    # ---- Judge reliability diagnostic (measurement-floor vs genuine-null, 2026-07-05) --------
    # A ~0 gain is ambiguous: either the metric doesn't predict, or the executor can't APPLY it
    # (its scores are noise that attenuates real signal). measure_reliability runs a per-metric
    # test-retest (reliability.judge_test_retest, temp>0, independent draws) and stamps the
    # ledger; the recoverable gain is capped by the retest correlation, so a ~0 gain under low
    # reliability is a measurement floor, NOT a result. Diagnostic only (never a hard drop —
    # tacit/hard-to-apply metrics are still legitimate; reconstruction-only discipline).
    measure_reliability: bool = False
    min_reliability: float = 0.5          # below this the attenuation flag fires
    reliability_temperature: float = 0.6
    # ---- Content-only guard (surface/leakage suppression, 2026-07-05) -----------------------
    # At thin signal scale a weak surface feature (markdown/length/pronoun-count, AUC ~0.52)
    # is competitive and gets surfaced ("Manual Markdown Formatting" won on peer-review). The
    # guard (a) adds an anti-surface instruction to the proposer prompts and (b) drops proposals
    # whose rubric is essentially a surface property. Content = properties of what the text SAYS,
    # not how it is formatted/how long it is.
    content_only_guard: bool = False
    # ---- Residual-contrast bank-quality guard (2026-07-05) ----------------------------------
    # The WRONG/RIGHT residual contrast is only informative if the bank actually predicts y;
    # against a garbage bank (CW medoid contamination) the residuals are random and the contrast
    # cues surface features. Skip the residual contrast when bank AUC < this (0.0 = off/legacy).
    min_bank_auc_for_residual: float = 0.0
    # Viability pre-filter (drop only confidently-degenerate rubrics, never rare-but-predictive
    # ones). Loosened from the earlier appl>0.3 / std>0.1 on 40 items (Codex methodology risk).
    viability_min_applicability: float = 0.10
    viability_min_std: float = 0.05
    # ---- Convergent-proposal dedup (2026-07-08) ----------------------------------------------
    # Paraphrased re-proposals (Clear Resolution x5, Tonal Clarity x3) evade name-dedup and burn
    # gate slots; an embedding match vs this leg's prior proposals skips materialization. The
    # dropped:duplicate status is ITSELF a convergence signal consumed by stage-2 replication.
    dedup_proposals: bool = True
    dedup_cosine: float = 0.86
    # ---- Operationalization (GEPA-style rubric iteration, 2026-07-09) ------------------------
    # Every proposed metric is iterated against LABEL-FREE instrument diagnostics (test-retest,
    # MI-recovery via blind reconstruction, score-distribution collapse) before gate scoring;
    # poor diagnostics trigger a proposer rewrite (max 2). The gate owns all label contact.
    operationalize_proposals: bool = False

    # ---- Outer loop / budget (spec §8, loop.py) -----------------------------------
    max_outer_rounds: int = 6             # hard cap on discover->generate->reinsert rounds
    max_features_per_round: int = 8       # cap features attempted per round (across gap nodes)
    # Role assigned to a newly discovered feature: "feature" keeps it out of the splitting
    # covariates z (clean regions, robust coverage); "both" lets it ALSO become a new splitting
    # variable -> the tree can discover a latent region no context covariate captured.
    discovered_feature_role: str = "feature"

    # ---- LLM backends (pluggable; bulk materialization defaults to local vLLM) -----
    proposer_backend: str = "anthropic"   # "anthropic" | "openai_compatible"
    # env-overridable pointer: `export SONNET_MODEL=claude-sonnet-5` to switch (default unchanged).
    proposer_model: str = field(
        default_factory=lambda: os.environ.get("SONNET_MODEL", "claude-sonnet-4-20250514"))
    materialize_backend: str = "vllm"     # "vllm" (local) | "openai_compatible" | "anthropic"
    materialize_model: str = "Qwen/Qwen3-Coder-Next-FP8"
    # Offline vLLM engine context. 8192 was too small: the residual arm's WRONG/RIGHT contrast
    # prompt overflowed it every round (2026-07-06), silently disabling that arm. 16384 fits the
    # residual prompt with margin on a B200; the proposer also clamps over-long prompts.
    vllm_max_model_len: int = 16384
    vllm_gpu_mem_util: float = 0.93
    openai_base_url: Optional[str] = None  # for openai_compatible backends (vLLM/OpenRouter)
    llm_concurrency: int = 10
    llm_temperature: float = 0.7

    # ---- Judge materialization batching (reused metric_tree scoring path) ----------
    label_batch_size: int = 8192
    max_text_tokens: int = 512

    # ---- I/O ----------------------------------------------------------------------
    output_dir: str = "outputs/metrics_tree_infilling"
    cache_dir: Optional[str] = None       # LabelCache dir; defaults under output_dir
    verbose: bool = True

    # Columns of the labeled corpus (spec §1; y := judgement)
    id_column: str = "id"
    text_column: str = "text"
    label_column: str = "judgement"
    # Extra item columns usable as partitioning covariates z (source/domain/annotator/etc.)
    extra_z_columns: List[str] = field(default_factory=list)
    include_text_length_in_z: bool = True   # add len(text) as a numeric splitting covariate
    # Restrict z to curated ITEM-level axes only (extra_z_columns + text_length), keeping all
    # metrics in X. Offering the whole bank as z inflates the Bonferroni divisor to ~2x
    # n_metrics: on CW a real moderator (source_half, base rates 0.44/0.18) sat at raw
    # p=0.003 but adj_p=0.144 under m_z=48, while curated z split it at adj_p=0.014.
    # partykit's intended design is a small hypothesized-moderator z set.
    curated_z_only: bool = False
