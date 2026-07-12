"""ImplementerConfig — backends per role, sample sizes, budget caps, objective weights."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class BudgetCaps:
    """Resource caps enforced by the optimizer — the scaling-law axes (design §3).

    A candidate artifact exceeding a cap is rejected before scoring (logged, not silently
    dropped). ``None`` = uncapped.
    """

    instruction_tokens: Optional[int] = None     # axis 1: rubric/definition length
    n_fewshots: Optional[int] = None             # axis 2: worked examples
    data_budget: Optional[int] = None            # axis 3: items visible to optimizer/reconstructor
    optimizer_rounds: Optional[int] = None       # axis 6: GEPA rounds
    code_ast_nodes: Optional[int] = None         # axis 7: structural complexity (code kind)
    inference_passes: int = 1                    # axis 5: ensemble k at scoring time

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "instruction_tokens", "n_fewshots", "data_budget",
            "optimizer_rounds", "code_ast_nodes", "inference_passes")}


@dataclass
class ImplementerConfig:
    task: str = "code-review"
    output_dir: str = str(REPO_ROOT / "outputs" / "metric_implementer")

    # ---- task framing (guards the v2 cross-task prompt bug: every role prompt that
    # mentions the item type derives from these fields, never from a hardcoded string) --
    domain_plural: str = "competitive-programming solutions"   # judge system prompt
    item_label: str = "SOLUTION"                               # judge template header
    item_noun: str = "solution"                                # CF/oracle prompts
    cf_keep_clause: str = "keep the algorithm and structure"   # CF edit invariant
    cf_hold_default: str = "comments, identifier style, and length"
    recon_item_phrase: str = "code-solution excerpts"          # reconstructor framing

    # ---- LLM roles (OpenRouter slugs; key from ~/.openrouter-api-key.txt) -----------
    # Cross-family rule (README): acceptance-time reconstructor/generator must be a
    # DIFFERENT family from the judge. In-loop roles may share a family.
    backend: str = "openrouter"
    judge_model: str = "meta-llama/llama-3.1-8b-instruct"
    reviser_model: str = "meta-llama/llama-3.3-70b-instruct"
    reconstructor_model: str = "meta-llama/llama-3.3-70b-instruct"
    acceptance_reconstructor_model: str = "qwen/qwen-2.5-72b-instruct"   # different family
    generator_model: str = "meta-llama/llama-3.3-70b-instruct"           # counterfactual edits
    acceptance_generator_model: str = "qwen/qwen-2.5-72b-instruct"       # different family
    grader_model: str = "meta-llama/llama-3.3-70b-instruct"              # semantic match 1-5

    # Strong-proposer / weak-judge asymmetry (our actual system): the ORACLE is a strong
    # model that scores a small fixed sample by the metric's canonical DESCRIPTION (not the
    # candidate prompt), giving a stable distillation target. A new fidelity component
    # measures how well (weak judge + candidate prompt) reproduces the oracle. Optimizing
    # at a FIXED weak judge tier produces one point of the judge-capability scaling axis;
    # sweeping --judge-models in run_trial traces the whole curve.
    # env-overridable pointer (OpenRouter slug form, so its own var):
    # `export ORACLE_MODEL=anthropic/claude-sonnet-5` to switch (default unchanged).
    oracle_model: str = field(
        default_factory=lambda: os.environ.get("ORACLE_MODEL", "anthropic/claude-sonnet-4.5"))
    n_oracle_items: int = 30                 # 0 disables the oracle component
    w_oracle: float = 0.45                   # upweighted 2026-06-23 (accuracy term; cf leniency fix)

    # Cross-family SAME-TIER executor for the reconstruction TVD-MI: re-execute the induced
    # rubric with a DIFFERENT model family at the judge's tier, so reconstruction recovery does
    # not count bias the original judge would share with a same-model executor. None -> the
    # original judge executes (in-loop / 1-GPU default; needs a second resident model when set).
    cross_executor_model: Optional[str] = None

    llm_concurrency: int = 8
    judge_temperature: float = 0.7      # >0 so test-retest reliability is non-trivial
    other_temperature: float = 0.7
    max_retries: int = 3                # detect-bad-output-and-retry; never repetition_penalty
    request_timeout_s: int = 120

    # ---- measure sample sizes (trial-scale; scale up on sk3/vLLM later) -------------
    # 2 passes x 24 items gave rho estimates swinging -0.01..0.47 across sibling candidates
    # (2026-06-11 run) — too noisy to rank mutations on. 3 passes x 32 items = 3 pairwise
    # correlations on a bigger sample; still cheap (judge is the 8B).
    n_reliability_items: int = 32
    reliability_passes: int = 3
    # Decorrelate FORMAT/position bias between reliability passes: when True, each pass after the
    # first scores a meaning-preserving format-perturbed rubric (reordered bullets, whitespace,
    # neutral framing), so reliability TVD-MI reflects content reproducibility, not shared layout
    # artifacts of the identical prompt. Default off (keeps pure same-prompt test-retest).
    reliability_format_perturb: bool = False
    n_reconstruct_label_items: int = 48      # items the judge labels for the reconstructor
    n_reconstruct_shown: int = 16            # stratified (text,score) pairs shown
    n_reconstruct_behavioral: int = 24       # held-out items for behavioral match
    n_cf_base_texts: int = 8                 # base items edited into counterfactual pairs
    n_consistency_items: int = 12
    n_convergence_items: int = 60            # code<->judge convergence sample
    n_disagreement_keep: int = 10            # worst |code - judge| items persisted as queue

    # ---- optimizer ------------------------------------------------------------------
    n_mutations: int = 3
    pool_keep_k: int = 2
    default_rounds: int = 2
    # fidelity scalarization (predictive contribution is structurally absent here).
    # Reweighted 2026-06-23 against the leniency attractor: upweight the ACCURACY/signal terms
    # (cf, recon, oracle, disc) and downweight reliability (which rewards test-retest STABILITY —
    # a constant-YES prompt scores rho~1 with zero signal, the leniency fixed point). The
    # discrimination term (w_disc) rewards the prompt VARYING across items, directly countering the
    # attractor; disc_floor/discrimin-promotability gate also block degenerate prompts.
    w_cf: float = 0.45
    w_recon: float = 0.45
    w_reliability: float = 0.05
    w_consistency: float = 0.05
    w_disc: float = 0.55                 # up 0.40→0.55 (2026-06-23): T_prose is the BINDING cap on
    # I(M,M_ω) (it sat at 97% of T_prose=0.072), so discrimination is the lever that raises the
    # decomposition number. Paired with warm-starting GEPA from a liked p̂ so the boost pushes
    # discrimination, not off-task length/lexicon shortcuts (the accuracy terms cf+recon+oracle
    # still aggregate larger, so a wrong-but-spread prompt cannot win).
    reliability_floor: float = 0.5      # hard floor; candidates below are not promotable
    disc_floor: float = 0.03            # below this T, the prompt is a degenerate leniency attractor
    disc_healthy: float = 0.15          # T at which disc_norm saturates to 1.0 (gradient to T=0.15)

    # ---- trial corpus ----------------------------------------------------------------
    pool_path: str = str(REPO_ROOT / "methods" / "metric_implementer" / "trial"
                         / "pool_competitive_code.jsonl.gz")
    text_column: str = "code"
    id_column: str = "candidate_id"
    max_text_chars: int = 4000
    random_seed: int = 7

    def registry_dir(self) -> Path:
        return Path(self.output_dir) / self.task / "registry"

    def runs_dir(self) -> Path:
        return Path(self.output_dir) / self.task / "runs"


# Per-task overrides: framing fields + pool. Defaults above WERE the code-review preset
# (competitive-programming trial pool). 2026-07-05: the R3 domain sweeps score the dense
# PR corpus (code_review_dense_4096tok.csv.gz — real GitHub pull requests), so the preset
# now frames items as pull requests; the competitive-code defaults remain only for the
# old pilot pool.
TASK_PRESETS: Dict[str, dict] = {
    "code-review": dict(
        task="code-review", domain_plural="GitHub pull requests under code review",
        item_label="PULL_REQUEST", item_noun="pull request",
        cf_keep_clause="keep the code change and its purpose",
        cf_hold_default="length, diff formatting, and identifier style",
        recon_item_phrase="pull-request excerpts", text_column="text"),
    "creative-writing": dict(
        task="creative-writing",
        domain_plural="short fiction pieces",
        item_label="STORY",
        item_noun="story",
        cf_keep_clause="keep the plot, characters, and events",
        cf_hold_default="plot, characters, length, and paragraph structure",
        recon_item_phrase="story excerpts",
        pool_path=str(REPO_ROOT / "methods" / "metric_implementer" / "trial"
                      / "pool_creative_writing.jsonl.gz"),
        text_column="text",
    ),
    "law": dict(
        task="law",
        domain_plural="fact sections of federal employment-discrimination opinions",
        item_label="FACTS",
        item_noun="fact section",
        cf_keep_clause="keep the underlying events, parties, and claims",
        cf_hold_default="length, party names, and dates",
        recon_item_phrase="legal fact-section excerpts",
        pool_path=str(REPO_ROOT / "methods" / "metric_implementer" / "trial"
                      / "pool_law.jsonl.gz"),
        text_column="facts",
        id_column="doc_id",
    ),
    # scale-out task presets (correct judge framing per domain — guards the cross-task
    # framing bug [[feedback_judge_prompt_cross_task_bug]]). pool_path is set per-run from
    # the manifest, so it is left at the code-review default and overridden by load_corpus.
    # NB (2026-07-05 framing fix, pre-first-sweep): the peer-review corpus text is the
    # PAPER submission (splits/train.csv.gz, judgement = accept), not the review text —
    # the earlier "REVIEW" framing would have hit the exact cross-task bug this block guards.
    "peer-review": dict(
        task="peer-review", domain_plural="academic paper submissions under peer review",
        item_label="PAPER", item_noun="paper",
        cf_keep_clause="keep the research contribution and the claims",
        cf_hold_default="length, section structure, and writing style",
        recon_item_phrase="academic-paper excerpts", text_column="text"),
    "news-homepages": dict(
        task="news-homepages", domain_plural="news article headlines/blurbs as placed on a homepage",
        item_label="ARTICLE", item_noun="article placement",
        cf_keep_clause="keep the article topic and facts",
        cf_hold_default="length, outlet, and topic",
        recon_item_phrase="news homepage item excerpts", text_column="text"),
    "math": dict(
        task="math", domain_plural="math Stack Exchange answers",
        item_label="ANSWER", item_noun="answer",
        cf_keep_clause="keep the mathematical content and the question being answered",
        cf_hold_default="length, notation style, and formatting",
        recon_item_phrase="math answer excerpts", text_column="text"),
    "humor": dict(
        task="humor", domain_plural="short humorous texts (jokes/standup/captions)",
        item_label="JOKE", item_noun="joke",
        cf_keep_clause="keep the premise and the topic",
        cf_hold_default="length, setup, and vocabulary",
        recon_item_phrase="joke excerpts", text_column="text"),
    "patents": dict(
        task="patents", domain_plural="patent application texts (abstract + claims)",
        item_label="PATENT", item_noun="patent text",
        cf_keep_clause="keep the invention and the claim scope",
        cf_hold_default="length, claim count, and terminology",
        recon_item_phrase="patent text excerpts", text_column="text"),
    "notice-and-comment": dict(
        task="notice-and-comment", domain_plural="public comments on proposed federal rules",
        item_label="COMMENT", item_noun="comment",
        cf_keep_clause="keep the rule being commented on and the commenter's position",
        cf_hold_default="length, tone, and citations",
        recon_item_phrase="public-comment excerpts", text_column="text"),
    "press-releases": dict(
        task="press-releases", domain_plural="corporate/organizational press releases",
        item_label="PRESS_RELEASE", item_noun="press release",
        cf_keep_clause="keep the announcement and the key facts",
        cf_hold_default="length, formatting, and boilerplate",
        recon_item_phrase="press release excerpts", text_column="text"),
    # key bridge: the R3 hierarchy files are named math-stackexchange_* while the original
    # preset/manifest key is "math" — this alias lets run_alpha_probe --task math-stackexchange
    # resolve preset + hierarchy + corpus consistently (same corpus as "math").
    "math-stackexchange": dict(
        task="math-stackexchange", domain_plural="math Stack Exchange answers",
        item_label="ANSWER", item_noun="answer",
        cf_keep_clause="keep the mathematical content and the question being answered",
        cf_hold_default="length, notation style, and formatting",
        recon_item_phrase="math answer excerpts", text_column="text"),
    # key bridge like math-stackexchange: hierarchy files are legal-outcome-prediction_*
    # while the original preset/manifest key is "law". Probes = title_vii fact sections.
    "legal-outcome-prediction": dict(
        task="legal-outcome-prediction",
        domain_plural="fact sections of federal employment-discrimination opinions",
        item_label="FACTS", item_noun="fact section",
        cf_keep_clause="keep the underlying events, parties, and claims",
        cf_hold_default="length, party names, and dates",
        recon_item_phrase="legal fact-section excerpts", text_column="facts"),
    "grant-funding": dict(
        task="grant-funding", domain_plural="research grant proposals",
        item_label="PROPOSAL", item_noun="proposal",
        cf_keep_clause="keep the proposed research and the funding case",
        cf_hold_default="length, formatting, and boilerplate headers",
        recon_item_phrase="grant-proposal excerpts", text_column="full_text"),
}


def apply_task_preset(cfg: "ImplementerConfig", task: str) -> "ImplementerConfig":
    for k, v in TASK_PRESETS[task].items():
        setattr(cfg, k, v)
    return cfg
