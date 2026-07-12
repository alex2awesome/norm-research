"""Verbatim prompts used by the anchor-free RECOVERY pipeline (label -> recover -> R/T).
Kept as named constants so the validation notebook can AST-extract and display them alongside the
GEPA optimizer prompts — the complete prompt surface to audit before launching all metrics."""

# --- 1. SCORING: how the open-source judge (Llama-70B FP8 on sk3) labels each datapoint with metric p.
#     (rubric = the metric prompt p; concatenated, NOT .format'd, so LaTeX braces never break it.)
SCORING_PROMPT = (
    "{rubric}"
    "\n\nScore the following math answer on the rubric above. A high score means it strongly "
    'satisfies the rubric. Output ONLY JSON: {"score": <one of 0.0,0.25,0.5,0.75,1.0>, '
    '"reason": "<=10 words"}.\n\nANSWER:\n{text}'
)

# --- 2a. RECOVERER (blind, free-text) — the FIRST version. Strong model (Sonnet) sees ONLY (text,label)
#     pairs, infers the metric. Superseded by the data-driven version below (it drifted to "completeness").
RECOVERER_BLIND_PROMPT = (
    "Below are N math.StackExchange answers, each with a numeric label in [0,1] assigned by some hidden "
    "scoring metric. Study the high-labeled vs low-labeled examples and infer what single metric/criterion "
    "best explains the labels. Output ONLY a concise scoring rubric (3-6 sentences) another grader could "
    "use to reproduce these labels: what the metric measures and what makes an answer score 0.0 vs "
    "0.25/0.5/0.75 vs 1.0. No preamble — output the rubric itself. The filename is meaningless; infer "
    "only from the labels."
)

# --- 2b. RECOVERER (data-driven) — the AGREED replacement (2026-06-21). Tests simple feature-hypotheses
#     against the labels and reports correlations BEFORE writing prose, so it can't impose a 'quality' prior.
RECOVERER_DATADRIVEN_PROMPT = (
    "You are given a dataset of (text, label) pairs; the label in [0,1] comes from one hidden metric.\n"
    "A table of precomputed feature<->label correlations is provided: length, n_words, has_url, has_code, "
    "latex_density.\n"
    "STEP 1: report which feature (if any) has |correlation| > 0.5 with the label — that is the dominant "
    "driver. If one does, the metric IS essentially that feature (e.g. 'score by length').\n"
    "STEP 2: if NO feature exceeds 0.5, the metric is semantic — read the high vs low examples and "
    "characterize it (correctness / clarity / elegance / etc.).\n"
    "STEP 3: if the labels match no feature AND show no semantic pattern, say the labels appear random.\n"
    "Output ONLY the recovered scoring rubric (lead with the dominant feature if one exists), 3-6 sentences."
)

# --- 3. SIMILARITY JUDGE — does the recovered metric m_hat match the STATED metric p? (sim-R, a SEPARATE
#     semantic axis from the same-f TVD behavioral R; not expected to obey R<=T.)
SIMILARITY_JUDGE_PROMPT = (
    "You are given two scoring metrics: STATED (what a hidden process was supposed to apply) and RECOVERED "
    "(reverse-engineered blind from the labels it produced). Judge: would these two metrics score a large "
    "set of math answers the SAME way? Score similarity 0.0-1.0 (1.0 = same criterion / near-identical "
    "ranking; 0.5 = overlapping but materially different; 0.0 = unrelated, or one is 'no metric/random'). "
    "Judge by what each MEASURES, not surface wording. Output ONLY JSON: {\"sim\": <0-1>, \"why\": \"<8 words>\"}."
)
