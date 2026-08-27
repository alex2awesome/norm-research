"""
Judge prompt: "Is this paper's primary contribution a novel dataset and/or benchmark?"

Used by sk3_dataset_benchmark_judge.py.

Output format: STRICT JSON, one object per paper:
  {"label": "DATASET" | "BENCHMARK" | "BOTH" | "NEITHER",
   "confidence": "high" | "medium" | "low",
   "reason": "<one-sentence rationale citing the abstract>"}

Definitions:
- DATASET: the paper releases a NEW collection of data that is the headline
  contribution (e.g., ImageNet, LAION-5B, The Pile, MS-MARCO).
- BENCHMARK: the paper releases a NEW evaluation suite / leaderboard /
  protocol whose central purpose is measuring model capability
  (e.g., GLUE, BIG-bench, MMLU, HumanEval).
- BOTH: a single artifact serving both roles (e.g., LAION-5B + CLIP-eval,
  MS-MARCO with eval splits).
- NEITHER: a new method/model/algorithm/theory paper that may *use* existing
  datasets/benchmarks for evaluation but does not contribute one.

The "primary contribution" test is critical. Many papers introduce a small
test set or curated probe as a side artifact for evaluating their method —
those should be NEITHER. The dataset/benchmark must be the *headline*.
"""

SYSTEM_PROMPT = """You are an expert ML research reviewer. Your job is to read a paper's title and abstract and decide whether its PRIMARY CONTRIBUTION is a novel dataset and/or benchmark.

Definitions:
- **DATASET**: The paper's headline contribution is releasing a NEW collection of data (corpus, image set, video set, conversations, etc.) intended for training or evaluation by the community. Examples: ImageNet, LAION-5B, The Pile, C4, MS-MARCO.
- **BENCHMARK**: The paper's headline contribution is a NEW evaluation suite — a curated set of tasks/queries/probes whose purpose is measuring model capabilities, with a defined evaluation protocol and (often) a leaderboard. Examples: GLUE, SuperGLUE, MMLU, BIG-bench, HumanEval, HELM.
- **BOTH**: The paper releases a single artifact that simultaneously serves as a dataset AND a benchmark with a defined eval protocol (e.g., MS-MARCO with eval splits + leaderboard).
- **NEITHER**: A new method, model, algorithm, theory, or analysis paper that may evaluate on existing datasets/benchmarks but does not contribute a new one as its primary deliverable.

CRITICAL: The "primary contribution" test. Many method papers introduce a small probe set or curated test cases as a SIDE artifact to evaluate their proposed method — those should be NEITHER. The dataset/benchmark must be the paper's HEADLINE contribution, not a supporting evaluation tool.

If the paper's main verb is "we propose/introduce/develop a method/model/architecture/algorithm/theory/framework" → almost always NEITHER, even if it builds a small evaluation set.

If the paper's main verb is "we release/present/introduce a dataset/benchmark/corpus/test suite" → DATASET, BENCHMARK, or BOTH.

Edge cases:
- New training corpus → DATASET
- New eval suite drawing from existing data → BENCHMARK
- A "probing dataset" used mainly to test one specific method → NEITHER (it's a method paper)
- Survey/analysis of existing benchmarks → NEITHER
- A challenge/competition paper → BENCHMARK
- A model card / data card → NEITHER (process artifact)

Output STRICT JSON only — no prose, no markdown fences:
{"label": "DATASET" | "BENCHMARK" | "BOTH" | "NEITHER", "confidence": "high" | "medium" | "low", "reason": "<one sentence quoting the abstract's primary-contribution claim>"}
"""

USER_TEMPLATE = """Title: {title}

Abstract: {abstract}

Classify this paper's primary contribution. Output only the JSON object."""


def build_messages(title: str, abstract: str):
    """Build chat messages for one paper. Truncates abstract to 3000 chars."""
    abstract = (abstract or "").strip()
    if len(abstract) > 3000:
        abstract = abstract[:3000] + "…"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(
            title=(title or "").strip()[:400],
            abstract=abstract,
        )},
    ]
